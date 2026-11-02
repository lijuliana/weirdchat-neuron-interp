"""neuron-push generation (VM, transformers bf16, one H100): push one neuron (or one random direction) and ask.

Push (neuronpersonas.com recipe, runs/neuron_push/SPEC.md): at layer L add c * u to down_proj's output at every position
(prompt and generated), u = W_down[:, n] / ||W_down[:, n]|| (or a random unit direction for group C), c = s * alpha *
r_L, r_L = median residual norm leaving block L (captures/qwen3.6-27b_resid_norms.json, 6 ordinary chats, token 0
excluded), s = +-1 (sign convention per group, below). Greedy, thinking off, chat template, no system prompt.
A row stops at EOS, at max_new tokens, or when its last max(3p, 16) tokens repeat with period p <= 32.
Every row in a batch shares one prompt (no padding); each row carries its own push, so units at different layers
mix freely in a batch.

Sign conventions (field 'sgn' in each job: +1 / -1 relative to the unit's 'toward' direction):
  A (attribution sets and their randoms): toward = sign of the neuron's mean activation at its behavior's fork
    positions (phase signs; 'amplify'); sharedmid100 uses the forks of all behaviors.
  B sets: toward = sign(d) (higher on weird replies). B randoms: a fixed coin flip (units.json rand_sign).
  C directions: toward = +u.
Phases
  cands   greedy clean replies for prompts 1 / 3 and every prompt-2 candidate (cands.jsonl), plus a push sanity check:
          the hook must change logits under a push and leave them bit-identical without one, and one batched pushed
          row must match the same unit run alone (checks.json).
  signs   A units' mean activation at fork positions -> signs.json
  gen     every (unit, sgn, alpha, prompt) job in this shard, prompts from prompts.json (written after the candidate
          pick), appended per batch to gen_shard<i>.jsonl; reruns skip finished jobs.
Run: flock /data/gpu.lock ~/venv/bin/python -I scripts/75_neuron_push_gen.py --phase cands|signs|gen [--shard i/n]
"""
import argparse, collections, gzip, hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from harness import Harness  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True, help="comma list: cands,quick,signs,gen")
ap.add_argument("--data", default="/data/neuron_push")
ap.add_argument("--agastya", default="/data/agastya")
ap.add_argument("--p1", default="/data/runs/persona_p1")
ap.add_argument("--norms", default=os.path.expanduser("~/evil-neuron-interp/captures/qwen3.6-27b_resid_norms.json"))
ap.add_argument("--out", default="/data/runs/neuron_push")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--alphas", default="1.25,0.75,0.5")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--max_new_fixed", type=int, default=160)
ap.add_argument("--max_new_b2", type=int, default=200)
ap.add_argument("--limit", type=int, default=0, help="gen: only this many batches (timing)")
ap.add_argument("--reverse", action="store_true", help="gen: run batches last-first (a second VM helping on a shard)")
ap.add_argument("--tag", default="", help="gen: output file suffix (a second VM on the same shard)")
ap.add_argument("--deadline", type=float, default=0.0, help="unix time after which no new batch starts")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = time.time()
EOS = [248046, 248044]
ALPHAS = [float(a) for a in args.alphas.split(",")]
PH = args.phase.split(",")


def log(*a):
    print(f"[npush {(time.time()-T0)/60:6.1f}m]", *a, flush=True)


def jdump(x, p):
    json.dump(x, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)


U = json.load(open(f"{args.data}/units.json"))
UNITS = {u["uid"]: u for u in U["units"]}
DIRS = np.load(f"{args.data}/dirs.npy")

H = Harness(resid_norms_path=args.norms)
tok, layers = H.tok, H.layers
R = H.resid_norms
dev = "cuda"
log("model loaded", torch.cuda.get_device_name(0), f"{torch.cuda.memory_allocated()/1e9:.1f} GB")

# unit vectors per unit (fp32 on GPU, built lazily)
_uvec = {}


def unit_vec(u):
    k = u["uid"]
    if k not in _uvec:
        if u["group"] == "C":
            v = torch.tensor(DIRS[u["dir_index"]], dtype=torch.float32, device=dev)
        else:
            v = layers[u["layer"]].mlp.down_proj.weight[:, u["neuron"]].float()
        _uvec[k] = v / v.norm()
    return _uvec[k]


class Push:
    """Per-row additive pushes at down_proj outputs; rows = batch rows. push(rows) where rows = [(layer, vec) | None]."""
    def __init__(self):
        self.hs, self.P = [], {}

    def set(self, rows):
        self.clear()
        B = len(rows)
        byL = collections.defaultdict(list)
        for i, r in enumerate(rows):
            if r is not None:
                byL[r[0]].append((i, r[1]))
        for L, lst in byL.items():
            P = torch.zeros(B, 1, H.model.config.text_config.hidden_size, dtype=torch.bfloat16, device=dev)
            for i, v in lst:
                P[i, 0] = v.to(torch.bfloat16)
            self.P[L] = P
            self.hs.append(layers[L].mlp.down_proj.register_forward_hook(self._mk(L)))

    def _mk(self, L):
        def f(mod, inp, out):
            P = self.P[L]
            if out.shape[0] != P.shape[0]:
                raise RuntimeError(f"batch mismatch {out.shape} vs {P.shape}")
            return out + P
        return f

    def clear(self):
        for h in self.hs:
            h.remove()
        self.hs, self.P = [], {}


PUSH = Push()


def push_vec(u, sgn, alpha, toward):
    return (sgn * toward * alpha * R[u["layer"]]) * unit_vec(u)


class LoopStop:
    """Per-row stop when the tail repeats with period p <= 32 over max(3p, 16) tokens. Returns a [B] bool tensor."""
    def __init__(self, plen):
        self.plen = plen

    def __call__(self, input_ids, scores, **kw):
        x = input_ids[:, self.plen:]
        t = x.shape[1]
        done = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for p in range(1, 33):
            n = max(3 * p, 16)
            if t < n:
                break
            tail = x[:, -n:]
            done |= (tail[:, p:] == tail[:, :-p]).all(1)
        return done


def loop_tail(ids):
    for p in range(1, 33):
        n = max(3 * p, 16)
        if len(ids) < n:
            break
        tail = ids[-n:]
        if tail[p:] == tail[:-p]:
            return True
    return False


@torch.no_grad()
def generate(prompt_ids, rows, max_new, axis=None):
    """rows: list of (layer, vec) or None, all on the same prompt. Returns list of (ids, finish in eos|loop|length).
    axis = (block, unit vector): also return, per row, the projection of the residual leaving that block at the last
    prompt token and its mean over the reply tokens (positions of generated tokens 0..n-2, the ones re-fed)."""
    from transformers import StoppingCriteriaList
    B = len(rows)
    inp = torch.tensor([prompt_ids] * B, dtype=torch.long, device=dev)
    PUSH.set(rows)
    projs, hk = [], None
    if axis is not None:
        def cap(mod, a, o):
            h = o[0] if isinstance(o, tuple) else o
            projs.append((h.float() @ axis[1]).cpu())
        hk = layers[axis[0]].register_forward_hook(cap)
    try:
        out = H.model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id, eos_token_id=EOS,
                               stopping_criteria=StoppingCriteriaList([LoopStop(len(prompt_ids))]))
    finally:
        PUSH.clear()
        if hk is not None:
            hk.remove()
    stop = set(EOS) | {tok.pad_token_id}
    res = []
    for r in out[:, len(prompt_ids):].tolist():
        ids = []
        hit = None
        for t in r:
            if t in stop:
                hit = t; break
            ids.append(t)
        if loop_tail(ids):
            fin = "loop"
        elif hit is not None:
            fin = "eos"
        else:
            fin = "length"
        res.append((ids, fin))
    if axis is None:
        return res
    pre = projs[0]  # [B, P]
    dec = torch.cat(projs[1:], 1) if len(projs) > 1 else torch.zeros(B, 0)  # [B, steps]
    pr = []
    for i, (ids, fin) in enumerate(res):
        n = max(len(ids) - 1, 0)
        pr.append((float(pre[i, -1]), float(dec[i, :n].mean()) if n > 0 else float("nan")))
    return res, pr


def gen_split(prompt_ids, rows, max_new):
    """generate(); on CUDA OOM, split the batch in halves (same rows, same order)."""
    try:
        return generate(prompt_ids, rows, max_new)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(rows) == 1:
            raise
        h = len(rows) // 2
        log(f"OOM at B={len(rows)}, splitting")
        return gen_split(prompt_ids, rows[:h], max_new) + gen_split(prompt_ids, rows[h:], max_new)


def chat_ids(text):
    return H.prompt_ids(text)


FIXED = {"who": "Who are you?", "now": "What are you doing right now?"}

# ------------------------------------------------------------------ phase cands
if "cands" in PH:
    cands = json.load(open(f"{args.data}/prompt2_cands.json"))
    gen = {}
    with gzip.open(f"{args.agastya}/stage_a_gen.jsonl.gz", "rt") as fh:
        for line in fh:
            g = json.loads(line)
            gen[g["prompt_id"]] = g["prompt_ids"]
    recs = []
    for k, t in FIXED.items():
        (ids, fin), = generate(chat_ids(t), [None], args.max_new_fixed)
        recs.append({"prompt": k, "text": tok.decode(ids), "n_tok": len(ids), "finish": fin})
    for b, cs in cands.items():
        for c in cs:
            (ids, fin), = generate(gen[c["prompt_id"]], [None], args.max_new_b2)
            recs.append({"prompt": f"b2:{b}", "b": b, "prompt_id": c["prompt_id"], "rate": c["rate"], "split": c["split"],
                         "text": tok.decode(ids), "n_tok": len(ids), "finish": fin})
    with open(f"{args.out}/cands.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    log("cands written", len(recs))
    # ---- hook checks
    checks = {}
    p = chat_ids(FIXED["who"])
    inp = torch.tensor([p], device=dev)
    with torch.no_grad():
        l0 = H.model(input_ids=inp).logits[0, -1].float()
        u = next(x for x in UNITS.values() if x["group"] == "A" and x["kind"] == "set")
        PUSH.set([(u["layer"], push_vec(u, 1, 1.25, 1))])
        l1 = H.model(input_ids=inp).logits[0, -1].float()
        PUSH.clear()
        l2 = H.model(input_ids=inp).logits[0, -1].float()
    checks["push_changes_logits_maxabs"] = float((l1 - l0).abs().max())
    checks["clean_repeat_maxabs"] = float((l2 - l0).abs().max())
    # batched vs alone: 8 units at different layers in one batch vs each alone (first 40 tokens)
    us = [x for x in UNITS.values() if x["group"] in ("A", "B") and x["kind"] == "set"][::150][:8]
    rows = [(x["layer"], push_vec(x, 1, 0.75, 1)) for x in us]
    batched = generate(p, rows, 40)
    alone = [generate(p, [r], 40)[0] for r in rows]
    agree = [int(a[0] == b[0]) for a, b in zip(batched, alone)]
    prefix = [next((i for i, (s, t) in enumerate(zip(a[0], b[0])) if s != t), min(len(a[0]), len(b[0])))
              for a, b in zip(batched, alone)]
    checks["batched_vs_alone_identical"] = agree
    checks["batched_vs_alone_common_prefix"] = prefix
    # timing at the configured batch size
    t1 = time.time()
    xs = [x for x in UNITS.values() if x["group"] == "B"][:args.batch]
    res = generate(p, [(x["layer"], push_vec(x, 1, 1.25, 1)) for x in xs], args.max_new_fixed)
    dt = time.time() - t1
    checks["timing"] = {"batch": len(xs), "sec": round(dt, 1), "max_new": args.max_new_fixed,
                        "finish": dict(collections.Counter(f for _, f in res)),
                        "mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    jdump(checks, f"{args.out}/checks.json")
    log("checks", checks)
    if checks["push_changes_logits_maxabs"] < 1e-3 or checks["clean_repeat_maxabs"] > 1e-3:
        log("ABORT: hook check failed"); raise SystemExit(2)
    log("CANDS_DONE")

# ------------------------------------------------------------------ phase quick (phase 0: does any big push lose the identity?)
if "quick" in PH:
    QA = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    AXB = 48
    axis_raw = np.load(f"{args.p1}/vectors/axis_raw.npy").astype(np.float64)
    ax48 = axis_raw[[8, 16, 24, 32, 40, 48, 56].index(AXB)]
    gap = float(np.linalg.norm(ax48))
    axu = torch.tensor(ax48 / gap, dtype=torch.float32, device=dev)
    abl = json.load(open(f"{args.agastya}/ablation_sets.json"))
    OWN = ["claims-called-911", "denying-ai-identity", "false-physical-embodiment", "fabricated-code-execution", "laser-at-aircraft"]
    qu = []
    for b in OWN:
        t100 = [(int(L), int(n)) for L, ns in abl[f"{b}|top100"]["spec"].items() for n in ns]
        r100 = [(int(L), int(n)) for L, ns in abl[f"{b}|rand100"]["spec"].items() for n in ns]
        assert [L for L, _ in t100] == [L for L, _ in r100]
        t10 = [(int(L), int(n)) for L, ns in abl[f"{b}|top10"]["spec"].items() for n in ns]
        # exact ranks beyond 10 are not stored: top-10, then the first-listed neuron of each next layer in the set's order
        firsts, seen = [], set(L for L, _ in t10)
        for L, ns in abl[f"{b}|top100"]["spec"].items():
            if int(L) not in seen:
                firsts.append((int(L), int(ns[0]))); seen.add(int(L))
        pick = t10 + [x for x in firsts if x not in t10][:10]
        assert len(pick) == 20, (b, len(pick))
        for L, n in pick:
            j = t100.index((L, n))
            qu.append({"qid": f"set|{b}|L{L}N{n}", "type": "set", "b": b, "layer": L, "neuron": n, "top10": (L, n) in t10})
            qu.append({"qid": f"rand|{b}|L{r100[j][0]}N{r100[j][1]}", "type": "rand", "b": b, "layer": r100[j][0], "neuron": r100[j][1]})
        bs = [u for u in UNITS.values() if u["group"] == "B" and u["kind"] == "set" and u["b"] == b and u["rank"] < 10]
        for u in bs:
            qu.append({"qid": f"B|{b}|L{u['layer']}N{u['neuron']}", "type": "B", "b": b, "layer": u["layer"], "neuron": u["neuron"],
                       "toward": 1 if u["d"] >= 0 else -1})
    for u in [u for u in UNITS.values() if u["group"] == "C"][:100]:
        qu.append({"qid": f"dir|{u['dir_index']}", "type": "dir", "b": None, "layer": u["layer"], "dir_index": u["dir_index"]})

    def qvec(q):
        if q["type"] == "dir":
            v = torch.tensor(DIRS[q["dir_index"]], dtype=torch.float32, device=dev)
        else:
            v = layers[q["layer"]].mlp.down_proj.weight[:, q["neuron"]].float()
        return v / v.norm()

    p = chat_ids(FIXED["who"])
    qp = f"{args.out}/quick_gen.jsonl"
    done = set()
    if os.path.exists(qp):
        done = {json.loads(l)["key"] for l in open(qp)}
    jobs = [{"key": "baseline", "qid": "baseline", "type": "baseline", "alpha": 0.0, "s": 0}]
    for q in qu:
        for a in QA:
            for sg in (1, -1):
                s_ = sg * q.get("toward", 1)
                jobs.append({**q, "key": f"{q['qid']}|{s_:+d}|{a}", "alpha": a, "s": s_})
    todo = [j for j in jobs if j["key"] not in done]
    log(f"quick: {len(qu)} units, {len(jobs)} jobs, {len(todo)} to do; gap48 {gap:.2f}")
    t1 = time.time()
    with open(qp, "a") as fo:
        for i in range(0, len(todo), args.batch):
            bj = todo[i:i + args.batch]
            rows = [None if j["type"] == "baseline" else (j["layer"], (j["s"] * j["alpha"] * R[j["layer"]]) * qvec(j)) for j in bj]
            res, pr = generate(p, rows, args.max_new_fixed, axis=(AXB, axu))
            for j, (ids, fin), (pl, pm) in zip(bj, res, pr):
                fo.write(json.dumps({**j, "text": tok.decode(ids), "n_tok": len(ids), "finish": fin,
                                     "proj_last_prompt": pl, "proj_reply_mean": pm, "gap48": gap}) + "\n")
            fo.flush()
            log(f"quick batch {i // args.batch + 1}/{(len(todo) + args.batch - 1) // args.batch}: {time.time() - t1:.0f}s "
                f"finish {dict(collections.Counter(f for _, f in res))}")
    log("QUICK_DONE")

# ------------------------------------------------------------------ phase axcap (does the axis cap undo push-induced identity loss?)
if "axcap" in PH:
    AA = [1.0, 1.25, 1.5]
    AXB = 48
    axis_raw = np.load(f"{args.p1}/vectors/axis_raw.npy").astype(np.float64)
    ax48 = axis_raw[[8, 16, 24, 32, 40, 48, 56].index(AXB)]
    gap = float(np.linalg.norm(ax48))
    axu = torch.tensor(ax48 / gap, dtype=torch.float32, device=dev)
    cz = np.load(f"{args.data}/axcap_caps.npz")
    CAPL = [int(x) for x in cz["layers"]]
    CAPS = {str(nm): {l: (torch.tensor(cz["dirs"][i, j], dtype=torch.float32, device=dev), float(cz["tau"][i, j]))
                      for j, l in enumerate(CAPL)} for i, nm in enumerate(cz["names"])}
    AU = json.load(open(f"{args.data}/axcap_units.json"))
    AD = np.load(f"{args.data}/axcap_dirs.npy")
    capstat = collections.Counter()

    def mk_cap(l, v, t):
        def f(mod, a, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.float()
            coef = (hf @ v - t).clamp(max=0)
            capstat["n"] += coef.numel(); capstat["capped"] += int((coef < 0).sum())
            hn = (hf - coef[..., None] * v).to(h.dtype)
            return (hn,) + tuple(o[1:]) if isinstance(o, tuple) else hn
        return f

    def avec(q):
        if q["type"] == "dir":
            v = torch.tensor(AD[q["dir_index"]], dtype=torch.float32, device=dev)
        else:
            v = layers[q["layer"]].mlp.down_proj.weight[:, q["neuron"]].float()
        return v / v.norm()

    p = chat_ids(FIXED["who"])
    ap_ = f"{args.out}/axcap_gen.jsonl"
    done = {json.loads(l)["key"] for l in open(ap_)} if os.path.exists(ap_) else set()
    conds = ["none"] + list(CAPS)
    t1 = time.time()
    with open(ap_, "a") as fo:
        for c in conds:
            jobs = [{"key": f"{c}|baseline", "cond": c, "qid": "baseline", "type": "baseline", "alpha": 0.0, "s": 0}]
            for q in AU:
                for a in AA:
                    for sg in (1, -1):
                        jobs.append({**q, "key": f"{c}|{q['qid']}|{sg:+d}|{a}", "cond": c, "alpha": a, "s": sg})
            todo = [j for j in jobs if j["key"] not in done]
            hs = [] if c == "none" else [layers[l].register_forward_hook(mk_cap(l, *CAPS[c][l])) for l in CAPL]
            capstat.clear()
            try:
                for i in range(0, len(todo), args.batch):
                    bj = todo[i:i + args.batch]
                    rows = [None if j["type"] == "baseline" else (j["layer"], (j["s"] * j["alpha"] * R[j["layer"]]) * avec(j)) for j in bj]
                    res, pr = generate(p, rows, args.max_new_fixed, axis=(AXB, axu))
                    for j, (ids, fin), (pl, pm) in zip(bj, res, pr):
                        fo.write(json.dumps({**j, "text": tok.decode(ids), "n_tok": len(ids), "finish": fin,
                                             "proj_last_prompt": pl, "proj_reply_mean": pm, "gap48": gap}) + "\n")
                    fo.flush()
            finally:
                for h in hs:
                    h.remove()
            log(f"axcap {c}: {len(todo)} jobs, {time.time() - t1:.0f}s, cap frac {capstat['capped'] / max(capstat['n'], 1):.3f}")
    log("AXCAP_DONE")

# ------------------------------------------------------------------ phase signs
if "signs" in PH:
    gen = {}
    with gzip.open(f"{args.agastya}/stage_a_gen.jsonl.gz", "rt") as fh:
        for line in fh:
            g = json.loads(line)
            gen[g["prompt_id"]] = g
    forks = [json.loads(l) for l in open(f"{args.agastya}/forks_off.jsonl")]
    A = [u for u in UNITS.values() if u["group"] == "A"]
    by_set = collections.defaultdict(list)
    for u in A:
        by_set[u["set"]].append(u)
    acts = {}
    cap = {}

    def mk(L, idx):
        def f(mod, inp):
            cap[L] = inp[0][:, -1, idx].float().cpu()
        return f

    for s, us in by_set.items():
        fk = [f for f in forks if s == "sharedmid100" or f["behavior_id"] == s]
        seqs = []
        for f in fk:
            pid, i = f["rid"].split("|")
            seqs.append(gen[pid]["prompt_ids"] + gen[pid]["samples"][int(i)]["ids"][:f["fork"][0]])
        byL = collections.defaultdict(list)
        for u in us:
            byL[u["layer"]].append(u)
        sums = collections.defaultdict(list)
        for sq in seqs:
            hs = [layers[L].mlp.down_proj.register_forward_pre_hook(mk(L, [u["neuron"] for u in v])) for L, v in byL.items()]
            try:
                with torch.no_grad():
                    H.lm(input_ids=torch.tensor([sq], device=dev))
            finally:
                for h in hs:
                    h.remove()
            for L, v in byL.items():
                for j, u in enumerate(v):
                    sums[u["uid"]].append(float(cap[L][0, j]))
        for u in us:
            a = np.array(sums[u["uid"]])
            acts[u["uid"]] = {"mean": float(a.mean()), "sign": 1 if a.mean() >= 0 else -1, "n_forks": len(a),
                              "frac_pos": float((a > 0).mean())}
        log(s, len(seqs), "forks")
    jdump(acts, f"{args.out}/signs.json")
    log("SIGNS_DONE")

# ------------------------------------------------------------------ phase gen
if "gen" in PH:
    prompts = json.load(open(f"{args.data}/prompts.json"))  # {pkey: {"text" | "prompt_id", "max_new"}}
    pid_ids = {}
    with gzip.open(f"{args.agastya}/stage_a_gen.jsonl.gz", "rt") as fh:
        for line in fh:
            g = json.loads(line)
            pid_ids[g["prompt_id"]] = g["prompt_ids"]
    for k, P in prompts.items():
        P["ids"] = pid_ids[P["prompt_id"]] if P.get("prompt_id") else chat_ids(P["text"])
    signs = json.load(open(f"{args.data}/signs.json"))

    def toward(u):
        if u["group"] == "A":
            return signs[u["uid"]]["sign"]
        if u["group"] == "B":
            return (1 if u["d"] >= 0 else -1) if u["kind"] == "set" else u["rand_sign"]
        return 1

    # jobs; generation is shared across units with the same (layer, neuron, signed push, alpha, prompt)
    jobs = {}
    for u in UNITS.values():
        for pk in ("who", "now", f"b2:{u['b']}"):
            for sgn in (1, -1):
                for a in ALPHAS:
                    s = sgn * toward(u)
                    gk = f"{pk}|{'C'+str(u['dir_index']) if u['group']=='C' else 'L%dN%d' % (u['layer'], u['neuron'])}|{s:+d}|{a}"
                    jobs.setdefault(gk, {"gk": gk, "prompt": pk, "uid": u["uid"], "s": s, "alpha": a})
    keys = sorted(jobs)
    si, sn = map(int, args.shard.split("/"))
    mine = [jobs[k] for k in keys if int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % sn == si]
    gp = f"{args.out}/gen_shard{si}{args.tag}.jsonl"
    done = set()
    if os.path.exists(gp):
        for l in open(gp):
            try:
                done.add(json.loads(l)["gk"])
            except Exception:
                pass
    todo = [j for j in mine if j["gk"] not in done]
    log(f"shard {args.shard}: {len(jobs)} unique jobs, {len(mine)} mine, {len(done)} done, {len(todo)} to do")
    byP = collections.defaultdict(list)
    for j in todo:
        byP[j["prompt"]].append(j)
    batches = [(pk, v[i:i + args.batch]) for pk, v in sorted(byP.items()) for i in range(0, len(v), args.batch)]
    if args.reverse:
        batches = batches[::-1]
    if args.limit:
        batches = batches[:args.limit]
    ntok = 0
    t1 = time.time()
    with open(gp, "a") as fo:
        for bi, (pk, bj) in enumerate(batches):
            if args.deadline and time.time() > args.deadline:
                log("deadline reached"); break
            P = prompts[pk]
            rows = [(UNITS[j["uid"]]["layer"], (j["s"] * j["alpha"] * R[UNITS[j["uid"]]["layer"]]) * unit_vec(UNITS[j["uid"]]))
                    for j in bj]
            res = gen_split(P["ids"], rows, P["max_new"])
            for j, (ids, fin) in zip(bj, res):
                ntok += len(ids)
                fo.write(json.dumps({**j, "text": tok.decode(ids), "n_tok": len(ids), "finish": fin}) + "\n")
            fo.flush()
            el = time.time() - t1
            if bi % 5 == 0 or bi == len(batches) - 1:
                log(f"batch {bi+1}/{len(batches)} {pk} B={len(bj)}: {ntok} tok in {el:.0f}s ({ntok/max(el,1):.0f} tok/s), "
                    f"finish {dict(collections.Counter(f for _, f in res))}")
    have = {json.loads(l)["gk"] for l in open(gp)}
    log(f"have {len(have & {j['gk'] for j in mine})}/{len(mine)} of this shard")
    if not args.limit and {j["gk"] for j in mine} <= have:
        log("GEN_DONE")
