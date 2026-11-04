"""Eval C pre-work on the VM (runs/evalc/SPEC.md; worker "pre"). Two phases, separate processes (each loads the model):

  states  base model, no adapter (~/olens/venv). (a) J-lens block-44 check against eval A's reads at 3 positions and
          the C6 keyword / control token directions d_t = J44^T (g * e_t) (g = 1 + final RMSNorm weight, e_t = lm_head row;
          the lens file is a plain {layer: J} map with no bias); (b) axis / role-pair cap thresholds tau at layer
          outputs 40-59 from P1's 24 benign default replies (74_persona_gen_gen.py's tau phase, HF capture instead of
          vLLM) and the comparison with X6's values (runs/persona_gen/checks.json) and P1's cap_tau.pt at 46-53;
          (c) matched-displacement norms: teacher-forced clean replies, down_proj columns of the top-10/100/1000 set
          zeroed, |delta h| per reply token at layer outputs 40-59.
  olens   s3d_rl600 merged, training carrier (W4 / scripts/86 settings): 5 validation neurons (2 samples, +-w), then
          1 sample per (neuron, sign) for the union of the 5 behaviors' top-1000 sets, read at the nearest trained
          lens layer >= the neuron's layer (capped at 60). Resumable, appends per batch.

Conventions: block L = output of decoder block L (hidden_states[L+1]); all vectors fp32; model bf16.
Run: cd ~/evil-neuron-interp && QWEN36_PATH=/data/models/Qwen3.6-27B flock /data/gpu.lock ~/olens/venv/bin/python -I \
     scripts/61_evalc_pre_vm.py --phase states|olens
"""
import argparse, gzip, json, os, sys, time, zlib

import numpy as np
import torch

MODEL = "/data/models/Qwen3.6-27B"
os.environ.setdefault("QWEN36_PATH", MODEL)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import olens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True, choices=["states", "olens"])
ap.add_argument("--out", default="/data/runs/evalc/pre")
ap.add_argument("--prep", default=os.path.join(REPO, "runs", "evalc", "prep"))
ap.add_argument("--p1", default="/data/runs/persona_p1")
ap.add_argument("--jlens", default="/data/lenses/qwen3.6-27b/j-lens/lens.pt")
ap.add_argument("--lens", default="s3d_rl600")
ap.add_argument("--batch", type=int, default=60)
ap.add_argument("--max_minutes", type=float, default=170)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--parts", default="j,tau,disp", help="states phase: which parts to run")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = time.time()
HOOK_LAYERS = list(range(40, 60))
AX_BLOCKS = [8, 16, 24, 32, 40, 48, 56]
PCTS = [5, 10, 25, 50, 75, 90, 95]
D = 5120


def log(*a):
    print(f"[pre {(time.time() - T0) / 60:6.1f}m]", *a, flush=True)


def jdump(x, p):
    tmp = p + ".tmp"
    json.dump(x, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


SETS = json.load(open(f"{args.prep}/sets.json"))
BEH = list(SETS["C1a"])  # 5 behaviors with own sets
SHORT = SETS["short"]


def set_lists(b, rung):
    return {int(L): [int(n) for n in ns] for L, ns in SETS["C1a"][b]["sets"][rung].items()}


# ====================================================================================== phase states
def phase_states():
    lm = olens._base(olens.MODEL, "cuda").eval()
    blocks = olens._blocks(lm)
    tm = next(m for m in lm.modules() if isinstance(getattr(m, "layers", None), torch.nn.ModuleList) and hasattr(m, "norm"))
    log("base loaded", type(blocks[0]).__name__)
    store = {}

    def grab(L):
        def hook(_m, _i, o):
            store[L] = (o[0] if isinstance(o, tuple) else o)[0]
        return hook

    handles = [blocks[L].register_forward_hook(grab(L)) for L in sorted(set(HOOK_LAYERS) | {44})]

    def zero(idx):
        def pre(_m, a):
            x = a[0].clone()
            x[..., idx] = 0
            return (x,) + tuple(a[1:])
        return pre

    @torch.no_grad()
    def forward(ids, zero_sets=None):
        """{L: [T, D] fp32 GPU} block outputs; zero_sets {L: [neuron idx]} zeroes down_proj inputs (= columns)."""
        hs = [blocks[L].mlp.down_proj.register_forward_pre_hook(zero(torch.tensor(ns, device="cuda")))
              for L, ns in (zero_sets or {}).items()]
        try:
            lm(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
        finally:
            for h in hs:
                h.remove()
        return {L: store[L].float() for L in store}

    rpath = f"{args.out}/states_results.json"
    results = json.load(open(rpath)) if os.path.exists(rpath) else {}
    PARTS = args.parts.split(",")
    # ------------------------------------------------------------------ (a) J-lens
    if "j" in PARTS:
        part_jlens(lm, tm, forward, results)
    if "tau" in PARTS:
        part_tau(forward, results)
    if "disp" in PARTS:
        part_disp(forward)
    for h in handles:
        h.remove()
    log("STATES_DONE")


def part_jlens(lm, tm, forward, results):
    J = torch.load(args.jlens, map_location="cpu", weights_only=True, mmap=True)
    jmeta = {k: (str(v)[:200] if k != "J" else None) for k, v in J.items() if k != "J"}
    jmeta["J_keys"] = f"{min(J['J'])}..{max(J['J'])} ({len(J['J'])} layers); value tensors only (no bias)"
    J44 = J["J"][44].float().cuda()
    del J
    g = 1.0 + tm.norm.weight.detach().float().cuda()  # Qwen3_5RMSNorm: x / rms(x) * (1 + weight)
    Wu = lm.lm_head.weight.detach()  # bf16 [V, D]
    eps = float(getattr(tm.norm, "variance_epsilon", getattr(tm.norm, "eps", 1e-6)))

    @torch.no_grad()
    def logits(x):  # 57_monitor_lens.Unembed: final norm fp32, lm_head bf16
        xn = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * g
        return (xn.to(Wu.dtype) @ Wu.T).float()

    # validation against eval A's reads (block 44 = hidden_states[45], readout log_softmax(lm_head(norm(J44 h))))
    seqs = {}
    val_rows = json.load(open(f"{args.prep}/vm/jlens_val_rows.json"))
    need = {r["sid"] for r in val_rows}
    for l in open(os.path.join(REPO, "data", "olens", "seqs.jsonl")):
        s = json.loads(l)
        if s["sid"] in need:
            seqs[s["sid"]] = s["ids"]
    tok0 = olens.AutoTokenizer.from_pretrained(olens.MODEL)
    jv = []
    for r in val_rows:
        h = forward(seqs[r["sid"]])[44][r["pos"]]
        ls = torch.log_softmax(logits((h @ J44.T)[None]), -1)[0]
        v, ix = ls.topk(20)
        ix = ix.tolist()
        jv.append({"sid": r["sid"], "pos": r["pos"], "stored_top20_ids": r["ids"], "here_top20_ids": ix,
                   "overlap20": len(set(ix) & set(r["ids"])), "top1_match": ix[0] == r["ids"][0],
                   "max_abs_dlogp_on_stored": float(max(abs(ls[t].item() - sc) for t, sc in zip(r["ids"], r["scores"]))),
                   "here_items": [tok0.decode([t]) for t in ix]})
        log("jlens val", r["sid"][:24], r["pos"], "overlap", jv[-1]["overlap20"], "top1", jv[-1]["top1_match"],
            "max|dlogp|", round(jv[-1]["max_abs_dlogp_on_stored"], 4))
    results["jlens_validation"] = jv
    # C6 directions
    tok = olens.AutoTokenizer.from_pretrained(olens.MODEL)
    c6 = SETS["C6"]["behaviors"]
    arrays, c6meta = {}, {"recipe": "d_t = J44^T (g * e_t), g = 1 + final RMSNorm weight (Qwen3_5RMSNorm), e_t = lm_head row t (fp32); the J-lens "
                                     "file is a plain linear map per layer, no bias. Group direction = unit(mean of "
                                     "unit(d_t) over the group's tokens); 'rawmean' = unit(mean of d_t).",
                          "block": 44, "per_token": {}, "self_readout_rank": {}, "cos": {}}

    @torch.no_grad()
    def tokdirs(ids):
        E = Wu[torch.tensor(ids, device="cuda")].float() * g  # [k, D]
        return E @ J44  # row i = J44^T (g * e_i)

    allvecs = {}
    for b, info in c6.items():
        groups = {"arm": [k["first_id"] for k in info["keywords"]]}
        for j, ctl in enumerate(info["controls"]):
            groups[f"ctrl_{j}"] = [c["id"] for c in ctl]
        for gname, ids in groups.items():
            Dt = tokdirs(ids)
            norms = Dt.norm(dim=-1)
            unit = Dt / norms[:, None]
            v = unit.mean(0)
            v = v / v.norm()
            raw = Dt.mean(0)
            raw = raw / raw.norm()
            key = f"{SHORT[b]}|{gname}"
            arrays[key] = v.cpu().numpy().astype(np.float32)
            arrays[key + "|rawmean"] = raw.cpu().numpy().astype(np.float32)
            arrays[key + "|token_norms"] = norms.cpu().numpy().astype(np.float32)
            arrays[key + "|token_ids"] = np.array(ids, dtype=np.int64)
            allvecs[key] = v
            # self-readout: rank of token t when its own direction is read through the lens
            lg = logits(Dt @ J44.T)
            ranks = [int((lg[i] > lg[i, t]).sum()) for i, t in enumerate(ids)]
            c6meta["per_token"][key] = [{"id": t, "piece": tok.decode([t]), "norm": round(float(n), 4), "self_rank": rk}
                                        for t, n, rk in zip(ids, norms.tolist(), ranks)]
            c6meta["self_readout_rank"][key] = ranks
            c6meta["cos"][key + "|unitmean_vs_rawmean"] = round(float(v @ raw), 4)
    for b in c6:
        a = allvecs[f"{SHORT[b]}|arm"]
        c6meta["cos"][f"{SHORT[b]}|arm_vs_ctrl"] = [round(float(a @ allvecs[f"{SHORT[b]}|ctrl_{j}"]), 4) for j in range(5)]
    bs = list(c6)
    c6meta["cos"]["arm_vs_arm"] = {f"{SHORT[a]}|{SHORT[b]}": round(float(allvecs[f"{SHORT[a]}|arm"] @ allvecs[f"{SHORT[b]}|arm"]), 4)
                                   for i, a in enumerate(bs) for b in bs[i + 1:]}
    # random unit directions in R^5120 have cos ~ N(0, 1/sqrt(5120)) = .014 sd
    c6meta["cos"]["note"] = "random unit vectors in R^5120: sd .014"
    c6meta["jlens_meta"] = jmeta
    np.savez(f"{args.out}/c6_dirs.npz", **arrays)
    jdump(c6meta, f"{args.out}/c6_meta.json")
    log("c6 arm-vs-ctrl cos", {k: v for k, v in c6meta["cos"].items() if k.endswith("arm_vs_ctrl")})
    log("c6 arm-vs-arm cos", c6meta["cos"]["arm_vs_arm"])
    del J44
    torch.cuda.empty_cache()
    jdump(results, f"{args.out}/states_results.json")


def part_tau(forward, results):
    V = f"{args.p1}/vectors"
    axis_raw = np.load(f"{V}/axis_raw.npy").astype(np.float64)
    roles = np.load(f"{V}/roles_raw.npy").astype(np.float64)
    pairs = json.load(open(f"{args.p1}/c1x_pairs.json"))["pairs"]
    ct = torch.load(f"{args.p1}/cap_tau.pt", weights_only=True)
    tau_p1 = {int(l): float(v[0]) for l, v in ct["tau"].items()}

    def unit(x):
        return x / max(np.linalg.norm(x), 1e-12)

    def lerp_raw(x, l):  # 74_persona_gen_gen.lerp_raw: P1/P3's lerp between saved blocks, clamped at 8 / 56
        if l <= 8:
            return x[0]
        if l >= 56:
            return x[6]
        i = (l - 8) // 8
        w = (l - AX_BLOCKS[i]) / 8.0
        return (1 - w) * x[i] + w * x[i + 1]

    DIRS = {"axis": {l: unit(lerp_raw(axis_raw, l)) for l in HOOK_LAYERS}}
    for j, (a, b) in enumerate(pairs):
        DIRS[f"rp{j}"] = {l: unit(lerp_raw(roles[a] - roles[b], l)) for l in HOOK_LAYERS}
    DN = list(DIRS)
    M = {l: torch.tensor(np.stack([DIRS[d][l] for d in DN]), dtype=torch.float32, device="cuda") for l in HOOK_LAYERS}
    benign = [json.loads(l) for l in open(f"{args.p1}/benign.jsonl")]
    bdef = [r for r in benign if r["style"] == "default"]
    if args.smoke:
        bdef = bdef[:3]
    projs = {l: [] for l in HOOK_LAYERS}
    for r in bdef:
        H = forward(r["prompt_ids"] + r["ids"])
        P, R = len(r["prompt_ids"]), len(r["ids"])
        for l in HOOK_LAYERS:
            projs[l].append((H[l][P:P + R] @ M[l].T).cpu().numpy())
    projs = {l: np.concatenate(v) for l, v in projs.items()}
    TAU = {d: {p: {l: float(np.percentile(projs[l][:, k], p)) for l in HOOK_LAYERS} for p in PCTS} for k, d in enumerate(DN)}
    fb = {l: float((projs[l][:, 0] < tau_p1[l]).mean()) for l in range(46, 54)}
    chk = {"n_benign_replies": len(bdef), "n_tokens": int(projs[40].shape[0]), "axis_frac_below_p1_tau_L46_53": fb,
           "axis_tau_p1_L46_53": tau_p1}
    x6 = json.load(open(f"{args.prep}/vm/persona_gen_checks.json"))["tau"]
    dev = {}
    for p, key in ((25, "axis_p25_here"), (50, "axis_p50"), (75, "axis_p75")):
        dev[f"p{p}"] = {l: round(TAU["axis"][p][l] - x6[key][str(l)], 4) for l in HOOK_LAYERS}
    chk["axis_tau_minus_x6_checks"] = dev
    chk["axis_tau_max_abs_dev_from_x6"] = max(abs(v) for d in dev.values() for v in d.values())
    chk["x6_values_rounded_to"] = "3 decimals (persona_gen/checks.json)"
    log("tau check: frac below P1 tau at 46-53", {l: round(v, 3) for l, v in fb.items()},
        "max |tau - X6|", round(chk["axis_tau_max_abs_dev_from_x6"], 4))
    cos_axis = {d: round(float(np.mean([DIRS[d][l] @ DIRS["axis"][l] for l in HOOK_LAYERS])), 4) for d in DN[1:]}
    np.savez(f"{args.out}/axis.npz",
             layers=np.array(HOOK_LAYERS),
             axis=np.stack([DIRS["axis"][l] for l in HOOK_LAYERS]).astype(np.float32),
             rp=np.stack([np.stack([DIRS[f"rp{j}"][l] for l in HOOK_LAYERS]) for j in range(len(pairs))]).astype(np.float32),
             pcts=np.array(PCTS),
             tau_axis=np.array([[TAU["axis"][p][l] for l in HOOK_LAYERS] for p in PCTS], dtype=np.float32),
             tau_rp=np.array([[[TAU[f"rp{j}"][p][l] for l in HOOK_LAYERS] for p in PCTS] for j in range(len(pairs))], dtype=np.float32),
             tau_axis_x6=np.array([[x6[k][str(l)] for l in HOOK_LAYERS] for k in ("axis_p25_here", "axis_p50", "axis_p75")], dtype=np.float32))
    results["tau"] = {"checks": chk, "pairs": pairs, "cos_rp_axis": cos_axis,
                      "tau_axis": {str(p): {str(l): round(TAU["axis"][p][l], 4) for l in HOOK_LAYERS} for p in (25, 50, 75)},
                      "projection_stats_axis": {str(l): {"mean": round(float(projs[l][:, 0].mean()), 3), "sd": round(float(projs[l][:, 0].std()), 3)} for l in HOOK_LAYERS}}
    jdump(results, f"{args.out}/states_results.json")
    del M
    torch.cuda.empty_cache()


def part_disp(forward):
    picks = json.load(open(f"{args.prep}/vm/clean_replies.json"))["picks"]
    want = {pid for b in BEH for pid, _ in picks[b]["replies"]}
    gen = {}
    with gzip.open("/data/agastya/stage_a_gen.jsonl.gz", "rt") as fh:
        for line in fh:
            g_ = json.loads(line)
            if g_["prompt_id"] in want:
                gen[g_["prompt_id"]] = g_
    disp = {"rule": "clean replies (clean_replies.json), teacher-forced prompt+reply; delta = block output with the set's "
                    "down_proj columns zeroed minus clean, reply positions only; norms per token; bf16 forward, fp32 diff",
            "layers": HOOK_LAYERS, "behaviors": {}}
    RUNGS = ["top10", "top100", "top1000"]
    for b in BEH:
        reps = picks[b]["replies"][:2] if args.smoke else picks[b]["replies"]
        acc = {r: {l: [] for l in HOOK_LAYERS} for r in RUNGS}
        rel = {r: {l: [] for l in HOOK_LAYERS} for r in RUNGS}
        hn = {l: [] for l in HOOK_LAYERS}
        per_reply = {r: {l: [] for l in HOOK_LAYERS} for r in RUNGS}
        ntok = 0
        for pid, i in reps:
            ids = gen[pid]["prompt_ids"] + gen[pid]["samples"][i]["ids"]
            P = len(gen[pid]["prompt_ids"])
            H0 = {l: h[P:].clone() for l, h in forward(ids).items()}
            ntok += len(ids) - P
            for l in HOOK_LAYERS:
                hn[l].append(H0[l].norm(dim=-1).cpu().numpy())
            for r in RUNGS:
                H1 = forward(ids, set_lists(b, r))
                for l in HOOK_LAYERS:
                    d = (H1[l][P:] - H0[l]).norm(dim=-1).cpu().numpy()
                    acc[r][l].append(d)
                    rel[r][l].append(d / np.maximum(H0[l].norm(dim=-1).cpu().numpy(), 1e-6))
                    per_reply[r][l].append(float(np.median(d)))
        out = {"n_replies": len(reps), "n_reply_tokens": ntok, "resid_norm_median": {}, "rungs": {}}
        for l in HOOK_LAYERS:
            out["resid_norm_median"][str(l)] = round(float(np.median(np.concatenate(hn[l]))), 3)
        for r in RUNGS:
            out["rungs"][r] = {}
            for l in HOOK_LAYERS:
                a = np.concatenate(acc[r][l]); q = np.concatenate(rel[r][l])
                out["rungs"][r][str(l)] = {"median": round(float(np.median(a)), 4), "p25": round(float(np.percentile(a, 25)), 4),
                                           "p75": round(float(np.percentile(a, 75)), 4), "mean": round(float(a.mean()), 4),
                                           "rel_median": round(float(np.median(q)), 5),
                                           "per_reply_median": [round(x, 4) for x in per_reply[r][l]]}
        disp["behaviors"][b] = out
        log(f"displacement {SHORT[b]}: top100 median |d| L44 {out['rungs']['top100']['44']['median']} "
            f"(rel {out['rungs']['top100']['44']['rel_median']}), L56 {out['rungs']['top100']['56']['median']} "
            f"(rel {out['rungs']['top100']['56']['rel_median']}); top1000 L56 {out['rungs']['top1000']['56']['median']}")
        jdump(disp, f"{args.out}/displacement.json")


# ====================================================================================== phase olens
def neuron_union():
    mem = {}
    for b in BEH:
        for L, ns in set_lists(b, "top1000").items():
            for n in ns:
                mem.setdefault((L, n), []).append(SHORT[b])
    return mem


def down_proj_columns(keys):
    """{(L, n): fp32 [D]} read from the model's safetensors shards (base weights, no adapter)."""
    from safetensors import safe_open
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    byL = {}
    for L, n in keys:
        byL.setdefault(L, []).append(n)
    out = {}
    for L, ns in sorted(byL.items()):
        name = next(k for k in idx if k.endswith(f"layers.{L}.mlp.down_proj.weight"))
        with safe_open(f"{MODEL}/{idx[name]}", framework="pt") as f:
            W = f.get_tensor(name)  # [D, D_ff] bf16
        for n in ns:
            out[(L, n)] = W[:, n].float().clone()
        del W
    return out


def phase_olens():
    c = olens.LENSES[args.lens]

    def read_layer(L):
        return next((l for l in c.layers if l >= L), c.layers[-1])

    mem = neuron_union()
    VAL = [(53, 15385), (54, 5531), (63, 4052), (62, 9559), (63, 9136)]
    keys = sorted(set(mem) | set(VAL))
    log(f"{len(mem)} neurons in the union of 5 top-1000 sets ({sum(len(v) > 1 for v in mem.values())} shared); reading columns")
    W = down_proj_columns(keys)
    torch.save({"keys": keys, "W": torch.stack([W[k] for k in keys]).to(torch.bfloat16), "membership": {f"{L}|{n}": v for (L, n), v in mem.items()}},
               f"{args.out}/neuron_dirs.pt")
    lens = olens.merge(olens.load_lens(args.lens))
    log(f"lens {args.lens} loaded and merged")
    # validation: 5 known neurons, +-w, 2 samples, W4 settings
    vpath = f"{args.out}/olens_validation.json"
    if not os.path.exists(vpath):
        val = []
        for (L, n) in VAL:
            for sign in (1, -1):
                texts = olens.read(lens, (sign * W[(L, n)])[None], read_layer(L), n=2, seed=zlib.crc32(f"val|{L}|{n}|{sign}".encode()))[0]
                val.append({"L": L, "n": n, "sign": sign, "read_layer": read_layer(L), "texts": texts})
                log(f"VAL L{L}N{n} {sign:+d} @L{read_layer(L)}: {texts[0][:300]!r}")
        jdump(val, vpath)
    # bulk: 1 sample per (neuron, sign)
    work = []
    for (L, n) in sorted(mem):
        for sign in (1, -1):
            work.append((f"N|{L}|{n}|{sign:+d}", read_layer(L), sign * W[(L, n)]))
    work.sort(key=lambda w: w[1])
    if args.smoke:
        work = work[:120]
    out = f"{args.out}/olens_reads.jsonl"
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["key"] for l in open(out)}
    todo = [w for w in work if w[0] not in done]
    log(f"{len(work)} reads, {len(done)} done, {len(todo)} to go")
    nread, t0 = 0, time.time()
    with open(out, "a") as f:
        i = 0
        while i < len(todo):
            L = todo[i][1]
            chunk = [w for w in todo[i:i + args.batch] if w[1] == L]
            texts = olens.read(lens, torch.stack([w[2] for w in chunk]), L, n=1, seed=zlib.crc32(chunk[0][0].encode()))
            for (key, layer, _), t in zip(chunk, texts):
                Lk, nk = int(key.split("|")[1]), int(key.split("|")[2])
                f.write(json.dumps({"key": key, "L": Lk, "n": nk, "sign": int(key.split("|")[3]), "layer": layer,
                                    "lens": args.lens, "sets": mem[(Lk, nk)], "texts": t}) + "\n")
            f.flush()
            i += len(chunk)
            nread += len(chunk)
            if (i // args.batch) % 10 == 0:
                log(f"{i}/{len(todo)} reads, {nread / (time.time() - t0):.1f} readouts/s")
            if time.time() - T0 > args.max_minutes * 60:
                log(f"time limit: {i}/{len(todo)}")
                return
    log("OLENS_DONE")


if args.phase == "states":
    phase_states()
else:
    phase_olens()
