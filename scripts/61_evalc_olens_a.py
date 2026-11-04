"""Eval C worker "olens_a": do the "affwhitened" O-lens (aff_rl600: camila-blank/qwen3.6-27b-olens-affwhitened-rl600
iter_000600, the s3d recipe with a closed-form ridge reconstructor scored in a whitened basis as the RL reward,
following-text carrier) and the final_olens lens B (lensB_rl600: agu18dec/final_olens lens_rl_step600, concept-bullet
carrier, alpha 1000, all 64 layers) read WeirdChat states better than s3d_rl600? (lens A = agu18dec/final_olens_whitened
lens_rl_step300 is third priority; its contract lensA_rl300 is in src/olens.py.)

VM (one process per lens; ~/olens/venv; the lens is merged into the base for speed):
  read      --lens aff_rl600|lensB_rl600|lensA_rl300 [--phase val,posb,neurons] [--ban auto|on|off]
    val      20 (b) positions at block 44 (and 56), 2 samples, with and without the template ban, plus the five
             validation neurons of scripts/61_evalc_pre_vm.py -> <out>/val_<lens>.json (with the decoded reply context).
    posb     every unique (b) position of the Stage-1 capture (index.jsonl set == "b", 4,061 rows: 2,844 k=4 + 1,217
             k=1) at blocks 44 and 56, 2 samples, 60 vectors per generate call seeded with crc32 of the call's first
             key (scripts/86b) -> <out>/reads_b_<lens>.jsonl rows {sid, pos, layer, k, texts, lens, ban, key};
             scripts/91_monitor_score.py kind "text" reads them (filter k=4 for the 2,844 canonical positions).
    neurons  +w / -w of every neuron of the union of the five behaviors' top-1000 sets (neuron_dirs.pt from
             scripts/61_evalc_pre_vm.py --phase olens: down_proj columns, membership), 1 sample per sign, read at the
             neuron's own block (both lenses cover all 64 layers) -> <out>/neurons_<lens>.jsonl, the schema of
             runs/evalc/prep/vm/olens_reads.jsonl. aff_rl600 (s3d layers 20-60 step 4) reads a neuron at the nearest
             trained layer >= its block, capped at 60, as scripts/61_evalc_pre_vm.py did for s3d; lens B at its own block.
  Sampling is each lens's contract (src/olens.LENSES: aff_rl600 alpha 16000, T 1, top-p 1, cap 256, following-text
  carrier; lensB_rl600 alpha 1000, T 1, top-p 1, cap 128, concept carrier), top-k 0. --ban auto applies the
  final_olens card's decode-time template ban to lens A / B only (their card recommends it; s3d and aff have none),
  re-implemented from the card's description without importing their olens.py: never sample <|im_start|>, <think>,
  </think>, <|endoftext|> (generate(suppress_tokens=...)) and no stop before 24 new tokens (min_new_tokens=24).
  Validation reads both with and without the ban. Resumable: appends per batch, skips keys already written, stops
  at --max_minutes.
  Run: cd ~/evil-neuron-interp && HF_HUB_CACHE=/data/lens_dl/hub HF_HUB_OFFLINE=1 QWEN36_PATH=/data/models/Qwen3.6-27B \
       flock /data/gpu.lock ~/olens/venv/bin/python -I scripts/61_evalc_olens_a.py read --lens aff_rl600

Local (.venv):
  score     --lens L: blind Haiku scoring of neurons_<lens>.jsonl with scripts/61_evalc_pre_local.py's scorer (same
            rubric: all five behaviors + weird per readout, shuffled across sets), cache <out>/haiku_<lens>.jsonl and
            token log <out>/haiku_calls.jsonl instead of that script's fixed paths. --limit N calls for the pilot.
  analyze   neuron recognition per behavior and lens (own-behavior score = max over signs, as C7: AUROC of the C1a
            top-100 vs the other 900 of the top-1000, share of the top-100 scoring >= 25, n >= 25 / > 50 of 1,000),
            the same for s3d from runs/evalc/prep/vm/haiku_raw.jsonl as a check of the reference numbers, and the
            eval A table rows (olens_a, olens_b, olens, text_b) from runs/monitors/eval/table_b.csv -> <out>/results.json
"""
import argparse, json, os, sys, time, zlib

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
OUT_LOCAL = os.path.join(REPO, "runs", "evalc", "olens_a")
BAN_TOKENS = ("<|im_start|>", "<think>", "</think>", "<|endoftext|>")
BAN_MIN_NEW = 24
BAN_DEFAULT = {"lensA_rl300": True, "lensA_rl600": True, "lensB_rl600": True, "lensB_rl300": True}  # others: off
VAL_NEURONS = [(53, 15385), (54, 5531), (63, 4052), (62, 9559), (63, 9136)]
T0 = time.time()


def log(*a):
    print(f"[olens_a {(time.time() - T0) / 60:6.1f}m]", *a, flush=True)


def jdump(x, p):
    tmp = p + ".tmp"
    json.dump(x, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


# ====================================================================================== VM: reads
def ban_ids(tok):
    ids = [tok.convert_tokens_to_ids(t) for t in BAN_TOKENS]
    assert all(isinstance(i, int) and i >= 0 for i in ids) and [tok.decode([i]) for i in ids] == list(BAN_TOKENS), ids
    return ids


def read_lens(lens, H, layer, n, seed, ban):
    """olens.read with the template ban: ``H`` [k, 5120] -> k lists of n readouts, one batched generate."""
    import torch
    import olens
    c = lens.contract
    torch.manual_seed(seed)
    hs = H.float().reshape(-1, olens.D_MODEL)
    ids, slot = olens._prompt(lens, layer)  # the contract's carrier
    ids = torch.tensor([ids], device=lens.lm.device).repeat(len(hs) * n, 1)
    emb = lens.lm.get_input_embeddings()(ids)
    unit = hs / hs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    emb[:, slot] = (c.alpha * unit).repeat_interleave(n, 0).to(emb.device, emb.dtype)
    extra = {"suppress_tokens": ban_ids(lens.tok), "min_new_tokens": min(BAN_MIN_NEW, c.max_new)} if ban else {}
    out = lens.lm.generate(inputs_embeds=emb, attention_mask=torch.ones_like(ids), max_new_tokens=c.max_new,
                           do_sample=True, temperature=c.temperature, top_p=c.top_p, top_k=0, use_cache=True,
                           pad_token_id=lens.tok.pad_token_id or lens.tok.eos_token_id, **extra)
    texts = lens.tok.batch_decode(out, skip_special_tokens=True)
    return [texts[i * n:(i + 1) * n] for i in range(len(hs))]


def degenerate(t):
    return len(t.strip()) < 8 or "<|im_start|>" in t or "assistant\n" in t


def cmd_read(args):
    import numpy as np
    import torch
    import olens
    os.makedirs(args.out, exist_ok=True)
    phases = args.phase.split(",")
    idx = [json.loads(l) for l in open(f"{args.cap}/index.jsonl")]
    rows_b = [m for m in idx if m["set"] == "b"]
    layers = [int(x) for x in args.layers.split(",")]
    R = {L: np.load(f"{args.cap}/resid_{L}.npy", mmap_mode="r") for L in layers}
    lens = olens.load_lens(args.lens, path=args.lens_path or None)  # raises unless every LoRA B is present and non-zero
    n_b = sum("lora_B" in k for k, _ in lens.lm.named_parameters())
    log(f"lens {args.lens} loaded: {n_b} LoRA B matrices live; contract {lens.contract}")
    lens = olens.merge(lens)
    ban = BAN_DEFAULT.get(args.lens, False) if args.ban == "auto" else args.ban == "on"
    c = lens.contract
    assert all(L in c.layers for L in layers), (layers, c.layers)

    def read_layer(L):  # a neuron at block L writes into every residual from block L on
        return next((l for l in c.layers if l >= L), c.layers[-1])

    log(f"merged; bulk ban {ban}; ban ids {ban_ids(lens.tok)}; carrier {c.prompt[-60:]!r}")
    # ------------------------------------------------------------------ validation
    if "val" in phases:
        vpath = f"{args.out}/val_{args.lens}.json"
        if not os.path.exists(vpath):
            seqs = {}
            k4 = [m for m in rows_b if m["k"] == 4]
            pick = [k4[i] for i in np.random.default_rng(0).choice(len(k4), 20, replace=False)]
            need = {m["sid"] for m in pick}
            for l in open(os.path.join(REPO, "runs", "monitors", "seqs_b.jsonl")):
                s = json.loads(l)
                if s["sid"] in need:
                    seqs[s["sid"]] = s["ids"]
            val = {"lens": args.lens, "contract": str(lens.contract), "n_lora_B": n_b, "positions": []}
            H = {L: torch.tensor(np.stack([R[L][m["row"]] for m in pick]).astype(np.float32)) for L in layers}
            reads = {(L, bn): read_lens(lens, H[L], L, 2, zlib.crc32(f"val|{L}|{bn}".encode()), bn)
                     for L in layers for bn in (True, False)}
            for i, m in enumerate(pick):
                ids = seqs[m["sid"]]
                ctx = lens.tok.decode(ids[max(0, m["pos"] - 60):m["pos"] + 1])
                nxt = lens.tok.decode(ids[m["pos"] + 1:m["pos"] + 25])
                val["positions"].append({"sid": m["sid"], "pos": m["pos"], "b": m["b"], "context_to_pos": ctx,
                                         "next_24_tokens": nxt,
                                         "reads": {f"L{L}|{'ban' if bn else 'noban'}": reads[(L, bn)][i]
                                                   for L in layers for bn in (True, False)}})
            val["degenerate_share"] = {f"L{L}|{'ban' if bn else 'noban'}": sum(degenerate(t) for ts in reads[(L, bn)] for t in ts) / 40
                                       for L in layers for bn in (True, False)}
            val["bulk_ban"] = ban
            nd = torch.load(f"{args.pre}/neuron_dirs.pt", weights_only=True)
            kpos = {tuple(k): i for i, k in enumerate(nd["keys"])}
            val["neurons"] = []
            for (L, n) in VAL_NEURONS:
                for sign in (1, -1):
                    w = sign * nd["W"][kpos[(L, n)]].float()
                    texts = read_lens(lens, w[None], read_layer(L), 2, zlib.crc32(f"val|{L}|{n}|{sign}".encode()), ban)[0]
                    val["neurons"].append({"L": L, "n": n, "sign": sign, "read_layer": read_layer(L), "texts": texts})
                    log(f"VAL L{L}N{n} {sign:+d}: {texts[0][:200]!r}")
            jdump(val, vpath)
            log("validation written; degenerate share", val["degenerate_share"])
            for p in val["positions"][:6]:
                log(f"VAL {p['b']} ...{p['context_to_pos'][-80:]!r}\n   L44 {'ban' if ban else 'noban'}: "
                    f"{p['reads']['L44|' + ('ban' if ban else 'noban')][0][:200]!r}")
    # ------------------------------------------------------------------ (b) positions
    if "posb" in phases:
        work = [(f"R|{m['sid']}|{m['pos']}|{L}", L, m) for L in layers for m in rows_b]
        out = f"{args.out}/reads_b_{args.lens}.jsonl"
        done = {json.loads(l)["key"] for l in open(out)} if os.path.exists(out) else set()
        todo = [w for w in work if w[0] not in done]
        log(f"posb: {len(work)} vectors ({len(rows_b)} positions x {layers}), {len(done)} done, {len(todo)} to read")
        nread, t0 = 0, time.time()
        with open(out, "a") as f:
            i = 0
            while i < len(todo):
                L = todo[i][1]
                chunk = [w for w in todo[i:i + args.batch] if w[1] == L]
                H = torch.tensor(np.stack([R[L][w[2]["row"]] for w in chunk]).astype(np.float32))
                texts = read_lens(lens, H, L, args.n, zlib.crc32(chunk[0][0].encode()), ban)
                for (key, layer, m), t in zip(chunk, texts):
                    f.write(json.dumps({"sid": m["sid"], "pos": m["pos"], "layer": layer, "k": m["k"], "texts": t,
                                        "lens": args.lens, "ban": ban, "key": key}) + "\n")
                f.flush()
                i += len(chunk)
                nread += len(chunk) * args.n
                if (i // args.batch) % 10 == 0:
                    log(f"posb {i}/{len(todo)} vectors, {nread / (time.time() - t0):.1f} readouts/s")
                if time.time() - T0 > args.max_minutes * 60:
                    log(f"time limit: posb {i}/{len(todo)}")
                    return
        log("posb done")
    # ------------------------------------------------------------------ neuron directions
    if "neurons" in phases:
        nd = torch.load(f"{args.pre}/neuron_dirs.pt", weights_only=True)
        mem = {tuple(int(x) for x in k.split("|")): v for k, v in nd["membership"].items()}
        work = []
        for (L, n), w in zip(nd["keys"], nd["W"]):
            if (L, n) not in mem:  # the five validation neurons are stored too
                continue
            for sign in (1, -1):
                work.append((f"N|{L}|{n}|{sign:+d}", read_layer(L), sign * w.float()))
        work.sort(key=lambda w: w[1])
        out = f"{args.out}/neurons_{args.lens}.jsonl"
        done = {json.loads(l)["key"] for l in open(out)} if os.path.exists(out) else set()
        todo = [w for w in work if w[0] not in done]
        log(f"neurons: {len(mem)} neurons, {len(work)} reads, {len(done)} done, {len(todo)} to go")
        nread, t0 = 0, time.time()
        with open(out, "a") as f:
            i = 0
            while i < len(todo):
                L = todo[i][1]
                chunk = [w for w in todo[i:i + args.batch] if w[1] == L]
                texts = read_lens(lens, torch.stack([w[2] for w in chunk]), L, 1, zlib.crc32(chunk[0][0].encode()), ban)
                for (key, layer, _), t in zip(chunk, texts):
                    Lk, nk = int(key.split("|")[1]), int(key.split("|")[2])
                    f.write(json.dumps({"key": key, "L": Lk, "n": nk, "sign": int(key.split("|")[3]), "layer": layer,
                                        "lens": args.lens, "ban": ban, "sets": mem[(Lk, nk)], "texts": t}) + "\n")
                f.flush()
                i += len(chunk)
                nread += len(chunk)
                if (i // args.batch) % 10 == 0:
                    log(f"neurons {i}/{len(todo)} reads, {nread / (time.time() - t0):.1f} readouts/s")
                if time.time() - T0 > args.max_minutes * 60:
                    log(f"time limit: neurons {i}/{len(todo)}")
                    return
        log("neurons done")
    log("READ_DONE")


# ====================================================================================== local: neuron scoring
def pre_local():
    """scripts/61_evalc_pre_local.py as a module (its cache paths are module globals, overridden per lens)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pre_local", os.path.join(REPO, "scripts", "61_evalc_pre_local.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_score(args):
    mod = pre_local()
    mod.RAW = os.path.join(args.out, f"haiku_{args.lens}.jsonl")
    mod.CALLS = os.path.join(args.out, "haiku_calls.jsonl")  # one token log for the whole task
    ns = argparse.Namespace(reads=os.path.join(args.out, f"neurons_{args.lens}.jsonl"), model="claude-haiku-5-5",
                            batch=10, workers=8, limit=args.limit, budget_in=args.budget_in, dry=args.dry)
    mod.cmd_score(ns)


# ====================================================================================== local: analysis
def auroc(pos, neg):
    """Rank AUROC with ties counted half (Mann-Whitney)."""
    from scipy.stats import rankdata
    import numpy as np
    x = np.array(list(pos) + list(neg), dtype=float)
    r = rankdata(x)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def neuron_stats(raw_path, sets, label):
    from collections import defaultdict
    raw = defaultdict(list)
    for l in open(raw_path):
        r = json.loads(l)
        raw[(r["L"], r["n"], r["sign"])].append(r["scores"])
    out = {}
    for b in sets["C1a"]:
        sb = sets["short"][b]
        order = [(int(L), int(n)) for L, ns in sets["C1a"][b]["sets"]["top1000"].items() for n in ns]
        top100 = {(int(L), int(n)) for L, ns in sets["C1a"][b]["sets"]["top100"].items() for n in ns}
        sc, miss = {}, 0
        for (L, n) in order:
            per = [sum(x[sb] for x in raw[(L, n, s)]) / len(raw[(L, n, s)]) for s in (1, -1) if raw.get((L, n, s))]
            if per:
                sc[(L, n)] = max(per)
            else:
                miss += 1
        pos = [sc[k] for k in order if k in top100 and k in sc]
        neg = [sc[k] for k in order if k not in top100 and k in sc]
        vals = list(sc.values())
        out[sb] = {"n_scored": len(vals), "n_unscored": miss, "auroc_top100_vs_rest": round(auroc(pos, neg), 3),
                   "top100_share_ge25": round(sum(v >= 25 for v in pos) / len(pos), 3),
                   "rest_share_ge25": round(sum(v >= 25 for v in neg) / len(neg), 3),
                   "top100_mean": round(sum(pos) / len(pos), 2), "rest_mean": round(sum(neg) / len(neg), 2),
                   "n_ge25_of_1000": sum(v >= 25 for v in vals), "n_gt50_of_1000": sum(v > 50 for v in vals),
                   "n_gt0_of_1000": sum(v > 0 for v in vals), "mean": round(sum(vals) / len(vals), 2)}
        print(f"{label:12s} {sb:11s} AUROC top100 vs rest {out[sb]['auroc_top100_vs_rest']:.3f}  top100 >=25 "
              f"{100 * out[sb]['top100_share_ge25']:.0f}%  of 1000: >=25 {out[sb]['n_ge25_of_1000']}, >50 "
              f"{out[sb]['n_gt50_of_1000']}  (scored {len(vals)})", flush=True)
    return out


def cmd_analyze(args):
    import csv
    sets = json.load(open(os.path.join(REPO, "runs", "evalc", "prep", "sets.json")))
    res = {"neurons": {}, "eval_a": {}}
    res["neurons"]["s3d_rl600 (runs/evalc/prep/vm/haiku_raw.jsonl)"] = neuron_stats(
        os.path.join(REPO, "runs", "evalc", "prep", "vm", "haiku_raw.jsonl"), sets, "s3d")
    for lens in ("aff_rl600", "lensB_rl600", "lensA_rl300"):
        p = os.path.join(args.out, f"haiku_{lens}.jsonl")
        if os.path.exists(p):
            res["neurons"][lens] = neuron_stats(p, sets, lens)
    tb = os.path.join(REPO, "runs", "monitors", "eval", "table_b.csv")
    keep = ["method", "variant", "k", "n_unique", "auroc", "auroc_lo", "auroc_hi", "auroc_sharp", "auroc_nonsharp", "gain",
            "gain_lo", "gain_hi", "gain_base", "gain_n_cuts", "status"]
    if os.path.exists(tb):
        for r in csv.DictReader(open(tb)):
            if r["method"] in ("olens_aff", "olens_b", "olens_a", "olens", "text_b") and (r["method"] == "text_b" or "behavior" in r["variant"]):
                key = f"{r['method']}:{r['variant']}|k={r['k']}|n={r['n_unique']}"
                res["eval_a"][key] = {k: r[k] for k in keep}
                print(f"{key:40s} AUROC {float(r['auroc']):.3f} [{float(r['auroc_lo']):.3f}, {float(r['auroc_hi']):.3f}] "
                      f"sharp {r['auroc_sharp']} other {r['auroc_nonsharp']} gain {r['gain']} [{r['gain_lo']}, {r['gain_hi']}]")
    # haiku tokens of this task
    cp = os.path.join(args.out, "haiku_calls.jsonl")
    if os.path.exists(cp):
        rows = [json.loads(l) for l in open(cp)]
        res["haiku_neuron_scoring"] = {"calls": len(rows), "in": sum(r["in"] for r in rows), "out": sum(r["out"] for r in rows)}
    jdump(res, os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["read", "score", "analyze"])
    ap.add_argument("--lens", default="aff_rl600")
    ap.add_argument("--ban", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--lens_path", default="", help="local checkpoint folder (the partial snapshot under /data/lens_dl/hub)")
    ap.add_argument("--phase", default="val,posb,neurons")
    ap.add_argument("--cap", default="/data/runs/monitors/capture")
    ap.add_argument("--pre", default="/data/runs/evalc/pre")
    ap.add_argument("--out", default=None, help="VM: /data/runs/evalc/olens_a; local: runs/evalc/olens_a")
    ap.add_argument("--layers", default="44,56")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--batch", type=int, default=60)
    ap.add_argument("--max_minutes", type=float, default=170)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-in", type=float, default=8e6)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.out is None:
        a.out = "/data/runs/evalc/olens_a" if a.cmd == "read" else OUT_LOCAL
    {"read": cmd_read, "score": cmd_score, "analyze": cmd_analyze}[a.cmd](a)
