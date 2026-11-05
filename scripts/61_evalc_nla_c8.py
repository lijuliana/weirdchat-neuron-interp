"""Eval C post-hoc arm C8 "NLA-filtered 100 of the top-1000" (runs/evalc/SPEC.md: C7's recipe with the natural-language
autoencoder agu18dec/qwen3.6-27b-all-layer-nla-rl2000 (src/nla.py) as the reader instead of the O-lens s3d_rl600).
C8 was dropped at spec time for lack of a neuron ranking; the attribution top-1000 per behavior (C7's input) supplies
it, so C8 = C7 with the NLA reading the +w / -w output directions of every top-1000 neuron at the neuron's own block.
Post hoc: outside the pre-registered Holm family; controls = C7's existing ten "random 100 of the top-1000" (their
judged replies reused, not regenerated).

VM (evalc-1: ~/olens/venv, /data/models/Qwen3.6-27B, NLA repo in the fresh dir /data/nla_dl, neuron_dirs.pt from
scripts/61_evalc_pre_vm.py --phase olens in /data/runs/evalc/pre):
  read      validation (the five neurons of 61_evalc_pre_vm.py, +-w, 2 samples) -> <out>/val_nla.json, then 1 sample
            per (neuron, sign) for the union of the five top-1000 sets at the neuron's own block (NLA covers 0..63)
            -> <out>/neurons_nla.jsonl, the schema of runs/evalc/prep/vm/olens_reads.jsonl (lens "nla"; texts = the
            extracted <explanation> text, raw reply kept in "raw", "tagged" = tags present; untagged readouts fall
            back to the raw reply). Sampling = src/nla.py's contract (T 1.0, top-p 1, top-k 0, 200 new tokens),
            LoRA merged (scripts/58's merge check). Resumable, appends per batch, stops at --max_minutes.
            Run: cd ~/evil-neuron-interp && QWEN36_PATH=/data/models/Qwen3.6-27B flock /data/gpu.lock \
                 ~/olens/venv/bin/python -I scripts/61_evalc_nla_c8.py read --out /data/runs/evalc/nla_c8

Local (.venv):
  score     blind Haiku scoring of neurons_nla.jsonl with scripts/61_evalc_pre_local.py's scorer (same rubric, model,
            batching and shuffle as C7), cache <out>/haiku_nla.jsonl, token log <out>/haiku_calls.jsonl. --limit N
            calls for the pilot, --dry for the projection.
  assemble  the C7 selection rule (61_evalc_pre_local.py cmd_assemble, imported and pointed at the NLA scores) ->
            <out>/c8_sets.json (c7_sets.json's format, plus the overlap with the O-lens-picked C7 100 per behavior)
            and neuron recognition (61_evalc_olens_a.py neuron_stats: AUROC of the C1a top-100 vs the other 900,
            share >= 25) -> <out>/recognition.json.
  patch     write a patched copy of scripts/61_evalc_gen.py that knows the C8 arm (ARMS / LABELS / FAMILY / LADDER
            entries, c8_sets.json loader, the C8_nla100 condition, and no positive controls when --arms is given) to
            --patched (run on evalc-2 with --out /data/evalc_c8/gen --vm /data/evalc_c8/vm --arms C8; every patch
            must match the source exactly once). The judge stage is the unmodified chain with OUT=/data/evalc_c8.
  analyze   section-0 statistic for C8_nla100 against C7_ctrl_00..09 on the common set, from the existing judged
            clean / controls rows (runs/evalc/judge/judged) and the C8 judged arm rows (<out>/judged/arms.jsonl),
            with scripts/61_evalc_analyze.py's own functions (same bootstrap seed) -> <out>/results.json; the C7 row
            is recomputed the same way as a check against runs/evalc/judge/eval/main_table.csv.
"""
import argparse, collections, importlib.util, json, os, sys, time, zlib

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
OUT_LOCAL = os.path.join(REPO, "runs", "evalc", "nla_c8")
VAL_NEURONS = [(53, 15385), (54, 5531), (63, 4052), (62, 9559), (63, 9136)]
ARM, COND_ARM, RUNG = "C8", "C8_nla100", "nla100"
T0 = time.time()


def log(*a):
    print(f"[nla_c8 {(time.time() - T0) / 60:6.1f}m]", *a, flush=True)


def jdump(x, p):
    tmp = p + ".tmp"
    json.dump(x, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ====================================================================================== VM: NLA reads
def degenerate(t):
    return len(t.strip()) < 8


def cmd_read(args):
    import torch
    import nla
    os.makedirs(args.out, exist_ok=True)
    nd = torch.load(f"{args.pre}/neuron_dirs.pt", weights_only=True)
    kpos = {tuple(int(x) for x in k): i for i, k in enumerate(nd["keys"])}
    mem = {tuple(int(x) for x in k.split("|")): v for k, v in nd["membership"].items()}
    av = nla.load_av(args.nla_dir)
    n_b = sum("lora_B" in k for k, _ in av.lm.named_parameters())
    log(f"AV loaded from {args.nla_dir}: {n_b} LoRA B matrices live; merging")
    av = nla.merge(av)
    log("merged; prompt for layer 44 ends", repr(av.tok.decode(nla.prompt_ids(av, 44)[0][-40:])))

    def read(H, L, n, seed):
        raw = nla.read(av, H, L, n=n, seed=seed)
        out = []
        for rs in raw:
            ex = [nla.explanation(t) for t in rs]
            out.append({"texts": [e if e is not None else t for e, t in zip(ex, rs)], "raw": rs,
                        "tagged": [e is not None for e in ex]})
        return out

    # ------------------------------------------------------------------ validation: five known neurons
    vpath = f"{args.out}/val_nla.json"
    if not os.path.exists(vpath):
        val = {"nla_dir": args.nla_dir, "n_lora_B": n_b, "neurons": []}
        for (L, n) in VAL_NEURONS:
            for sign in (1, -1):
                w = sign * nd["W"][kpos[(L, n)]].float()
                r = read(w[None], L, 2, zlib.crc32(f"val|{L}|{n}|{sign}".encode()))[0]
                val["neurons"].append({"L": L, "n": n, "sign": sign, "read_layer": L, **r})
                log(f"VAL L{L}N{n} {sign:+d} tagged {r['tagged']}: {r['texts'][0][:220]!r}")
        jdump(val, vpath)
    # ------------------------------------------------------------------ bulk
    work = []
    for (L, n), w in zip(nd["keys"], nd["W"]):
        L, n = int(L), int(n)
        if (L, n) not in mem:  # the five validation neurons are stored too
            continue
        for sign in (1, -1):
            work.append((f"N|{L}|{n}|{sign:+d}", L, sign * w.float()))
    work.sort(key=lambda w: w[1])
    out = f"{args.out}/neurons_nla.jsonl"
    done = {json.loads(l)["key"] for l in open(out)} if os.path.exists(out) else set()
    todo = [w for w in work if w[0] not in done]
    if args.smoke:
        todo = todo[:2 * args.batch]
    log(f"neurons: {len(mem)} neurons, {len(work)} reads, {len(done)} done, {len(todo)} to go, batch {args.batch}")
    nread, ntag, ndeg, t0, shown = 0, 0, 0, time.time(), 0
    with open(out, "a") as f:
        i = 0
        while i < len(todo):
            L = todo[i][1]
            chunk = [w for w in todo[i:i + args.batch] if w[1] == L]
            res = read(torch.stack([w[2] for w in chunk]), L, 1, zlib.crc32(chunk[0][0].encode()))
            for (key, layer, _), r in zip(chunk, res):
                Lk, nk = int(key.split("|")[1]), int(key.split("|")[2])
                f.write(json.dumps({"key": key, "L": Lk, "n": nk, "sign": int(key.split("|")[3]), "layer": layer, "lens": "nla",
                                    "sets": mem[(Lk, nk)], "texts": r["texts"], "raw": r["raw"], "tagged": r["tagged"]}) + "\n")
                ntag += sum(r["tagged"])
                ndeg += sum(degenerate(t) for t in r["texts"])
                if shown < 10:
                    log(f"SAMPLE {key} @L{layer} {mem[(Lk, nk)]}: {r['texts'][0][:200]!r}")
                    shown += 1
            f.flush()
            i += len(chunk)
            nread += len(chunk)
            if (i // args.batch) % 5 == 0 or i >= len(todo):
                log(f"neurons {i}/{len(todo)} reads, {nread / (time.time() - t0):.1f} readouts/s, tagged {ntag}/{nread}, degenerate {ndeg}")
            if time.time() - T0 > args.max_minutes * 60:
                log(f"time limit: neurons {i}/{len(todo)}")
                return
    log(f"READ_DONE tagged {ntag}/{nread} degenerate {ndeg}")


# ====================================================================================== local: Haiku scoring
def cmd_score(args):
    mod = load_module("pre_local", "scripts/61_evalc_pre_local.py")
    mod.RAW = os.path.join(args.out, "haiku_nla.jsonl")
    mod.CALLS = os.path.join(args.out, "haiku_calls.jsonl")
    ns = argparse.Namespace(reads=os.path.join(args.out, "neurons_nla.jsonl"), model="claude-haiku-5-5", batch=10, workers=8,
                            limit=args.limit, budget_in=args.budget_in, dry=args.dry)
    mod.cmd_score(ns)


# ====================================================================================== local: C8 set + recognition
def cmd_assemble(args):
    mod = load_module("pre_local", "scripts/61_evalc_pre_local.py")
    mod.RAW = os.path.join(args.out, "haiku_nla.jsonl")
    mod.CALLS = os.path.join(args.out, "haiku_calls.jsonl")
    mod.VM = args.out  # cmd_assemble writes <VM>/c7_sets.json; renamed below
    mod.cmd_assemble(argparse.Namespace())
    tmp = os.path.join(args.out, "c7_sets.json")
    c8 = json.load(open(tmp))
    os.remove(tmp)
    c8["recipe"] = ("NLA (agu18dec/qwen3.6-27b-all-layer-nla-rl2000) <explanation> readout of +w and -w (1 sample each) of every "
                    "top-1000 neuron at its own block, scored blind by Haiku 5.5 on all five behaviors + weird (0-100). Neuron "
                    "score for behavior b = max over signs of the b score; C8 set = top 100 by (b score desc, weird desc, position "
                    "in the top-1000 list), i.e. scripts/61_evalc_pre_local.py's C7 rule applied unchanged. Controls: the C7 "
                    "controls (sets.json C7.<b>.controls), replies reused.")
    c7 = json.load(open(os.path.join(REPO, "runs", "evalc", "prep", "vm", "c7_sets.json")))
    for b, v in c8["behaviors"].items():
        s8 = {tuple(x) for x in v["neurons"]}
        s7 = {tuple(x) for x in c7["behaviors"][b]["neurons"]}
        v["overlap_olens100"] = len(s8 & s7)
        v["overlap_olens100_and_top100"] = len({k for k in s8 & s7 if v["scores"][f"{k[0]}|{k[1]}"]["in_top100"]})
        print(f"{b}: overlap with C1a top-100 {v['overlap_top100']}, with the O-lens C7 100 {v['overlap_olens100']}, "
              f"C7 controls {v['overlap_controls']}", flush=True)
    jdump(c8, os.path.join(args.out, "c8_sets.json"))
    # recognition (olens_a's neuron_stats) for nla and, as the reference, the s3d scores
    oa = load_module("olens_a", "scripts/61_evalc_olens_a.py")
    sets = json.load(open(os.path.join(REPO, "runs", "evalc", "prep", "sets.json")))
    rec = {"nla": oa.neuron_stats(mod.RAW, sets, "nla"),
           "s3d_rl600 (runs/evalc/prep/vm/haiku_raw.jsonl)": oa.neuron_stats(os.path.join(REPO, "runs", "evalc", "prep", "vm", "haiku_raw.jsonl"), sets, "s3d")}
    reads = [json.loads(l) for l in open(os.path.join(args.out, "neurons_nla.jsonl"))]
    rec["reads"] = {"n": len(reads), "tagged": sum(all(r.get("tagged", [True])) for r in reads),
                    "degenerate": sum(degenerate(t) for r in reads for t in r["texts"]),
                    "mean_chars": round(sum(len(t) for r in reads for t in r["texts"]) / max(1, sum(len(r["texts"]) for r in reads)), 1)}
    rec["haiku_tokens"] = c8["haiku_tokens"]
    jdump(rec, os.path.join(args.out, "recognition.json"))
    print("recognition", json.dumps({k: {b: v["auroc_top100_vs_rest"] for b, v in rec[k].items()} for k in list(rec)[:2]}), flush=True)


def cmd_assemble_aff(args):
    """C10: the affine-map O-lens's Haiku scores of the same neuron directions (runs/evalc/olens_a/haiku_aff_rl600.jsonl,
    written by scripts/61_evalc_olens_a.py score) through the unchanged C7 rule -> <out>/c10_sets.json."""
    mod = load_module("pre_local", "scripts/61_evalc_pre_local.py")
    mod.RAW = os.path.join(REPO, "runs", "evalc", "olens_a", "haiku_aff_rl600.jsonl")
    mod.CALLS = os.path.join(args.out, "haiku_calls_aff_unused.jsonl")  # the aff scores were paid for by olens_a; empty log here
    open(mod.CALLS, "a").close()
    mod.VM = args.out
    mod.cmd_assemble(argparse.Namespace())
    tmp = os.path.join(args.out, "c7_sets.json")
    c10 = json.load(open(tmp))
    os.remove(tmp)
    c10["recipe"] = ("Affine-map O-lens (camila-blank/qwen3.6-27b-olens-affwhitened-rl600 iter_000600) readout of +w and -w of every "
                     "top-1000 neuron at the nearest trained layer (scripts/61_evalc_olens_a.py read --lens aff_rl600), scored blind by "
                     "Haiku 5.5 with the C7 rubric (runs/evalc/olens_a/haiku_aff_rl600.jsonl); C10 set = scripts/61_evalc_pre_local.py's "
                     "C7 rule applied unchanged. Controls: the C7 controls, replies reused.")
    c7 = json.load(open(os.path.join(REPO, "runs", "evalc", "prep", "vm", "c7_sets.json")))
    for b, v in c10["behaviors"].items():
        s10 = {tuple(x) for x in v["neurons"]}
        s7 = {tuple(x) for x in c7["behaviors"][b]["neurons"]}
        v["overlap_olens100"] = len(s10 & s7)
        print(f"{b}: overlap with C1a top-100 {v['overlap_top100']}, with the O-lens C7 100 {v['overlap_olens100']}", flush=True)
    jdump(c10, os.path.join(args.out, "c10_sets.json"))


# ====================================================================================== VM: patched gen script
C9_ARM, C9_MAIN, C9_FILE, C9_KEY, C9_KEY4 = "C9", "C9_ceil_p50", "joint_disc_L48.npz", "all11_disc_whole_L48", "four_disc_whole_L48"
C9_SECONDARY = ["C9_ceil_p75", "C9_ceil_p25", "C9_four_p50"]
PATCHES = [
    ('ARMS = ["C1a", "C1b", "C2", "C3", "C5", "C6", "C7"]', 'ARMS = ["C1a", "C1b", "C2", "C3", "C5", "C6", "C7", "C8", "C9", "C10"]'),
    ('"C7": "O-lens-filtered 100 of top-1000"}', '"C7": "O-lens-filtered 100 of top-1000", "C8": "NLA-filtered 100 of top-1000 (post hoc)", '
                                                '"C9": "Joint weird-vs-normal direction all11_disc_whole_L48 (one direction for every behavior), ceil L40-56 (post hoc)", "C10": "Affine-map O-lens-filtered 100 of top-1000 (post hoc)"}'),
    ('"C6": "damage-matched", "C7": "size-matched"}', '"C6": "damage-matched", "C7": "size-matched", "C8": "size-matched", "C9": "damage-matched", "C10": "size-matched"}'),
    ('"C7": ["olens100"]}', '"C7": ["olens100"], "C8": ["nla100"], "C9": ["ceil_p50"], "C10": ["aff100"]}'),
    ('    f = f"{args.vm}/axis.npz"\n    out["axis"] = f if os.path.exists(f) else None\n',
     '    f = f"{args.vm}/c8_sets.json"  # C8 (post hoc): c8_sets.json in c7_sets.json\'s format\n'
     '    out["c8"] = None\n'
     '    if os.path.exists(f):\n'
     '        c8s = {}\n'
     '        for b, v in json.load(open(f)).get("behaviors", {}).items():\n'
     '            if b in SETS["C7"] and isinstance(v, dict) and "neurons" in v:\n'
     '                d = collections.defaultdict(list)\n'
     '                for L, n in v["neurons"]:\n'
     '                    d[str(int(L))].append(int(n))\n'
     '                c8s[b] = {L: sorted(x) for L, x in d.items()}\n'
     '        out["c8"] = c8s or None\n'
     '        if out["c8"]:\n'
     '            out["c8_sizes"] = {b: sum(len(x) for x in v.values()) for b, v in out["c8"].items()}\n'
     '    f = f"{args.vm}/c10_sets.json"  # C10 (post hoc): affine-map O-lens-filtered 100, same format\n'
     '    out["c10"] = None\n'
     '    if os.path.exists(f):\n'
     '        c10s = {}\n'
     '        for b, v in json.load(open(f)).get("behaviors", {}).items():\n'
     '            if b in SETS["C7"] and isinstance(v, dict) and "neurons" in v:\n'
     '                d = collections.defaultdict(list)\n'
     '                for L, n in v["neurons"]:\n'
     '                    d[str(int(L))].append(int(n))\n'
     '                c10s[b] = {L: sorted(x) for L, x in d.items()}\n'
     '        out["c10"] = c10s or None\n'
     '    f = f"{args.vm}/joint_disc_L48.npz"  # C9 (post hoc): joint weird-vs-normal direction (runs/direction/joint_disc_L48.npz)\n'
     '    out["c9"] = f if os.path.exists(f) else None\n'
     '    f = f"{args.vm}/axis.npz"\n    out["axis"] = f if os.path.exists(f) else None\n'),
    ('    if args.smoke:  # 2 controls per family\n',
     '    # ---- C8 (post hoc: NLA-filtered 100 of the top-1000; controls = C7\'s, reused, so none are planned here)\n'
     '    if VM.get("c8"):\n'
     '        add("C8_nla100", "C8", "arm", {b: [{"type": "neurons", "spec": s}] for b, s in VM["c8"].items()}, rung="nla100")\n'
     '    if VM.get("c10"):\n'
     '        add("C10_aff100", "C10", "arm", {b: [{"type": "neurons", "spec": s}] for b, s in VM["c10"].items()}, rung="aff100")\n'
     '    # ---- C9 (post hoc: one joint weird-vs-normal direction for every behavior, C3\'s ceil edit over L40-56 at the benign\n'
     '    # percentile; controls = C3\'s ten matched-norm random unit directions at p50, reused). The single ladder rung is p50\n'
     '    # (C3\'s gate rung, so it is the gate / side-effect / off-target condition); p75, p25 and the four-behavior mean are\n'
     '    # secondary rungs (damage measured, 8 replies) so the fallback rung exists whichever way the gate goes.\n'
     '    if VM.get("c9"):\n'
     '        add("C9_ceil_p50", "C9", "ladder", {"*": [{"type": "ceil", "dir": "C9_all11", "layers": C3_LAYERS, "pct": 50}]}, rung="ceil_p50")\n'
     '        add("C9_ceil_p75", "C9", "secondary", {"*": [{"type": "ceil", "dir": "C9_all11", "layers": C3_LAYERS, "pct": 75}]}, rung="ceil_p75")\n'
     '        add("C9_ceil_p25", "C9", "secondary", {"*": [{"type": "ceil", "dir": "C9_all11", "layers": C3_LAYERS, "pct": 25}]}, rung="ceil_p25")\n'
     '        add("C9_four_p50", "C9", "secondary", {"*": [{"type": "ceil", "dir": "C9_four", "layers": C3_LAYERS, "pct": 50}]}, rung="four_ceil_p50")\n'
     '    if args.smoke:  # 2 controls per family\n'),
    ('    c6 = CF["vm_prereqs"].get("c6")\n    if c6:\n',
     '    c9 = CF["vm_prereqs"].get("c9")\n'
     '    if c9 and os.path.exists(c9):\n'
     '        z = np.load(c9, allow_pickle=False)\n'
     '        for name, key in (("C9_all11", "all11_disc_whole_L48"), ("C9_four", "four_disc_whole_L48")):\n'
     '            v = z[key].astype(np.float64)\n'
     '            assert v.shape == (D,) and abs(np.linalg.norm(v) - 1) < 1e-3, (key, v.shape, np.linalg.norm(v))\n'
     '            DIRS[name] = {l: unit(v) for l in C3_LAYERS}\n'
     '        log("C9 joint directions from", c9, "cos(all11, four) =", round(float(DIRS["C9_all11"][48] @ DIRS["C9_four"][48]), 4))\n'
     '    c6 = CF["vm_prereqs"].get("c6")\n    if c6:\n'),
    ('{k: (v if k not in ("disp", "c7") else bool(v)) for k, v in VM.items() if k != "c6"}',
     '{k: (v if k not in ("disp", "c7", "c8", "c10") else bool(v)) for k, v in VM.items() if k != "c6"}'),
    ('if v["kind"] in ("secondary", "positive", "disp") and (v["arm"] in ARM_FILTER or v["arm"] in ("P1", "P2")):',
     'if v["kind"] in ("secondary", "positive", "disp") and (v["arm"] in ARM_FILTER or (v["arm"] in ("P1", "P2") and not args.arms)):'),
    ('    conds += [c for c, v in COND.items() if v["kind"] == "positive"]\n',
     '    conds += [c for c, v in COND.items() if v["kind"] == "positive" and not args.arms]\n'),
]


def cmd_patch(args):
    src = open(os.path.join(REPO, "scripts", "61_evalc_gen.py")).read()
    for old, new in PATCHES:
        assert src.count(old) == 1, ("patch anchor not unique", old[:80], src.count(old))
        src = src.replace(old, new)
    src = src.replace('"""Eval C generation', '"""[PATCHED COPY for the post-hoc C8 arm; made by scripts/61_evalc_nla_c8.py patch]\nEval C generation', 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.patched)), exist_ok=True)
    open(args.patched, "w").write(src)
    print(f"patched copy written to {args.patched} ({len(PATCHES)} patches)")


# ====================================================================================== local: analysis
def cmd_analyze(args):
    import numpy as np
    an = load_module("evalc_analyze", "scripts/61_evalc_analyze.py")
    pool = json.load(open(os.path.join(REPO, "runs", "evalc", "prep", "pool.json")))
    meta7 = json.load(open(os.path.join(REPO, "runs", "evalc", "judge", "judged", "conditions.json")))
    meta8 = json.load(open(os.path.join(args.out, "judged", "conditions.json")))
    rows = collections.defaultdict(list)
    files = [os.path.join(REPO, "runs", "evalc", "judge", "judged", f) for f in ("clean.jsonl", "controls.jsonl", "arms.jsonl")] + \
            [os.path.join(args.out, "judged", "arms.jsonl")]
    for fp in files:
        for line in open(fp):
            if line.strip():
                r = json.loads(line)
                rows[r["condition"]].append(r)
    ctrls = meta7["arms"]["C7"]["controls"]
    assert COND_ARM in rows, (COND_ARM, "no judged rows")
    P = pool["pool"]
    ids_common = {p["prompt_id"] for p in P if p["in_common"]}
    W, pidx = an.make_boot(P, args.boot)
    dmg8 = meta8["damage"][COND_ARM]
    dmg7 = meta7["damage"]["C7_olens100"]
    res = {"arm": ARM, "condition": COND_ARM, "rung": RUNG, "label": "NLA-filtered 100 of top-1000 (post hoc)",
           "control_family": "size-matched (C7's random 100 of the top-1000, replies reused)", "controls": ctrls,
           "n_replies": {"C8_nla100": len(rows[COND_ARM]), "clean": len(rows["clean"]), "controls": [len(rows[c]) for c in ctrls]},
           "seeds": sorted({r["seed_j"] for r in rows[COND_ARM]}),
           "prompts": {"common": len({r["prompt_id"] for r in rows[COND_ARM]} & ids_common), "other": len({r["prompt_id"] for r in rows[COND_ARM]} - ids_common)}}
    st = an.arm_stats(rows, "clean", COND_ARM, ctrls, ids_common, W, pidx, dmg8)
    cd = [meta7["damage"][c] for c in ctrls if meta7["damage"].get(c) is not None]
    st["ctrl_damage_mean"], st["ctrl_damage_max"] = float(np.mean(cd)), float(np.max(cd))
    res["main"] = st
    chk = an.arm_stats(rows, "clean", "C7_olens100", ctrls, ids_common, W, pidx, dmg7)
    res["c7_check"] = {k: chk[k] for k in ("clean", "arm4", "arm8", "raw_drop", "raw_drop_lo", "raw_drop_hi", "ctrl_q", "excess", "excess_lo", "excess_hi", "null_excess", "boot_p", "rank_p")}
    # C7 vs C8 head to head: the same statistic with the O-lens arm as the control
    res["c8_vs_c7"] = {k: v for k, v in an.arm_stats(rows, "clean", COND_ARM, ["C7_olens100"], ids_common, W, pidx, dmg8).items()
                       if k in ("arm4", "ctrl_q", "excess", "excess_lo", "excess_hi", "boot_p_raw")}
    res["by"] = {}
    for b in sorted({p["behavior"] for p in P if p["in_common"]}, key=lambda b: -sum(p["behavior"] == b for p in P)):
        ids_b = {p["prompt_id"] for p in P if p["behavior"] == b}
        if len(ids_b) >= an.MIN_BEH_PROMPTS:
            sb = an.arm_stats(rows, "clean", COND_ARM, ctrls, ids_b, W, pidx, dmg8)
            s7 = an.arm_stats(rows, "clean", "C7_olens100", ctrls, ids_b, W, pidx, dmg7)
            res["by"][b] = {"status": "per-behavior (exploratory)", **{k: sb[k] for k in sb if k != "control_rates"},
                            "c7_arm8": s7["arm8"], "c7_excess": s7["excess"]}
        else:
            sc, nc = an.prompt_rates(rows, "clean", 10 ** 6, ids_b)
            sa, na = an.prompt_rates(rows, COND_ARM, an.J_ARM, ids_b)
            s7, n7 = an.prompt_rates(rows, "C7_olens100", an.J_ARM, ids_b)
            wc, wa = an.wilson(sum(sc.values()), sum(nc.values())), an.wilson(sum(sa.values()), sum(na.values()))
            res["by"][b] = {"status": "exploratory (Wilson on replies)", "n_prompts": len(ids_b), "clean": wc[0], "clean_lo": wc[1], "clean_hi": wc[2],
                            "arm8": wa[0], "arm8_lo": wa[1], "arm8_hi": wa[2], "raw_drop": (wc[0] - wa[0]) / wc[0] if wc[0] else None,
                            "c7_arm8": sum(s7.values()) / max(1, sum(n7.values()))}
    se = an.side_effects(rows, [COND_ARM, "C7_olens100", "clean"], {"damage": {**meta7["damage"], **meta8["damage"]},
                                                                   "side_effects": {**meta7.get("side_effects", {}), **meta8.get("side_effects", {})}})
    res["side_effects"] = {r["condition"]: {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items() if k != "condition"} for r in se.to_dict("records")}
    by_id = {p["prompt_id"]: p for p in P}
    ot = an.off_target(rows, [COND_ARM, "C7_olens100", "clean"], pool["common_behaviors"], by_id)
    res["off_target"] = ot.to_dict("records")
    c8 = json.load(open(os.path.join(args.out, "c8_sets.json")))
    res["overlap"] = {b: {"with_c1a_top100": v["overlap_top100"], "with_c7_olens100": v["overlap_olens100"], "with_c7_controls": v["overlap_controls"],
                          "nla_score_cutoff_100th": v["distribution"]["cutoff_score_100th"], "n_at_cutoff": v["distribution"]["n_at_cutoff"],
                          "n_ge25": v["distribution"]["n_ge25"], "n_gt50": v["distribution"]["n_gt50"]} for b, v in c8["behaviors"].items()}
    res["recognition"] = json.load(open(os.path.join(args.out, "recognition.json")))
    res["damage"] = {"C8_nla100": dmg8, "C8_per_behavior": meta8["arms"][ARM].get("damage_max_over_behaviors"), "C7_olens100": dmg7,
                     "gate": an.GATE, "ok": dmg8 <= an.GATE}
    res["gpu_seconds_evalc2"] = meta8.get("gpu_seconds")
    res["haiku_tokens"] = c8["haiku_tokens"]
    res["note"] = ("Post hoc arm: not in the pre-registered Holm family of runs/evalc/judge/eval; same statistic, bootstrap seed, prompts, "
                   "seeds and controls as C7; controls' judged replies reused from runs/evalc/judge/judged/controls.jsonl.")

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return None if np.isnan(x) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    res = clean(res)
    jdump(res, os.path.join(args.out, "results.json"))
    m = res["main"]
    print(f"C8_nla100: clean {m['clean']:.3f} arm8 {m['arm8']:.3f} [{m['arm8_lo']:.3f}, {m['arm8_hi']:.3f}] arm4 {m['arm4']:.3f} "
          f"raw drop {m['raw_drop']:.3f} [{m['raw_drop_lo']:.3f}, {m['raw_drop_hi']:.3f}] ctrl q {m['ctrl_q']:.3f} ({m['ctrl_min']:.3f}-{m['ctrl_max']:.3f}, N={m['n_controls']}) "
          f"excess {m['excess']:.3f} [{m['excess_lo']:.3f}, {m['excess_hi']:.3f}] null {m['null_excess']:.3f} centred p {m['boot_p']:.4f} "
          f"rank p {m['rank_p']:.3f} pctile {m['arm_pctile']:.2f} damage {m['damage']:.4f}")
    print("C7 check:", json.dumps(res["c7_check"]))
    for b, v in res["by"].items():
        print(f"  {b}: clean {v['clean']:.3f} arm8 {v['arm8']:.3f} drop {v['raw_drop']:.3f}" + (f" excess {v['excess']:.3f} p {v['boot_p']:.3f}" if "excess" in v else "") + f" (C7 arm8 {v['c7_arm8']:.3f})")


def cmd_analyze_c9(args):
    """Section-0 statistic for the post-hoc C9 arm (one joint weird-vs-normal direction, ceil L40-56 at benign p50, applied to
    every behavior's prompts) against C3's ten matched-norm random unit directions (C3_ctrl_00..09, ceiled at p50, judged
    replies reused), plus the C9 secondary rungs (p75, p25, four-behavior mean at p50), a head-to-head with C3 (own direction
    per behavior), per-behavior rows, side effects, off-target rates and the cosines of the joint direction with the seven
    C3 directions -> <out>/results_c9.json."""
    import numpy as np
    an = load_module("evalc_analyze", "scripts/61_evalc_analyze.py")
    pool = json.load(open(os.path.join(REPO, "runs", "evalc", "prep", "pool.json")))
    sets = json.load(open(os.path.join(REPO, "runs", "evalc", "prep", "sets.json")))
    meta7 = json.load(open(os.path.join(REPO, "runs", "evalc", "judge", "judged", "conditions.json")))
    meta9 = json.load(open(os.path.join(args.out, "judged", "conditions.json")))
    rows = collections.defaultdict(list)
    files = [os.path.join(REPO, "runs", "evalc", "judge", "judged", f) for f in ("clean.jsonl", "controls.jsonl", "arms.jsonl")] + \
            [os.path.join(args.out, "judged", "arms.jsonl")]
    for fp in files:
        for line in open(fp):
            if line.strip():
                r = json.loads(line)
                rows[r["condition"]].append(r)
    ctrls = meta7["arms"]["C3"]["controls"]
    assert C9_MAIN in rows, (C9_MAIN, "no judged rows")
    P = pool["pool"]
    ids_common = {p["prompt_id"] for p in P if p["in_common"]}
    ids_pool = {p["prompt_id"] for p in P}
    W, pidx = an.make_boot(P, args.boot)
    dmg = {**meta7["damage"], **meta9["damage"]}
    conds9 = [c for c in [C9_MAIN] + C9_SECONDARY if c in rows]
    res = {"arm": C9_ARM, "condition": C9_MAIN, "rung": "ceil_p50", "label": meta9["arms"][C9_ARM]["label"],
           "direction": {"file": "runs/direction/joint_disc_L48.npz", "key": C9_KEY, "secondary_key": C9_KEY4,
                         "edit": "ceil over layers 40-56 at the benign percentile (C3's recipe; tau per layer = percentile of <h, v> over "
                                 "the response tokens of P1's 24 benign default replies, 61_evalc_gen.py ensure_bproj / tau_of)"},
           "control_family": "C3's matched-norm random unit directions, ceil L40-56 at p50 (replies reused)", "controls": ctrls,
           "n_replies": {c: len(rows[c]) for c in conds9} | {"clean": len(rows["clean"]), "controls": [len(rows[c]) for c in ctrls]},
           "seeds": sorted({r["seed_j"] for r in rows[C9_MAIN]}),
           "prompts": {"common": len({r["prompt_id"] for r in rows[C9_MAIN]} & ids_common), "other": len({r["prompt_id"] for r in rows[C9_MAIN]} - ids_common)}}
    st = an.arm_stats(rows, "clean", C9_MAIN, ctrls, ids_common, W, pidx, dmg[C9_MAIN])
    cd = [dmg[c] for c in ctrls if dmg.get(c) is not None]
    st["ctrl_damage_mean"], st["ctrl_damage_max"] = float(np.mean(cd)), float(np.max(cd))
    res["main"] = st
    res["main_all53"] = {k: v for k, v in an.arm_stats(rows, "clean", C9_MAIN, ctrls, ids_pool, W, pidx, dmg[C9_MAIN]).items() if k != "control_rates"}
    chk = an.arm_stats(rows, "clean", "C3_ceil_p50", ctrls, ids_common, W, pidx, dmg["C3_ceil_p50"])
    keys = ("clean", "arm4", "arm8", "raw_drop", "raw_drop_lo", "raw_drop_hi", "ctrl_q", "excess", "excess_lo", "excess_hi", "null_excess", "boot_p", "rank_p", "damage")
    res["c3_check"] = {k: chk[k] for k in keys}
    res["secondary"] = {c: {k: v for k, v in an.arm_stats(rows, "clean", c, ctrls, ids_common, W, pidx, dmg.get(c)).items() if k != "control_rates"}
                        for c in C9_SECONDARY if c in rows}
    # head to head: C9 against C3 as the single "control" (same prompts, seeds j < 4 on both sides), and the reverse
    hh = lambda a, b: {k: v for k, v in an.arm_stats(rows, "clean", a, [b], ids_common, W, pidx, dmg.get(a)).items()  # noqa: E731
                       if k in ("arm4", "arm8", "ctrl_q", "excess", "excess_lo", "excess_hi", "boot_p_raw")}
    res["c9_vs_c3"] = {"C9_ceil_p50 vs C3_ceil_p50": hh(C9_MAIN, "C3_ceil_p50"), "C3_ceil_p50 vs C9_ceil_p50": hh("C3_ceil_p50", C9_MAIN),
                       "note": "excess = (other arm's j<4 rate - this arm's j<4 rate) / clean; positive = this arm drops more; boot_p_raw one-sided"}
    res["by"] = {}
    for b in sorted({p["behavior"] for p in P}, key=lambda b: -sum(p["behavior"] == b for p in P)):
        ids_b = {p["prompt_id"] for p in P if p["behavior"] == b}
        if len(ids_b) >= an.MIN_BEH_PROMPTS:
            sb = an.arm_stats(rows, "clean", C9_MAIN, ctrls, ids_b, W, pidx, dmg[C9_MAIN])
            s3 = an.arm_stats(rows, "clean", "C3_ceil_p50", ctrls, ids_b, W, pidx, dmg["C3_ceil_p50"])
            res["by"][b] = {"status": "per-behavior (exploratory)", **{k: sb[k] for k in sb if k != "control_rates"},
                            "c3_arm8": s3["arm8"], "c3_excess": s3["excess"], "c3_raw_drop": s3["raw_drop"]}
        else:
            sc, nc = an.prompt_rates(rows, "clean", 10 ** 6, ids_b)
            sa, na = an.prompt_rates(rows, C9_MAIN, an.J_ARM, ids_b)
            s3, n3 = an.prompt_rates(rows, "C3_ceil_p50", an.J_ARM, ids_b)
            wc, wa = an.wilson(sum(sc.values()), sum(nc.values())), an.wilson(sum(sa.values()), sum(na.values()))
            res["by"][b] = {"status": "exploratory (Wilson on replies)", "n_prompts": len(ids_b), "clean": wc[0], "clean_lo": wc[1], "clean_hi": wc[2],
                            "arm8": wa[0], "arm8_lo": wa[1], "arm8_hi": wa[2], "raw_drop": (wc[0] - wa[0]) / wc[0] if wc[0] else None,
                            "c3_arm8": sum(s3.values()) / max(1, sum(n3.values()))}
    meta_se = {"damage": dmg, "side_effects": {**meta7.get("side_effects", {}), **meta9.get("side_effects", {})}}
    se = an.side_effects(rows, conds9 + ["C3_ceil_p50", "clean"], meta_se)
    res["side_effects"] = {r["condition"]: {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items() if k != "condition"} for r in se.to_dict("records")}
    by_id = {p["prompt_id"]: p for p in P}
    res["off_target"] = an.off_target(rows, [C9_MAIN, "C3_ceil_p50", "clean"], pool["common_behaviors"], by_id).to_dict("records")
    g = meta9["arms"][C9_ARM]
    res["damage"] = {c: dmg.get(c) for c in [C9_MAIN] + C9_SECONDARY} | {"C3_ceil_p50": dmg["C3_ceil_p50"], "gate": an.GATE, "p50_ok": dmg[C9_MAIN] <= an.GATE,
                                                                        "gate_rule": g.get("gate_rule"), "none_passed": g.get("none_passed"), "over_gate": g.get("over_gate")}
    # cosines of the joint direction with the seven C3 directions (and the four-behavior mean)
    z = np.load(os.path.join(REPO, "runs", "direction", "joint_disc_L48.npz"), allow_pickle=False)
    j11, j4 = z[C9_KEY].astype(np.float64), z[C9_KEY4].astype(np.float64)
    cos = {"all11_vs_four": float(j11 @ j4 / np.linalg.norm(j11) / np.linalg.norm(j4))}
    for b, c in sets["C3"].items():
        f = os.path.join(REPO, c["file"])
        if os.path.exists(f):
            v = np.load(f, allow_pickle=False)[c["key"]].astype(np.float64)
            v /= np.linalg.norm(v)
            cos[sets["short"][b]] = {"all11": float(j11 @ v), "four": float(j4 @ v)}
    res["cosines"] = cos
    res["gpu_seconds_evalc2"] = meta9.get("gpu_seconds")
    res["note"] = ("Post hoc arm: not in the pre-registered Holm family of runs/evalc/judge/eval; same statistic, bootstrap seed, prompts and "
                   "seeds as C3; controls' judged replies reused from runs/evalc/judge/judged/controls.jsonl (C3_ctrl_*, each ceiled at its own "
                   "benign p50). The secondary rungs (p75, p25, four) are compared with the same p50 controls, so their excess is not percentile-matched.")

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return None if np.isnan(x) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    res = clean(res)
    jdump(res, os.path.join(args.out, "results_c9.json"))
    m = res["main"]
    print(f"{C9_MAIN}: clean {m['clean']:.3f} arm8 {m['arm8']:.3f} [{m['arm8_lo']:.3f}, {m['arm8_hi']:.3f}] arm4 {m['arm4']:.3f} "
          f"raw drop {m['raw_drop']:.3f} [{m['raw_drop_lo']:.3f}, {m['raw_drop_hi']:.3f}] ctrl q {m['ctrl_q']:.3f} ({m['ctrl_min']:.3f}-{m['ctrl_max']:.3f}, N={m['n_controls']}) "
          f"excess {m['excess']:.3f} [{m['excess_lo']:.3f}, {m['excess_hi']:.3f}] null {m['null_excess']:.3f} centred p {m['boot_p']:.2e} "
          f"rank p {m['rank_p']:.3f} pctile {m['arm_pctile']:.2f} damage {m['damage']:.4f}")
    print("C3 check:", json.dumps(res["c3_check"]))
    for c, v in res["secondary"].items():
        print(f"  {c}: arm8 {v['arm8']:.3f} drop {v['raw_drop']:.3f} excess {v['excess']:.3f} p {v['boot_p']:.2e} damage {v['damage']}")
    print("C9 vs C3:", json.dumps(res["c9_vs_c3"]))
    for b, v in res["by"].items():
        print(f"  {b}: clean {v['clean']:.3f} arm8 {v['arm8']:.3f} drop {v['raw_drop']:.3f}" + (f" excess {v['excess']:.3f} p {v['boot_p']:.3f}" if "excess" in v else "") + f" (C3 arm8 {v['c3_arm8']:.3f})")
    print("cosines", json.dumps(res["cosines"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["read", "score", "assemble", "assemble_aff", "patch", "analyze", "analyze_c9"])
    ap.add_argument("--out", default=None, help="VM: /data/runs/evalc/nla_c8; local: runs/evalc/nla_c8")
    ap.add_argument("--pre", default="/data/runs/evalc/pre")
    ap.add_argument("--nla_dir", default=os.environ.get("NLA_DIR", "/data/nla_dl"))
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--max_minutes", type=float, default=150)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-in", type=float, default=4e6)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--patched", default="/data/evalc_c8/61_evalc_gen_c8.py")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    if a.out is None:
        a.out = "/data/runs/evalc/nla_c8" if a.cmd == "read" else OUT_LOCAL
    {"read": cmd_read, "score": cmd_score, "assemble": cmd_assemble, "assemble_aff": cmd_assemble_aff, "patch": cmd_patch, "analyze": cmd_analyze, "analyze_c9": cmd_analyze_c9}[a.cmd](a)
