"""Within-cut AUROC (eval A, set b, k = 4) split into the 34 sharp forks and the 466 other cut points, with 95%
bootstrap intervals over cuts, for the monitors the per-method figures need. The harness (scripts/55) reports the
sharp/other split as point estimates only.

Run from the repo root: .venv/bin/python -I scripts/63_sharp_ci.py
Writes runs/monitors/eval/sharp_ci.json (copied to findings/readouts_notes/sharp_ci.json).
"""
import importlib.util, json, os, shutil, sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(REPO)
spec = importlib.util.spec_from_file_location("me", "scripts/55_monitor_eval.py")
me = importlib.util.module_from_spec(spec); spec.loader.exec_module(me)

WANT = {"text_b": ["text"], "m3": None, "m6a": None, "m2": ["fold_on_whole_L48"], "jlens": None, "nla": None, "olens": None}
rng = np.random.default_rng(0)
ctx = me.Ctx(200)
methods = me.load_methods(ctx, list(WANT))
d = ctx.B["b"][4]; T = d["T"]
sharp = T.drop_duplicates("cut_id").set_index("cut_id").sharp
out, per = {}, {}
for m in methods:
    for v, (s, _) in m["variants"].items():
        if WANT[m["stem"]] and v not in WANT[m["stem"]]:
            continue
        x = me.lookup(T.key, s, T.own)
        a = pd.Series(ctx.cut_auroc(d, x), index=d["cuts"]).dropna()
        per[f"{m['stem']}:{v}"] = a
        sh = sharp.reindex(a.index).values.astype(bool)
        row = {}
        for name, sel in (("all", np.ones(len(a), bool)), ("sharp", sh), ("other", ~sh)):
            v_ = a.values[sel]
            bo = [v_[rng.integers(0, len(v_), len(v_))].mean() for _ in range(2000)]
            row[name] = [float(v_.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)), int(len(v_))]
        out[f"{m['stem']}:{v}"] = row
# paired differences on the same cuts (bootstrap over cuts), sharp forks and all cuts
PAIRS = [("text_b:text", f"m3:pooled_L{L}") for L in (16, 24, 32, 40, 42, 44, 48, 56)] + \
        [("m6a:err_ridge", "m6a:recon_ridge"), ("m6a:raw42_ridge", "m6a:recon_ridge"), ("m6a:raw42_ridge", "m6a:err_ridge"),
         ("m6a:raw42_ridge", "m6a:k256"), ("text_b:text", "m6a:raw42_ridge")]
out["_paired"] = {}
for a_, b_ in PAIRS:
    A_, B_ = per[a_], per[b_]
    ix = A_.index.intersection(B_.index)
    sh = sharp.reindex(ix).values.astype(bool)
    r = {}
    for name, sel in (("all", np.ones(len(ix), bool)), ("sharp", sh)):
        dlt = (A_.reindex(ix) - B_.reindex(ix)).values[sel]
        bo = np.array([dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(4000)])
        r[name] = [float(dlt.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)), float((bo <= 0).mean()), int(len(dlt))]
    out["_paired"][f"{a_} - {b_}"] = r
    print(f"{a_} - {b_}: all {r['all'][0]:+.3f} [{r['all'][1]:+.3f},{r['all'][2]:+.3f}] p {r['all'][3]:.3f} | sharp {r['sharp'][0]:+.3f} [{r['sharp'][1]:+.3f},{r['sharp'][2]:+.3f}] p {r['sharp'][3]:.3f}")
p = "runs/monitors/eval/sharp_ci.json"
json.dump(out, open(p, "w"), indent=1)
os.makedirs("findings/readouts_notes", exist_ok=True)
shutil.copy(p, "findings/readouts_notes/sharp_ci.json")
for k, r in out.items():
    if k.startswith("_"):
        continue
    print(f"{k:32s} all {r['all'][0]:.3f}  sharp {r['sharp'][0]:.3f} [{r['sharp'][1]:.3f},{r['sharp'][2]:.3f}] n={r['sharp'][3]}  other {r['other'][0]:.3f}")
