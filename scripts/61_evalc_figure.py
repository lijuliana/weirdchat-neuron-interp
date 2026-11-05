"""figures/evalc_control.svg: eval C drop and excess per method (README figure). Rows: the seven pre-registered arms
(runs/evalc/judge/eval/main_table.csv), the derived random-100-of-top-1000 row (random_pool.json) and, when present,
the post-hoc arms analysed by scripts/61_evalc_nla_c8.py (runs/evalc/nla_c8/results*.json: C8 NLA-picked, C10 affine-map
O-lens-picked, C9 joint direction). Post-hoc and derived rows are drawn faded. Run: .venv/bin/python -I scripts/61_evalc_figure.py"""
import json, os
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
mt = pd.read_csv("runs/evalc/judge/eval/main_table.csv").set_index("arm")
rows = {a: dict(drop=r.raw_drop, lo=r.raw_drop_lo, hi=r.raw_drop_hi, exc=r.excess, elo=r.excess_lo, ehi=r.excess_hi, dmg=r.damage, faded=False) for a, r in mt.iterrows()}
R = json.load(open("runs/evalc/judge/eval/random_pool.json"))
rows["R"] = dict(drop=R["raw_drop"], lo=R["raw_drop_lo"], hi=R["raw_drop_hi"], exc=R["excess"], elo=R["excess_lo"], ehi=R["excess_hi"], dmg=R["damage"], faded=True)
for path, arm in (("runs/evalc/nla_c8/results.json", "C8"), ("runs/evalc/nla_c8/results_c10.json", "C10"), ("runs/evalc/nla_c8/results_c9.json", "C9")):
    if os.path.exists(path):
        m = json.load(open(path))["main"]
        rows[arm] = dict(drop=m["raw_drop"], lo=m["raw_drop_lo"], hi=m["raw_drop_hi"], exc=m["excess"], elo=m["excess_lo"], ehi=m["excess_hi"], dmg=m["damage"], faded=True)
short = {"C1a": "Top-100 neurons\n(attribution)", "R": "Random 100 of the\ntop-1000 (10 draws)", "C3": "Weird-vs-normal\ndirection", "C9": "One joint direction\nfor all behaviors",
         "C2": "Assistant Axis\ncap", "C6": "J-lens keyword\ndirection", "C1b": "Shared mid-layer\n100 neurons", "C5": "SAE features",
         "C7": "O-lens-picked 100\nof top-1000", "C10": "Affine O-lens-picked\n100 of top-1000", "C8": "NLA-picked 100\nof top-1000"}
order = [a for a in ["C1a", "R", "C3", "C9", "C2", "C6", "C1b", "C5", "C7", "C10", "C8"] if a in rows]
fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(order) + 0.9), constrained_layout=True)
y = list(range(len(order)))[::-1]
hot, acc = "#b5451b", "#2b5f8a"
for yi, a in zip(y, order):
    r = rows[a]; al = 0.45 if r["faded"] else 1.0
    ax.plot([r["lo"], r["hi"]], [yi + 0.17, yi + 0.17], color=hot, lw=1, alpha=.6 * al)
    ax.plot(r["drop"], yi + 0.17, "o", color=hot, ms=5, alpha=al, label="Drop in behavior rate" if yi == y[0] else None)
    ax.plot([r["elo"], min(r["ehi"], 1.3)], [yi - 0.17, yi - 0.17], color=acc, lw=1, alpha=.6 * al)
    ax.plot(r["exc"], yi - 0.17, "s", color=acc, ms=5, alpha=al, label="Drop beyond matched random controls" if yi == y[0] else None)
    ax.text(1.32, yi, f"{r['dmg']:.4f}", va="center", ha="left", fontsize=8, color="#555")
ax.axvline(0.5, color="#999", lw=.8, ls="--"); ax.text(0.5, len(order) - 0.45, "bar: 50%", fontsize=8, color="#777", ha="center")
ax.set_yticks(y); ax.set_yticklabels([short[a] for a in order], fontsize=8)
ax.set_xlim(-0.1, 1.3); ax.set_xticks([0, .25, .5, .75, 1.0]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of the clean behavior rate removed on 48 held-out prompts", fontsize=9)
ax.text(1.32, len(order) - 0.45, "damage\n(nats/token)", fontsize=8, color="#555", ha="left", va="bottom")
ax.axvline(0, color="#bbb", lw=.6)
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.savefig("figures/evalc_control.svg"); print("saved", order)
