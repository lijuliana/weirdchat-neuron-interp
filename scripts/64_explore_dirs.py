"""Exploratory figures on the weird-vs-normal directions and the Assistant Axis / persona vectors (not about monitor
AUROC or eval C). Reads saved outputs only (CPU):
  runs/persona_fwd/a1/{directions/*.npz, reply_means.npz}  weird-vs-normal directions and reply-mean states (blocks 16-56)
  runs/persona_p1/vectors/                                  community Assistant Axis, default and 265 role means, traits
  runs/persona_fwd/screens.json                             role and trait screens with label-shuffle nulls
Writes figures/fig_x_dir_roles, fig_x_dir_traits, fig_x_axis_position (.pdf/.svg) and runs/explore/dirs_axis.json.

Run: .venv/bin/python -I scripts/64_explore_dirs.py
"""
import importlib.util, json, os, collections

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("figs62", os.path.join(REPO, "scripts/62_figures.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
build, f, esc = F.build, F.f, F.esc

V = os.path.join(REPO, "runs/persona_p1/vectors")
DEF = np.load(f"{V}/default_raw.npy"); ROLES = np.load(f"{V}/roles_raw.npy"); AX = np.load(f"{V}/axis_unit.npy")
NAMES = json.load(open(f"{V}/role_names.json"))["roles"]
BL_AX = [8, 16, 24, 32, 40, 48, 56]; BL = [16, 24, 32, 40, 48, 56]
SCR = json.load(open(os.path.join(REPO, "runs/persona_fwd/screens.json")))
SHORT = {"dai": "Denying AI", "911": "Called 911", "laser": "Laser at aircraft", "embodiment": "Claiming a body",
         "drunk": "Drunk driving", "code": "Fake code execution", "chemtrails": "Chemtrails", "cutting": "Cutting",
         "purging": "Purging", "sexual": "Sexual advances", "calorie": "Calorie restriction"}
LONG = {"recommends-drunk-driving": "drunk", "chemtrails-assertion": "chemtrails", "cutting-instructions": "cutting",
        "laser-at-aircraft": "laser", "claims-called-911": "911", "denying-ai-identity": "dai", "fabricated-code-execution": "code",
        "false-physical-embodiment": "embodiment", "purging-advice": "purging", "unsolicited-sexual-advances": "sexual",
        "extreme-calorie-restriction": "calorie"}
OUT = {}


def role_scores(b, kind):
    """s_i = -mean over blocks 16..56 of cos(w_L, unit(default_L - role_i,L)); > 0 means the weird direction points toward role i."""
    d = np.load(os.path.join(REPO, f"runs/persona_fwd/a1/directions/{b}.npz"))
    s, sa = np.zeros(len(NAMES)), 0.0
    for L in BL:
        w = d[f"on16_{kind}_whole_L{L}"].astype(np.float64); w /= np.linalg.norm(w); i = BL_AX.index(L)
        dd = DEF[i][None] - ROLES[:, i]; dd /= np.linalg.norm(dd, axis=1, keepdims=True)
        s += -(dd @ w) / len(BL); sa += -(AX[i] @ w) / len(BL)
    return s, sa


# ------------------------------------------------------------------------------------------- 1. roles per behavior
def fig_roles():
    panels = ["dai", "911", "laser", "embodiment", "drunk", "code"]
    res = {}
    L = [r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=3 by 2, horizontal sep=2.3cm, vertical sep=1.5cm},"
         r" width=4.6cm, height=3.9cm, xmin=0, xmax=0.56, xtick={0,0.2,0.4}, xticklabel style={/pgf/number format/.cd, fixed, precision=1},"
         r" ymin=-4.6, ymax=0.6, ytick={0,-1,-2,-3,-4}, y tick label style={font=\footnotesize}, major y tick style={draw=none},"
         r" axis x line*=bottom, axis y line*=left, xmajorgrids, title style={font=\small, yshift=-2pt}]"]
    for k, b in enumerate(panels):
        s, sa = role_scores(b, "disc"); s2, sa2 = role_scores(b, "ho")
        top = list(np.argsort(-s)[:5]); rank_ho = {i: int((s2 > s2[i]).sum()) + 1 for i in top}
        null = SCR["a2_cos"][b]["raw"]["max_null95_shuffle"]
        res[b] = {"top": [(NAMES[i], float(s[i]), float(s2[i]), rank_ho[i]) for i in top], "axis_disc": float(sa), "axis_ho": float(sa2),
                  "null95_max_shuffle": null, "r_disc_ho_265": float(np.corrcoef(s, s2)[0, 1]),
                  "n_disc_prompts_mixed": SCR["a2_cos"][b]["raw"].get("n_disc_prompts_mixed")}
        labels = ",".join("{" + esc(NAMES[i]) + "}" for i in top)
        xl = r", xlabel={Lean toward the role (cosine)}" if k >= 3 else ""
        L.append(rf"\nextgroupplot[title={{{SHORT[b]}}}, yticklabels={{{labels}}}{xl}]")
        L.append(rf"\fill[black!8] (axis cs:0,-4.6) rectangle (axis cs:{f(null,3)},0.6);")
        for j, i in enumerate(top):
            L.append(rf"\fill[dirc!75] (axis cs:0,{-j-0.28}) rectangle (axis cs:{f(s[i],3)},{-j+0.28});")
            L.append(rf"\addplot[only marks, mark=o, mark size=2.1pt, ink, line width=0.8pt] coordinates {{({f(s2[i],3)},{-j})}};")
            L.append(rf"\node[font=\scriptsize, black!60, anchor=west] at (axis cs:{f(max(s[i], s2[i]) + 0.01,3)},{-j}) {{{rank_ho[i]}}};")
        L.append(rf"\draw[pers, dashed, line width=1pt] (axis cs:{f(sa,3)},-4.6) -- (axis cs:{f(sa,3)},0.6);")
    L += [r"\end{groupplot}",
          r"\node[font=\footnotesize, anchor=south, align=center] at ($(group c2r1.north)+(0,0.7cm)$) {"
          r"\tikz\fill[dirc!75] (0,0) rectangle (0.35,0.18); discovery prompts\qquad"
          r" \tikz\draw[ink, line width=0.8pt] (0,0) circle (2.1pt); held-out prompts (number: rank among 265 roles)\\"
          r" \tikz\draw[pers, dashed, line width=1pt] (0,-0.1) -- (0,0.2); Assistant Axis itself\qquad"
          r" \tikz\fill[black!8] (0,0) rectangle (0.35,0.18); 95th pct.\ of the top role with labels shuffled};",
          r"\end{tikzpicture}"]
    build("fig_x_dir_roles", "\n".join(L))
    OUT["roles"] = res


# ------------------------------------------------------------------------------------------- 2. trait vectors
def fig_traits():
    order = ["drunk", "embodiment", "dai", "911", "code", "laser", "chemtrails", "cutting", "purging", "calorie", "sexual"]
    traits = ["sycophantic", "optimistic", "helpful", "hallucinating", "humorous", "impolite", "evil", "dismissive", "apathetic"]
    stops = [(-0.5, (178, 24, 43)), (-0.2, (244, 165, 130)), (0.0, (247, 247, 247)), (0.2, (146, 197, 222)), (0.55, (33, 102, 172))]
    def rgb(v):
        v = max(-0.5, min(0.55, v))
        for (a, ca), (b_, cb) in zip(stops, stops[1:]):
            if v <= b_:
                t = (v - a) / (b_ - a); return tuple(int(round(ca[k] + t * (cb[k] - ca[k]))) for k in range(3))
        return stops[-1][1]
    res = {}
    n_prompts = {LONG[k]: v for k, v in F.C["dirs11"]["n_prompts"].items()}
    L = [r"\begin{tikzpicture}",
         rf"\begin{{axis}}[width=10cm, height=10.4cm, xmin=-0.5, xmax={len(traits)-0.5}, ymin={-len(order)+0.5}, ymax=0.5,"
         rf" xtick={{{','.join(str(j) for j in range(len(traits)))}}}, xticklabels={{{','.join(traits)}}},"
         r" x tick label style={rotate=40, anchor=north east, font=\footnotesize},"
         rf" ytick={{{','.join(str(-i) for i in range(len(order)))}}}, yticklabels={{{','.join(SHORT[b] + ' (' + str(n_prompts[b]) + ')' for b in order)}}},"
         r" y tick label style={font=\footnotesize}, tick style={draw=none}, axis line style={draw=none}, enlargelimits=false,"
         r" xlabel={Persona trait vector}]"]
    for i, b in enumerate(order):
        r = SCR["a3_cos"][b]["response_avg"]; res[b] = {}
        for j, t in enumerate(traits):
            v, beyond, nul = r[t]["cos_mean"], r[t]["beyond95"], r[t]["null_abs95"]
            res[b][t] = {"cos": v, "null_abs95": nul, "beyond": beyond}
            c = rgb(v)
            L.append(rf"\fill[fill={{rgb,255:red,{c[0]};green,{c[1]};blue,{c[2]}}}] (axis cs:{j-0.5},{-i-0.5}) rectangle (axis cs:{j+0.5},{-i+0.5});")
            txt = (f"{v:+.2f}".replace("0.", ".")) + ("$^{*}$" if beyond else "")
            L.append(rf"\node[font=\scriptsize, text={'white' if abs(v) > 0.38 else 'black'}] at (axis cs:{j},{-i}) {{{txt}}};")
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_x_dir_traits", "\n".join(L))
    OUT["traits"] = res


# ------------------------------------------------------------------------------------------- 3. position on the axis
def fig_position(L_=48):
    z = np.load(os.path.join(REPO, "runs/persona_fwd/a1/reply_means.npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"])); X = z["whole"][:, BL.index(L_)].astype(np.float32)
    ia = BL_AX.index(L_); u = AX[ia]; pd_ = DEF[ia] @ u; gap = pd_ - (ROLES[:, ia] @ u).mean()
    pos = (pd_ - X @ u) / gap
    roles_pos = (pd_ - ROLES[:, ia] @ u) / gap
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: {"w": [], "n": []}))
    for m, p in zip(meta, pos):
        if m["on"] is None:
            continue
        by[LONG[m["b"]]][m["prompt_id"]]["w" if m["on"] else "n"].append(float(p))
    rng = np.random.default_rng(0); res = {}
    for b, P in by.items():
        mixed = [(np.mean(v["w"]), np.mean(v["n"])) for v in P.values() if v["w"] and v["n"]]
        norm_means = np.array([np.mean(v["n"]) for v in P.values() if v["n"]])
        if len(mixed) < 8:
            continue
        dlt = np.array([a - c for a, c in mixed])
        bo = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(4000)]
        res[b] = {"n_mixed": len(mixed), "n_prompts": len(norm_means), "normal_mean": float(norm_means.mean()),
                  "normal_p10": float(np.percentile(norm_means, 10)), "normal_p90": float(np.percentile(norm_means, 90)),
                  "shift": float(dlt.mean()), "shift_lo": float(np.percentile(bo, 2.5)), "shift_hi": float(np.percentile(bo, 97.5))}
    order = sorted(res, key=lambda b: res[b]["normal_mean"])
    OUT["position"] = {"block": L_, "gap": float(gap), "roles_p5_p50_p95": [float(x) for x in np.percentile(roles_pos, [5, 50, 95])],
                       "per_behavior": res}
    Lx = [r"\begin{tikzpicture}",
          rf"\begin{{axis}}[width=11cm, height={0.62*len(order)+2.6:.2f}cm, xmin=-0.15, xmax=1.95, xtick={{0,0.5,1,1.5}},"
          r" xticklabels={0,.5,1,1.5}, ymin=" + f"{-len(order)+0.3}" + r", ymax=2.3,"
          rf" ytick={{{','.join(str(-i) for i in range(len(order)))}}}, yticklabels={{{','.join(SHORT[b] for b in order)}}},"
          r" y tick label style={font=\footnotesize}, major y tick style={draw=none}, axis x line*=bottom, axis y line*=left, xmajorgrids,"
          r" xlabel={Position on the Assistant Axis, block 48 (0 = default assistant, 1 = mean of 265 roles)}]"]
    # role positions as a rug at the top
    rug = " ".join(f"({f(p,3)},1.15)" for p in roles_pos)
    Lx.append(rf"\addplot[only marks, mark=|, mark size=3pt, pers!55, line width=0.4pt] coordinates {{{rug}}};")
    Lx.append(r"\node[font=\scriptsize, pers!85!black, anchor=south west] at (axis cs:-0.13,1.28) {265 role-play personas};")
    for x, lab in ((0, "assistant"), (1, "mean role")):
        Lx.append(rf"\draw[black!45, dashed, line width=0.7pt] (axis cs:{x},{-len(order)+0.3}) -- (axis cs:{x},0.75);")
    for i, b in enumerate(order):
        r = res[b]; y = -i
        Lx.append(rf"\draw[black!14, line width=5pt] (axis cs:{f(r['normal_p10'],3)},{y}) -- (axis cs:{f(r['normal_p90'],3)},{y});")
        a, c = r["normal_mean"], r["normal_mean"] + r["shift"]
        Lx.append(rf"\addplot[only marks, mark=o, mark size=2.3pt, ink, line width=0.8pt] coordinates {{({f(a,3)},{y})}};")
        Lx.append(rf"\addplot[only marks, mark=*, mark size=2.3pt, dirc] coordinates {{({f(c,3)},{y})}};")
        Lx.append(rf"\node[font=\scriptsize, anchor=west, dirc!80!black] at (axis cs:{f(max(r['normal_p90'], c, a)+0.05,3)},{y}) "
                  rf"{{{r['shift']:+.2f} [{r['shift_lo']:+.2f}, {r['shift_hi']:+.2f}]}};")
    Lx += [r"\node[font=\scriptsize, anchor=south west, align=left] at (axis cs:-0.13,1.75) {\tikz\draw[ink, line width=0.8pt] (0,0) circle (2.3pt); normal replies (mean)\quad"
           r" \tikz\fill[dirc] (0,0) circle (2.3pt); + within-prompt shift of weird replies\quad \tikz\fill[black!14] (0,0) rectangle (0.35,0.15); 10th to 90th pct.\ of prompts};",
           r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_x_axis_position", "\n".join(Lx))


if __name__ == "__main__":
    fig_roles(); fig_traits(); fig_position()
    os.makedirs(os.path.join(REPO, "runs/explore"), exist_ok=True)
    json.dump(OUT, open(os.path.join(REPO, "runs/explore/dirs_axis.json"), "w"), indent=1)
    print(json.dumps({k: OUT["roles"][k]["top"][:3] for k in OUT["roles"]}, indent=0)[:1500])
    for b, r in OUT["position"]["per_behavior"].items():
        print(b, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
