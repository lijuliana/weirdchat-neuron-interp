"""Paper figures in LaTeX (pgfplots). Collects the data from the analysis outputs, writes one standalone .tex per figure
into figures/tex/ (shared style figures/tex/style.tex), compiles each with pdflatex and converts it to SVG with
pdftocairo (glyphs as paths), writing figures/<name>.pdf and figures/<name>.svg. The viewer and the README use the SVGs.

Data sources: viewer/out_public/compare.json (scripts/53), runs/evalc/judge/{judged,eval} (eval C, pre-registered),
runs/evalc/nla_c8/ (post-hoc eval C arms), runs/evalc/olens_a/results.json and runs/evalc/nla_c8/recognition.json
(neuron recognition by lens), runs/direction/transfer_L48.json.

Run: .venv/bin/python -I scripts/62_figures.py [--only name,name]
"""
import argparse, collections, importlib.util, json, math, os, subprocess, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TEX, OUT = os.path.join(REPO, "figures", "tex"), os.path.join(REPO, "figures")
C = json.load(open(os.path.join(REPO, "viewer", "out_public", "compare.json")))

STYLE = r"""\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots,fillbetween}
\usetikzlibrary{calc}
% one colour per kind of method, the same in every figure (Okabe-Ito)
\definecolor{neur}{HTML}{D55E00}   % neurons chosen by attribution
\definecolor{dirc}{HTML}{009E73}   % directions from the weird-vs-normal contrast
\definecolor{read}{HTML}{0072B2}   % readout methods and interventions chosen from readouts
\definecolor{pers}{HTML}{CC79A7}   % Assistant Axis / persona
\definecolor{ctrl}{HTML}{8C8C8C}   % random controls
\definecolor{meas}{HTML}{E69F00}   % measured outcome
\definecolor{ink}{HTML}{222222}
\pgfplotsset{
  every axis/.append style={clip=false, font=\small, axis line style={black!55}, tick style={black!55}, tick align=outside,
    label style={font=\small}, tick label style={font=\footnotesize}, title style={font=\small, align=center}},
  grid style={black!9, line width=0.4pt},
  every axis legend/.append style={font=\footnotesize, draw=none, fill=none},
  ci/.style={/pgfplots/error bars/x dir=both, /pgfplots/error bars/x explicit, /pgfplots/error bars/error bar style={line width=0.8pt}, /pgfplots/error bars/error mark=none},
  yci/.style={/pgfplots/error bars/y dir=both, /pgfplots/error bars/y explicit, /pgfplots/error bars/error bar style={line width=0.8pt}, /pgfplots/error bars/error mark=none},
}
"""


def f(x, d=4):
    return f"{x:.{d}f}"


def doc(body):
    return "\\documentclass[border=4pt]{standalone}\n\\input{style.tex}\n\\begin{document}\n" + body + "\n\\end{document}\n"


def build(name, body):
    os.makedirs(TEX, exist_ok=True)
    open(os.path.join(TEX, "style.tex"), "w").write(STYLE)
    p = os.path.join(TEX, name + ".tex")
    open(p, "w").write(doc(body))
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], cwd=TEX,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit(f"pdflatex failed for {name}")
    os.replace(os.path.join(TEX, name + ".pdf"), os.path.join(OUT, name + ".pdf"))
    r = subprocess.run(["pdftocairo", "-svg", os.path.join(OUT, name + ".pdf"), os.path.join(OUT, name + ".svg")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit(f"pdftocairo failed for {name}")
    for ext in ("aux", "log"):
        q = os.path.join(TEX, f"{name}.{ext}")
        if os.path.exists(q):
            os.remove(q)
    print("built", name)


def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def esc(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def rows_plot(rows, header_fmt=r"\textbf{%s}"):
    """rows: list of ('h', label) or ('r', label, key). Returns (ys, ticks, labels) with y descending from 0."""
    ys, ticks, labels, y = {}, [], [], 0.0
    for r in rows:
        if r[0] == "h":
            y -= 0.35 if ticks else 0
            ticks.append(y); labels.append(header_fmt % esc(r[1])); y -= 1
        else:
            ys[r[2]] = y; ticks.append(y); labels.append(esc(r[1])); y -= 1
    return ys, ticks, labels, y


# ============================================================================================ eval C: stopping
def evalc_data():
    an = load_module("evalc_analyze", "scripts/61_evalc_analyze.py")
    pool = json.load(open(os.path.join(REPO, "runs/evalc/prep/pool.json")))
    meta = json.load(open(os.path.join(REPO, "runs/evalc/judge/judged/conditions.json")))
    meta_p = json.load(open(os.path.join(REPO, "runs/evalc/nla_c8/judged/conditions.json")))
    rows = collections.defaultdict(list)
    for fp in [os.path.join(REPO, "runs/evalc/judge/judged", x) for x in ("clean.jsonl", "controls.jsonl", "arms.jsonl")] + \
              [os.path.join(REPO, "runs/evalc/nla_c8/judged/arms.jsonl")]:
        for line in open(fp):
            if line.strip():
                r = json.loads(line); rows[r["condition"]].append(r)
    P = pool["pool"]; ids = {p["prompt_id"] for p in P if p["in_common"]}
    W, pidx = an.make_boot(P, 1000)
    dmg = {**meta["damage"], **meta_p["damage"]}
    A = meta["arms"]
    # the 10 random 100-of-top-1000 draws (C7's controls) pooled as one derived arm
    rows["R"] = [r for c in A["C7"]["controls"] for r in rows[c]]
    dmg["R"] = sum(dmg[c] for c in A["C7"]["controls"]) / len(A["C7"]["controls"])
    spec = {"C1a": (A["C1a"]["gate"], A["C1a"]["controls"]), "R": ("R", A["C1a"]["controls"]),
            "C1b": (A["C1b"]["gate"], A["C1b"]["controls"]), "C3": (A["C3"]["gate"], A["C3"]["controls"]),
            "C9": ("C9_ceil_p50", A["C3"]["controls"]), "C5": (A["C5"]["gate"], A["C5"]["controls"]),
            "C6": (A["C6"]["gate"], A["C6"]["controls"]), "C7": (A["C7"]["gate"], A["C7"]["controls"]),
            "C10": ("C10_aff100", A["C7"]["controls"]), "C8": ("C8_nla100", A["C7"]["controls"]),
            "C2": (A["C2"]["gate"], A["C2"]["controls"])}
    out = {}
    for k, (cond, ctrls) in spec.items():
        st = an.arm_stats(rows, meta["clean"], cond, ctrls, ids, W, pidx, dmg.get(cond))
        out[k] = {"clean": st["clean"], "arm4": st["arm4"], "lo": st["arm4_lo"], "hi": st["arm4_hi"], "ctrl": st["control_rates"],
                  "ctrl_q": st["ctrl_q"], "excess": st["excess"], "damage": dmg[cond], "ctrl_dmg": [dmg[c] for c in ctrls if c in dmg],
                  "arm8": st["arm8"], "drop": st["raw_drop"]}
    return out


EVALC_ROWS = [("h", "Neurons chosen by attribution"), ("r", "Top-100 neurons", "C1a"), ("r", "Random 100 of the top 1,000", "R"),
              ("r", "Shared mid-layer 100", "C1b"),
              ("h", "Directions from the weird-vs-normal contrast"), ("r", "Own direction per behavior", "C3"),
              ("r", "One direction for all behaviors", "C9"),
              ("h", "Chosen from a readout"), ("r", "SAE features picked by label", "C5"), ("r", "J-lens keyword direction", "C6"),
              ("r", "O-lens-picked 100 of the top 1,000", "C7"), ("r", "Affine O-lens-picked 100", "C10"), ("r", "NLA-picked 100", "C8"),
              ("h", "Persona"), ("r", "Assistant Axis cap", "C2")]
GROUP_COLOR = {"C1a": "neur", "R": "neur", "C1b": "neur", "C3": "dirc", "C9": "dirc", "C5": "read", "C6": "read", "C7": "read",
               "C10": "read", "C8": "read", "C2": "pers"}


def fig_evalc(D):
    ys, ticks, labels, ymin = rows_plot(EVALC_ROWS)
    clean = 100 * D["C1a"]["clean"]
    H = 0.52 * (len(ticks) + 1.5)
    L = []
    L.append(r"\begin{tikzpicture}")
    L.append(r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=0.45cm, y descriptions at=edge left},"
             rf" height={H:.2f}cm, ymin={ymin + 0.4:.2f}, ymax=0.6, ytick={{{','.join(f(t, 2) for t in ticks)}}},"
             rf" yticklabels={{{','.join('{' + l + '}' for l in labels)}}}, y tick label style={{font=\footnotesize}},"
             r" major y tick style={draw=none}, axis y line*=left, axis x line*=bottom, xmajorgrids]")
    # left: behavior rate
    L.append(r"\nextgroupplot[width=9.6cm, xmin=0, xmax=37, xtick={0,10,20,30}, xticklabel={$\pgfmathprintnumber{\tick}\%$},"
             r" xlabel={Behavior rate with the edit (48 held-out prompts)},"
             r" legend style={at={(0.5,1.02)}, anchor=south, legend columns=2, /tikz/every even column/.append style={column sep=6pt}},"
             r" legend cell align=left]")
    L.append(rf"\draw[black!45, dashed, line width=0.7pt] (axis cs:{f(clean,2)},{ymin+0.4:.2f}) -- (axis cs:{f(clean,2)},0.6);")
    L.append(r"\addlegendimage{only marks, mark=*, mark size=2.2pt, black}\addlegendentry{rate with the edit, 95\% CI}")
    L.append(r"\addlegendimage{only marks, mark=o, mark size=1.4pt, ctrl}\addlegendentry{matched random edits}")
    L.append(r"\addlegendimage{only marks, mark=|, mark size=4pt, line width=1.1pt, black}\addlegendentry{rate that removes 50\% beyond the worst random edit}")
    L.append(r"\addlegendimage{dashed, black!45, line width=0.7pt}\addlegendentry{no edit (" + f"{clean:.0f}" + r"\%)}")
    for k, y in ys.items():
        d = D[k]
        pts = " ".join(f"({f(100*c,2)},{f(y,2)})" for c in d["ctrl"])
        L.append(rf"\addplot[only marks, mark=o, mark size=1.4pt, ctrl, forget plot] coordinates {{{pts}}};")
        thr = 100 * (d["ctrl_q"] - 0.5 * d["clean"])
        if thr > 0:
            L.append(rf"\addplot[only marks, mark=|, mark size=4pt, line width=1.1pt, black, forget plot] coordinates {{({f(thr,2)},{f(y,2)})}};")
        else:
            L.append(rf"\node[font=\scriptsize, anchor=west, black!70] at (axis cs:0.3,{f(y+0.02,2)}) {{bar below 0}};")
        x, lo, hi = 100 * d["arm4"], 100 * d["lo"], 100 * d["hi"]
        L.append(rf"\addplot[only marks, mark=*, mark size=2.2pt, {GROUP_COLOR[k]}, ci, forget plot] coordinates "
                 rf"{{({f(x,2)},{f(y,2)}) += ({f(hi-x,2)},0) -= ({f(x-lo,2)},0)}};")
    # right: damage
    L.append(r"\nextgroupplot[width=4.6cm, xmode=log, xmin=4e-5, xmax=0.08, xtick={1e-4,1e-3,1e-2},"
             r" xticklabels={$10^{-4}$,$10^{-3}$,$10^{-2}$}, xlabel={Damage (nats/token)}, yticklabels={}]")
    L.append(rf"\draw[black!45, dashed, line width=0.7pt] (axis cs:0.02,{ymin+0.4:.2f}) -- (axis cs:0.02,0.6) node[above, font=\scriptsize, black!70] {{gate .02}};")
    for k, y in ys.items():
        d = D[k]
        if d["ctrl_dmg"]:
            pts = " ".join(f"({max(c, 5e-5):.6f},{f(y,2)})" for c in d["ctrl_dmg"])
            L.append(rf"\addplot[only marks, mark=o, mark size=1.4pt, ctrl] coordinates {{{pts}}};")
        L.append(rf"\addplot[only marks, mark=*, mark size=2.2pt, {GROUP_COLOR[k]}] coordinates {{({max(d['damage'],5e-5):.6f},{f(y,2)})}};")
    L.append(r"\end{groupplot}")
    L.append(r"\end{tikzpicture}")
    build("fig_evalc", "\n".join(L))


# ============================================================================================ eval A: monitors
def fig_monitors():
    M = C["mon"]
    ref = next(r for r in M["rows"] if r["grp"] == "ref")
    groups = [("Neurons", "neur"), ("Directions and probes", "dirc"), ("Readouts", "read"), ("Baselines", "ctrl")]
    rows = []
    for g, _ in groups:
        rr = [r for r in M["rows"] if r["grp"] == g and r.get("auroc")]
        if rr:
            rows.append(("h", g)); rows += [("r", r["lab"].replace("Weird-vs-normal direction", "Weird-vs-normal direction"), r["id"]) for r in rr]
    ys, ticks, labels, ymin = rows_plot(rows)
    byid = {r["id"]: r for r in M["rows"]}
    col = {r["id"]: dict(groups)[r["grp"]] for r in M["rows"] if r["grp"] in dict(groups)}
    H = 0.48 * (len(ticks) + 2)
    L = [r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=0.9cm, y descriptions at=edge left},"
         rf" height={H:.2f}cm, ymin={ymin+0.4:.2f}, ymax=0.6, ytick={{{','.join(f(t,2) for t in ticks)}}},"
         rf" yticklabels={{{','.join('{' + l + '}' for l in labels)}}}, y tick label style={{font=\footnotesize}},"
         r" major y tick style={draw=none}, axis y line*=left, axis x line*=bottom, xmajorgrids]"]
    tj, ceil = ref["auroc"][0], M["ceil"]
    L.append(r"\nextgroupplot[width=7.6cm, xmin=0.45, xmax=0.92, xtick={0.5,0.6,0.7,0.8,0.9},"
             r" xlabel={AUROC, weird vs.\ normal continuation},"
             r" title={(a) Warning at 500 cut points}]")
    L.append(r"\fill[black!6] (axis cs:" + f(tj, 3) + f",{ymin+0.4:.2f}) rectangle (axis cs:" + f(ceil, 3) + ",0.6);")
    L.append(rf"\node[font=\scriptsize, black!60, align=center] at (axis cs:{f((tj+ceil)/2,3)},{f((ymin+0.4)/2,2)}) {{not reached by\\any monitor}};")
    for x, lab, st, anc in ((0.5, "", "black!40", "west"), (tj, "text judge", "black", "west"), (ceil, "ceiling (judge noise)", "black!55", "east")):
        L.append(rf"\draw[{st}, dashed, line width=0.7pt] (axis cs:{f(x,3)},{ymin+0.4:.2f}) -- (axis cs:{f(x,3)},0.6);")
        if lab:
            L.append(rf"\node[{st}, font=\scriptsize, anchor={anc}] at (axis cs:{f(x,3)},0) {{{lab}}};")
    for k, y in ys.items():
        r = byid[k]; m, lo, hi = r["auroc"]
        L.append(rf"\addplot[only marks, mark=*, mark size=2pt, {col[k]}, ci] coordinates {{({f(m,3)},{f(y,2)}) += ({f(hi-m,3)},0) -= ({f(m-lo,3)},0)}};")
    L.append(r"\nextgroupplot[width=6.2cm, xmin=-0.12, xmax=0.2, xtick={-0.1,0,0.1,0.2}, scaled x ticks=false,"
             r" xticklabel style={/pgf/number format/.cd, fixed, precision=1}, xlabel={Correlation with P(weird), position-corrected},"
             r" title={(b) Tracking the chance of weird within a reply}]")
    tp = ref["partial"][0]
    L.append(rf"\draw[black!35, dashed, line width=0.7pt] (axis cs:0,{ymin+0.4:.2f}) -- (axis cs:0,0.6);")
    L.append(rf"\draw[black, dashed, line width=0.7pt] (axis cs:{f(tp,3)},{ymin+0.4:.2f}) -- (axis cs:{f(tp,3)},0.6);")
    L.append(rf"\node[font=\scriptsize, anchor=west] at (axis cs:{f(tp,3)},0) {{text judge}};")
    for k, y in ys.items():
        r = byid[k]
        if r.get("partial"):
            m, lo, hi = r["partial"]
            L.append(rf"\addplot[only marks, mark=*, mark size=2pt, {col[k]}, ci] coordinates {{({f(m,3)},{f(y,2)}) += ({f(hi-m,3)},0) -= ({f(m-lo,3)},0)}};")
    L += [r"\end{groupplot}", r"\end{tikzpicture}"]
    build("fig_monitors", "\n".join(L))


# ============================================================================================ eval A: fork-aligned
def fig_fork():
    F = C["fork"]; rel = F["rel"]
    def band(vals, colr, dash=""):
        pts = [(r, v) for r, v in zip(rel, vals) if v]
        up = " ".join(f"({r},{f(v[2],3)})" for r, v in pts); dn = " ".join(f"({r},{f(v[1],3)})" for r, v in pts)
        mid = " ".join(f"({r},{f(v[0],3)})" for r, v in pts)
        return (rf"\addplot[name path=u, draw=none, forget plot] coordinates {{{up}}};"
                rf"\addplot[name path=d, draw=none, forget plot] coordinates {{{dn}}};"
                rf"\addplot[{colr}, fill opacity=0.15, forget plot] fill between[of=u and d];"
                rf"\addplot[{colr}, line width=1.1pt{', ' + dash if dash else ''}] coordinates {{{mid}}};")
    b, a = F["P"]["before"], F["P"]["after"]
    L = [r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=1 by 3, vertical sep=0.55cm, x descriptions at=edge bottom},"
         r" width=11cm, height=3.6cm, xmin=-20, xmax=10, xtick={-20,-15,-10,-5,0,5,10}, axis x line*=bottom, axis y line*=left,"
         r" ymajorgrids, legend style={at={(1.02,0.5)}, anchor=west}, legend cell align=left]",
         r"\nextgroupplot[ymin=0, ymax=1, ytick={0,0.5,1}, yticklabels={0\%,50\%,100\%}, ylabel={Chance of weird}]",
         rf"\addplot[meas, line width=1.4pt, const plot] coordinates {{(-20,{f(b[0],3)}) (0,{f(a[0],3)}) (10,{f(a[0],3)})}};\addlegendentry{{measured}}",
         band(F["text"], "black", "dashed") + r"\addlegendentry{text judge}",
         r"\draw[black!50, dotted, line width=0.8pt] (axis cs:0,0) -- (axis cs:0,1);",
         r"\nextgroupplot[ymin=0, ymax=10, ytick={0,5,10}, ylabel={O-lens score}]",
         band(F["lens"], "read") + r"\addlegendentry{reply read in the chat}",
         band(F["raw"], "read!45", "dashed") + r"\addlegendentry{same words as a plain document}",
         r"\draw[black!50, dotted, line width=0.8pt] (axis cs:0,0) -- (axis cs:0,10);"]
    av = [v for v in F["axis"] if v]; lo = math.floor(min(v[1] for v in av) - 1); hi = math.ceil(max(v[2] for v in av) + 1)
    L += [rf"\nextgroupplot[ymin={lo}, ymax={hi}, ylabel={{Axis projection}}, xlabel={{Tokens relative to the fork token (0 = the state that chooses it)}}]",
          band(F["axis"], "pers") + r"\addlegendentry{Assistant Axis, block 48}",
          rf"\draw[black!50, dotted, line width=0.8pt] (axis cs:0,{lo}) -- (axis cs:0,{hi});",
          r"\end{groupplot}", r"\end{tikzpicture}"]
    build("fig_fork", "\n".join(L))


# ============================================================================================ direction tab
SHORTB = {"recommends-drunk-driving": "Drunk driving", "chemtrails-assertion": "Chemtrails", "cutting-instructions": "Cutting",
          "laser-at-aircraft": "Laser at aircraft", "claims-called-911": "Called 911", "denying-ai-identity": "Denying AI",
          "fabricated-code-execution": "Fake code exec.", "false-physical-embodiment": "Claiming a body", "purging-advice": "Purging",
          "unsolicited-sexual-advances": "Sexual advances", "extreme-calorie-restriction": "Calorie restriction"}


def fig_dir_own():
    D = C["dirs11"]; J = D["joint"]
    order = sorted(D["behaviors"], key=lambda b: -(D["own"][b]["cv"] or 0))
    rows = [("r", f"{SHORTB[b]} ({D['n_prompts'][b]})", b) for b in order]
    ys, ticks, labels, ymin = rows_plot(rows)
    H = 0.5 * (len(ticks) + 2.5)
    L = [r"\begin{tikzpicture}",
         rf"\begin{{axis}}[width=10.5cm, height={H:.2f}cm, ymin={ymin+0.5:.2f}, ymax=0.5, ytick={{{','.join(f(t,2) for t in ticks)}}},"
         rf" yticklabels={{{','.join('{' + l + '}' for l in labels)}}}, major y tick style={{draw=none}}, axis y line*=left, axis x line*=bottom,"
         r" xmin=0.4, xmax=1.0, xmajorgrids, xlabel={Within-prompt AUROC, weird vs.\ normal replies (0.5 = chance)},"
         r" legend style={at={(0.5,1.02)}, anchor=south, legend columns=2, /tikz/every even column/.append style={column sep=6pt}}, legend cell align=left]",
         rf"\draw[black!35, dashed, line width=0.7pt] (axis cs:0.5,{ymin+0.5:.2f}) -- (axis cs:0.5,0.5);"]
    def ser(get, style, name, dy):
        pts = " ".join(f"({f(get(b),3)},{f(ys[b]+dy,2)})" for b in order if get(b) is not None)
        return rf"\addplot[only marks, {style}] coordinates {{{pts}}};\addlegendentry{{{name}}}"
    L.append(ser(lambda b: D["own"][b]["cv"], "mark=*, mark size=2.4pt, dirc", "own direction", 0.14))
    L.append(ser(lambda b: J["all11_cv"][b], "mark=square*, mark size=2.1pt, ink", "one direction fit on all eleven", 0.0))
    L.append(ser(lambda b: D["lobo"][b]["auroc"], "mark=diamond*, mark size=2.5pt, read", "mean of the other ten (behavior never seen)", -0.14))
    nul = " ".join(f"({f(D['own'][b]['cv_shuf95'],3)},{f(ys[b],2)})" for b in order if D["own"][b]["cv_shuf95"])
    L.append(rf"\addplot[only marks, mark=|, mark size=5pt, line width=1pt, ctrl] coordinates {{{nul}}};\addlegendentry{{95th pct.\ with labels shuffled}}")
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_dir_own", "\n".join(L))


def fig_dir_transfer():
    D = C["dirs11"]; B = D["behaviors"]; n = len(B)
    S = {"recommends-drunk-driving": "Drunk", "chemtrails-assertion": "Chemtrails", "cutting-instructions": "Cutting", "laser-at-aircraft": "Laser",
         "claims-called-911": "911", "denying-ai-identity": "Deny AI", "fabricated-code-execution": "Code", "false-physical-embodiment": "Body",
         "purging-advice": "Purging", "unsolicited-sexual-advances": "Sexual", "extreme-calorie-restriction": "Calorie"}
    stops = [(0.3, (178, 24, 43)), (0.4, (244, 165, 130)), (0.5, (247, 247, 247)), (0.75, (146, 197, 222)), (1.0, (33, 102, 172))]
    def rgb(v):
        v = max(0.3, min(1.0, v))
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            if v <= b:
                t = (v - a) / (b - a)
                return tuple(int(round(ca[k] + t * (cb[k] - ca[k]))) for k in range(3))
        return stops[-1][1]
    cells = []
    for i, a in enumerate(B + ["lobo"]):
        for j, b in enumerate(B):
            v = D["cv_diag"][b] if a == b else (D["lobo"][b]["auroc"] if a == "lobo" else D["transfer"][a][b])
            cells.append((j, -(i + (0.4 if a == "lobo" else 0)), v))
    ylab = [SHORTB[b] for b in B] + ["Mean of the other ten"]
    yt = [-i for i in range(n)] + [-(n + 0.4)]
    cmap = "; ".join(f"rgb255({int(round((v-0.3)/0.7*1000))})=({c[0]},{c[1]},{c[2]})" for v, c in stops)
    L = [r"\begin{tikzpicture}",
         rf"\begin{{axis}}[width=11.5cm, height=10.6cm, xmin=-0.5, xmax={n-0.5}, ymin={-(n+0.9):.2f}, ymax=0.5,"
         rf" xtick={{{','.join(str(j) for j in range(n))}}}, xticklabels={{{','.join(S[b] for b in B)}}}, x tick label style={{rotate=40, anchor=north east, font=\footnotesize}},"
         rf" ytick={{{','.join(f(t,2) for t in yt)}}}, yticklabels={{{','.join(ylab)}}}, y tick label style={{font=\footnotesize}},"
         r" xlabel={Behavior whose replies are scored}, ylabel={Direction fit on}, tick style={draw=none}, axis line style={draw=none},"
         rf" colormap={{rb}}{{{cmap}}}, point meta min=0.3, point meta max=1.0, colorbar,"
         r" colorbar style={width=0.25cm, ytick={0.3,0.5,0.7,0.9,1.0}, ylabel={Within-prompt AUROC}, ylabel style={font=\footnotesize}, yticklabel style={font=\scriptsize}, axis line style={draw=black!40}},"
         r" enlargelimits=false]",
         r"\addplot[draw=none, mark=none, point meta=0.5, forget plot] coordinates {(0,0)};"]
    for j, y, v in cells:
        r_, g_, b_ = rgb(v)
        L.append(rf"\fill[fill={{rgb,255:red,{r_};green,{g_};blue,{b_}}}] (axis cs:{j-0.5},{f(y-0.5,2)}) rectangle (axis cs:{j+0.5},{f(y+0.5,2)});")
        txt = "1.0" if v >= 0.995 else f(v, 2)[1:]
        L.append(rf"\node[font=\scriptsize, text={'white' if abs(v-0.5) > 0.3 else 'black'}] at (axis cs:{j},{f(y,2)}) {{{txt}}};")
    for i in range(n):
        L.append(rf"\draw[black, line width=1pt] (axis cs:{i-0.5},{-i-0.5}) rectangle (axis cs:{i+0.5},{-i+0.5});")
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_dir_transfer", "\n".join(L))


def fig_dir_prefix():
    D = C["dirs11"]
    order = sorted([b for b in D["behaviors"] if D["prefix"].get(b) and sum(v is not None for v in D["prefix"][b]["auc"]) > 3],
                   key=lambda b: -(D["own"][b]["cv"] or 0))
    cols = 5; rws = math.ceil(len(order) / cols)
    L = [r"\begin{tikzpicture}",
         rf"\begin{{groupplot}}[group style={{group size={cols} by {rws}, horizontal sep=0.3cm, vertical sep=1.25cm, y descriptions at=edge left, x descriptions at=edge bottom}},"
         r" width=3.75cm, height=3.3cm, xmode=log, log basis x=2, xmin=1, xmax=1024, xtick={1,8,64,512}, xticklabels={1,8,64,512}, ymin=0.4, ymax=1.0, ytick={0.5,0.75,1},"
         r" yticklabels={.5,.75,1}, axis x line*=bottom, axis y line*=left, ymajorgrids, title style={font=\footnotesize, align=center, yshift=-3pt}]"]
    for k, b in enumerate(order):
        d = D["prefix"][b]
        pts = " ".join(f"({t},{f(v,3)})" for t, v in zip(d["t"], d["auc"]) if v is not None)
        nul = " ".join(f"({t},{f(v,3)})" for t, v in zip(d["t"], d["null"]) if v is not None)
        L.append(rf"\nextgroupplot[title={{{SHORTB[b]}\\[-1pt]{{\scriptsize\color{{black!60}}{D['n_prompts'][b]} prompts}}}}{', ylabel={AUROC}' if k % cols == 0 else ''}]")
        L.append(r"\draw[black!30, line width=0.5pt] (axis cs:1,0.5) -- (axis cs:1024,0.5);")
        if nul:
            L.append(rf"\addplot[ctrl, dashed, line width=0.8pt] coordinates {{{nul}}};")
        L.append(rf"\addplot[dirc, line width=1.2pt, mark=*, mark size=1.1pt] coordinates {{{pts}}};")
    L += [r"\end{groupplot}",
          rf"\node[font=\small, anchor=north] at ($(group c3r{rws}.south)+(0,-0.7cm)$) {{Reply tokens read (log scale)}};",
          r"\end{tikzpicture}"]
    build("fig_dir_prefix", "\n".join(L))


# ============================================================================================ Assistant Axis tab
def fig_axis():
    X = C["axiscap"]
    beh = [("denying-ai-identity", "Denying AI", "*"), ("false-physical-embodiment", "Body", "square*"),
           ("fabricated-code-execution", "Fake code", "triangle*"), ("laser-at-aircraft", "Laser", "diamond*")]
    rungs = ["C2_p25", "C2_p50", "C2_p75"]
    dm = [0.0] + [X["ladder"][r]["damage"] for r in rungs]
    xl = ",".join("{" + n + r"\\[-2pt]{\scriptsize\color{black!55}" + (f"{d:.3f}".lstrip("0") if d else "0") + "}}" for n, d in zip(["none", "25th", "50th", "75th"], dm))
    L = [r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=3.5cm}, height=6.2cm, axis x line*=bottom, axis y line*=left, ymajorgrids]",
         rf"\nextgroupplot[width=6.0cm, xmin=-0.2, xmax=3.2, xtick={{0,1,2,3}}, xticklabels={{{xl}}}, xticklabel style={{align=center}}, ymin=0, ymax=45,"
         r" ytick={0,10,20,30,40}, yticklabel={$\pgfmathprintnumber{\tick}\%$}, ylabel={Behavior rate},"
         r" xlabel={Cap strength (benign-chat percentile; damage in nats/token)}, title={(a) Tightening the cap}]",
         r"\fill[black!6] (axis cs:1.5,0) rectangle (axis cs:3.2,45);",
         r"\node[font=\scriptsize, black!55, anchor=north] at (axis cs:2.35,44.5) {over the damage gate};"]
    ends = []
    for b, lab, mk in beh:
        vals = [X["ladder"][rungs[0]]["by"][b]["clean"]] + [X["ladder"][r]["by"][b]["arm8"] for r in rungs]
        col = "ctrl" if "laser" in b else "pers"
        L.append(rf"\addplot[{col}, line width=0.9pt, mark={mk}, mark size=1.7pt] coordinates {{{' '.join(f'({i},{f(100*v,2)})' for i, v in enumerate(vals))}}};")
        ends.append([100 * vals[-1], lab, col])
    pv = [X["clean"]] + [X["ladder"][r]["arm8"] for r in rungs]
    ci = [(0, 0)] + [(100 * (X["ladder"][r]["hi"] - X["ladder"][r]["arm8"]), 100 * (X["ladder"][r]["arm8"] - X["ladder"][r]["lo"])) for r in rungs]
    L.append(rf"\addplot[black, line width=1.8pt, mark=square*, mark size=1.9pt, yci] coordinates {{{' '.join(f'({i},{f(100*v,2)}) += (0,{f(c[0],2)}) -= (0,{f(c[1],2)})' for i, (v, c) in enumerate(zip(pv, ci)))}}};")
    ends.append([100 * pv[-1], r"\textbf{all four}", "black"])
    ends.sort(key=lambda e: -e[0])
    for i in range(1, len(ends)):
        if ends[i - 1][0] - ends[i][0] < 3.4:
            ends[i][0] = ends[i - 1][0] - 3.4
    for yv, lab, col in ends:
        L.append(rf"\node[font=\scriptsize, {col}, anchor=west] at (axis cs:3.42,{f(yv,2)}) {{{lab}}};")
    # (b) removal against damage
    pts = [q for q in X["points"] if q["damage"] is not None and q["kind"] != "secondary"]
    L.append(r"\nextgroupplot[width=8.2cm, xmode=log, xmin=4e-5, xmax=0.06, xtick={1e-4,1e-3,1e-2}, xticklabels={$10^{-4}$,$10^{-3}$,$10^{-2}$},"
             r" ymin=-25, ymax=100, ytick={0,25,50,75,100}, yticklabel={$\pgfmathprintnumber{\tick}\%$},"
             r" xlabel={Damage (nats/token, log scale)}, ylabel={Share of the behavior removed}, title={(b) Removal against damage, every edit}]")
    L.append(r"\draw[black!45, dashed, line width=0.7pt] (axis cs:0.02,-25) -- (axis cs:0.02,100) node[above, font=\scriptsize, black!70] {gate};")
    L.append(r"\draw[black!30, line width=0.5pt] (axis cs:4e-5,0) -- (axis cs:0.06,0);")
    cc = " ".join(f"({max(q['damage'],5e-5):.6f},{f(100*q['drop'],2)})" for q in pts if q["kind"] == "control" and q["arm"] != "C7")
    L.append(rf"\addplot[only marks, mark=o, mark size=1.3pt, ctrl] coordinates {{{cc}}};")
    cp = " ".join(f"({max(q['damage'],5e-5):.6f},{f(100*q['drop'],2)})" for q in pts if q["kind"] == "control" and q["arm"] == "C7")
    L.append(rf"\addplot[only marks, mark=o, mark size=1.5pt, neur!70, line width=0.7pt] coordinates {{{cp}}};")
    def ladder(arm, col, mk):
        o = sorted([q for q in pts if q["arm"] == arm and q["kind"] != "control"], key=lambda q: q["damage"])
        return rf"\addplot[{col}, line width=1.1pt, mark={mk}, mark size=2pt] coordinates {{{' '.join(f'({q['damage']:.6f},{f(100*q['drop'],2)})' for q in o)}}};"
    L.append(ladder("C1a", "neur", "*"))
    L.append(ladder("C2", "pers", "square*"))
    R9 = json.load(open(os.path.join(REPO, "runs/evalc/nla_c8/results_c9.json")))
    j9 = sorted([(R9["secondary"]["C9_ceil_p75"]["damage"], R9["secondary"]["C9_ceil_p75"]["raw_drop"]), (R9["main"]["damage"], R9["main"]["raw_drop"]),
                 (R9["secondary"]["C9_ceil_p25"]["damage"], R9["secondary"]["C9_ceil_p25"]["raw_drop"])])
    L.append(rf"\addplot[dirc, line width=1.1pt, mark=triangle*, mark size=2.3pt, dashed] coordinates {{{' '.join(f'({d:.6f},{f(100*v,2)})' for d, v in j9)}}};")
    c3 = next(q for q in pts if q["arm"] == "C3" and q["kind"] == "gate")
    L.append(rf"\addplot[only marks, dirc, mark=*, mark size=2.4pt] coordinates {{({c3['damage']:.6f},{f(100*c3['drop'],2)})}};")
    rd = [(q["damage"], q["drop"]) for q in pts if q["arm"] in ("C5", "C6", "C7") and q["kind"] == "gate"]
    for fn in ("results.json", "results_c10.json"):
        m = json.load(open(os.path.join(REPO, "runs/evalc/nla_c8", fn)))["main"]; rd.append((m["damage"], m["raw_drop"]))
    L.append(rf"\addplot[only marks, read, mark=diamond*, mark size=2.5pt] coordinates {{{' '.join(f'({max(d,5e-5):.6f},{f(100*v,2)})' for d, v in rd)}}};")
    L += [r"\node[font=\scriptsize, neur, anchor=east] at (axis cs:0.0075,90) {neurons: top 10, 100, 1,000};",
          r"\node[font=\scriptsize, neur!80, anchor=south east] at (axis cs:0.0016,63) {random 100s from the top 1,000};",
          r"\node[font=\scriptsize, read, anchor=north west] at (axis cs:1.15e-4,28) {picked by a readout};",
          r"\node[font=\scriptsize, pers, anchor=west] at (axis cs:0.047,47) {axis cap};",
          rf"\node[font=\scriptsize, dirc, anchor=west] at (axis cs:{1.15*c3['damage']:.6f},{f(100*c3['drop'],2)}) {{own direction}};",
          r"\node[font=\scriptsize, dirc, anchor=north] at (axis cs:0.0075,27) {one joint direction};",
          r"\node[font=\scriptsize, ctrl, anchor=north west] at (axis cs:5e-5,-13) {random edits};",
          r"\end{groupplot}", r"\end{tikzpicture}"]
    build("fig_axis", "\n".join(L))


# ============================================================================================ reading the neurons
def fig_lens_measure_pick():
    A = C["abl"]["all"]
    conds = [("top", "Top-100 zeroed", "neur", "*"), ("amp", "Top-100 pushed the other way", "neur", "o"),
             ("rand_s", "Random 100, size-matched", "ctrl", "*"), ("rand", "Random 100", "ctrl", "o"), ("gauss", "Random direction, same size", "ctrl", "square")]
    rec = json.load(open(os.path.join(REPO, "runs/evalc/olens_a/results.json")))["neurons"]
    nla = json.load(open(os.path.join(REPO, "runs/evalc/nla_c8/recognition.json")))["nla"]
    lenses = [("O-lens (s3d)", next(v for k, v in rec.items() if k.startswith("s3d")), "read", "*"),
              ("Affine-map O-lens", rec.get("aff_rl600"), "read!60", "square*"),
              ("O-lens B", rec.get("lensB_rl600"), "read!35", "diamond*"),
              ("NLA", nla, "black", "triangle*")]
    bnames = [("911", "Called 911"), ("dai", "Denying AI"), ("code", "Fake code exec."), ("embodiment", "Claiming a body"), ("laser", "Laser")]
    L = [r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=3.1cm}, height=5.4cm, axis x line*=bottom, axis y line*=left, xmajorgrids]",
         r"\nextgroupplot[width=6.4cm, xmin=-1.3, xmax=0.8, ymin=-4.6, ymax=0.6, ytick={0,-1,-2,-3,-4},"
         rf" yticklabels={{{','.join('{' + c[1] + '}' for c in conds)}}}, y tick label style={{font=\footnotesize}}, major y tick style={{draw=none}},"
         r" xlabel={Change in O-lens behavior score (0 to 10)}, title={(a) Reading a real state after the edit}]",
         r"\draw[black!35, dashed, line width=0.7pt] (axis cs:0,-4.6) -- (axis cs:0,0.6);"]
    for i, (k, lab, col, mk) in enumerate(conds):
        for L_, dy in (("56", 0.12), ("44", -0.12)):
            m, lo, hi, n = A[k][L_]
            L.append(rf"\addplot[only marks, mark={mk}, mark size=2pt, {col}, {'opacity=0.55,' if L_=='44' else ''} ci, line width=0.8pt] coordinates {{({f(m,3)},{f(-i+dy,2)}) += ({f(hi-m,3)},0) -= ({f(m-lo,3)},0)}};")
    L.append(r"\nextgroupplot[width=6.4cm, xmin=0.4, xmax=0.7, xtick={0.4,0.5,0.6,0.7}, ymin=-4.6, ymax=0.6, ytick={0,-1,-2,-3,-4},"
             rf" yticklabels={{{','.join('{' + b[1] + '}' for b in bnames)}}}, y tick label style={{font=\footnotesize}}, major y tick style={{draw=none}},"
             r" xlabel={AUROC, causal top-100 vs.\ the other 900}, title={(b) Picking the neurons from their directions},"
             r" legend style={at={(1.03,0.5)}, anchor=west, font=\scriptsize}, legend cell align=left]")
    L.append(r"\draw[black!35, dashed, line width=0.7pt] (axis cs:0.5,-4.6) -- (axis cs:0.5,0.6);")
    for li, (name, R, col, mk) in enumerate(lenses):
        if not R:
            continue
        dy = 0.24 - 0.16 * li
        pts = " ".join(f"({f(R[b]['auroc_top100_vs_rest'],3)},{f(-i+dy,2)})" for i, (b, _) in enumerate(bnames))
        L.append(rf"\addplot[only marks, mark={mk}, mark size=2pt, {col}] coordinates {{{pts}}};\addlegendentry{{{name}}}")
    L += [r"\end{groupplot}", r"\end{tikzpicture}"]
    build("fig_lens", "\n".join(L))


# ============================================================================================ behaviors without a set
def fig_shared():
    S = C["shared"]; P = S["behaviors"]["pooled"]
    conds = [("sharedmid100", "Shared mid-layer 100"), ("shared100", "Shared 100"), ("loo100", "Shared 100, behavior left out")]
    beh = [("drunk", "drunk driving", "*"), ("chemtrails", "chemtrails", "square*"), ("cutting", "cutting", "triangle*")]
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=10cm, height=5.4cm, xmin=0, xmax=26, xtick={0,5,10,15,20,25}, xticklabel={$\pgfmathprintnumber{\tick}\%$},"
         r" ymin=-2.7, ymax=0.7, ytick={0,-1,-2}, yticklabels={" + ",".join("{" + c[1] + "}" for c in conds) + "},"
         r" y tick label style={font=\footnotesize}, major y tick style={draw=none}, axis x line*=bottom, axis y line*=left, xmajorgrids,"
         r" xlabel={Behavior rate, pooled over drunk driving, chemtrails and cutting (46 prompts)},"
         r" legend style={at={(0.5,1.03)}, anchor=south, legend columns=3, /tikz/every even column/.append style={column sep=6pt}}, legend cell align=left]"]
    cl = P["clean"]
    L.append(rf"\fill[black!7] (axis cs:{f(100*cl[1],2)},-2.7) rectangle (axis cs:{f(100*cl[2],2)},0.7);")
    L.append(rf"\draw[black!45, dashed, line width=0.7pt] (axis cs:{f(100*cl[0],2)},-2.7) -- (axis cs:{f(100*cl[0],2)},0.7);")
    L.append(r"\addlegendimage{only marks, mark=*, mark size=2.2pt, neur}\addlegendentry{set zeroed, 95\% Wilson CI}")
    L.append(r"\addlegendimage{only marks, mark=*, mark size=2.2pt, ctrl}\addlegendentry{one random set, same layers and size}")
    L.append(r"\addlegendimage{dashed, black!45}\addlegendentry{no edit, with its 95\% CI}")
    for i, (k, lab) in enumerate(conds):
        for kk, col, dy in ((k, "neur", 0.13), ("rand_" + k, "ctrl", -0.13)):
            m, lo, hi, n = P[kk]
            L.append(rf"\addplot[only marks, mark=*, mark size=2.2pt, {col}, ci, forget plot] coordinates {{({f(100*m,2)},{f(-i+dy,2)}) += ({f(100*(hi-m),2)},0) -= ({f(100*(m-lo),2)},0)}};")
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_shared", "\n".join(L))


# ============================================================================================ eval B
def fig_chive():
    R = {r["lab"]: r for r in C["chive"]["rows"]}
    rows = [("+ neuron set", "+ random neuron set (control)", "Neuron set with attribution", "neur"),
            ("+ O-lens", "+ O-lens of the prompt read as a plain document (control)", "O-lens reading", "read"),
            ("+ Assistant Axis", "+ random direction (control)", "Assistant Axis position", "pers")]
    tj = R["transcript only"]["auroc"]
    pos = R["+ sampled outcomes (positive control)"]["auroc"]
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=10.5cm, height=5.2cm, xmin=0.74, xmax=1.0, xtick={0.75,0.8,0.85,0.9,0.95,1.0},"
         r" xticklabel style={/pgf/number format/.cd, fixed, precision=2}, ymin=-3.6, ymax=0.7, ytick={0,-1,-2,-3},"
         r" yticklabels={Transcript only,Neuron set with attribution,O-lens reading,Assistant Axis position,8 sampled outcomes (positive control)},"
         r" ytick={0,-1,-2,-3,-3.0}, y tick label style={font=\footnotesize}, major y tick style={draw=none}, axis x line*=bottom, axis y line*=left, xmajorgrids,"
         r" xlabel={Stratified AUROC: edits that change the rate vs.\ edits that do not},"
         r" legend style={at={(0.5,1.03)}, anchor=south, legend columns=3, /tikz/every even column/.append style={column sep=6pt}}, legend cell align=left]"]
    # rows: 0 transcript, -1..-3 methods, then positive control at -4 -> squeeze
    L[1] = L[1].replace("ymin=-3.6", "ymin=-4.6").replace("ytick={0,-1,-2,-3},", "").replace("ytick={0,-1,-2,-3,-3.0}", "ytick={0,-1,-2,-3,-4}")
    L.append(rf"\fill[black!7] (axis cs:{f(tj[1],3)},-4.6) rectangle (axis cs:{f(tj[2],3)},0.7);")
    L.append(rf"\draw[black, dashed, line width=0.7pt] (axis cs:{f(tj[0],3)},-4.6) -- (axis cs:{f(tj[0],3)},0.7);")
    L.append(r"\addlegendimage{only marks, mark=*, mark size=2.2pt, black}\addlegendentry{with the method's output}")
    L.append(r"\addlegendimage{only marks, mark=o, mark size=2pt, black, line width=0.8pt}\addlegendentry{with its control block}")
    L.append(r"\addlegendimage{dashed, black}\addlegendentry{transcript only, 95\% CI}")
    L.append(rf"\addplot[only marks, mark=*, mark size=2.2pt, black, ci, forget plot] coordinates {{({f(tj[0],3)},0) += ({f(tj[2]-tj[0],3)},0) -= ({f(tj[0]-tj[1],3)},0)}};")
    for i, (m, c, lab, col) in enumerate(rows):
        for key, mk, dy in ((m, "*", 0.13), (c, "o", -0.13)):
            v = R[key]["auroc"]
            L.append(rf"\addplot[only marks, mark={mk}, mark size=2.1pt, {col}, line width=0.8pt, ci, forget plot] coordinates {{({f(v[0],3)},{f(-(i+1)+dy,2)}) += ({f(v[2]-v[0],3)},0) -= ({f(v[0]-v[1],3)},0)}};")
    L.append(rf"\addplot[only marks, mark=*, mark size=2.2pt, meas, ci, forget plot] coordinates {{({f(pos[0],3)},-4) += ({f(pos[2]-pos[0],3)},0) -= ({f(pos[0]-pos[1],3)},0)}};")
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_chive", "\n".join(L))


FIGS = {"fig_evalc": lambda: fig_evalc(evalc_data()), "fig_monitors": fig_monitors, "fig_fork": fig_fork, "fig_dir_own": fig_dir_own,
        "fig_dir_transfer": fig_dir_transfer, "fig_dir_prefix": fig_dir_prefix, "fig_axis": fig_axis, "fig_lens": fig_lens_measure_pick,
        "fig_shared": fig_shared, "fig_chive": fig_chive}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for name, fn in FIGS.items():
        if not a.only or name in a.only.split(","):
            fn()
