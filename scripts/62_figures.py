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
\usetikzlibrary{calc,arrows.meta}
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
    for x, lab, st, anc in ((0.5, "", "black!40", "west"), (tj, "text judge", "black", "west"), (ceil, "ceiling: identical prefixes", "black!55", "east")):
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


# ============================================================================================ one figure per method
SHARP = None


def sharp_ci():
    global SHARP
    if SHARP is None:
        SHARP = json.load(open(os.path.join(REPO, "runs/monitors/eval/sharp_ci.json")))
    return SHARP


def by_beh():
    import csv
    return {(r["arm"], r["behavior"]): r for r in csv.DictReader(open(os.path.join(REPO, "findings/evalc/by_behavior.csv")))}


def fig_m_neurons():
    """Attribution: the top 1,000 hold the behavior; the ranking inside them adds little."""
    F = json.load(open(os.path.join(REPO, "findings/evalc/frontier.json")))["points"]
    out = [q for q in F if q["arm"] == "C1a" and q["kind"] == "control"]
    pool = [q for q in F if q["arm"] == "C7" and q["kind"] == "control"]
    rung = sorted([q for q in F if q["arm"] == "C1a" and q["kind"] in ("rung", "gate")], key=lambda q: q["damage"])
    P = lambda qs: " ".join(f"({q['damage']:.6f},{f(100*q['drop'],2)})" for q in qs)
    R = json.load(open(os.path.join(REPO, "runs/evalc/judge/eval/random_pool.json")))
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=10.5cm, height=7cm, xmode=log, xmin=1e-4, xmax=0.05, xtick={1e-4,1e-3,1e-2},"
         r" xticklabels={$10^{-4}$,$10^{-3}$,$10^{-2}$}, ymin=-25, ymax=105, ytick={0,25,50,75,100}, yticklabel={$\pgfmathprintnumber{\tick}\%$},"
         r" axis x line*=bottom, axis y line*=left, ymajorgrids,"
         r" xlabel={Damage: added next-token loss on benign text (nats/token, log scale)}, ylabel={Share of the behavior removed}]",
         r"\draw[black!30, line width=0.5pt] (axis cs:1e-4,0) -- (axis cs:0.05,0);",
         r"\draw[black!45, dashed, line width=0.7pt] (axis cs:0.02,-25) -- (axis cs:0.02,105) node[anchor=south, font=\scriptsize, black!65] {damage limit};",
         rf"\addplot[only marks, mark=o, mark size=1.7pt, ctrl, line width=0.6pt] coordinates {{{P(out)}}};",
         rf"\addplot[only marks, mark=o, mark size=2.1pt, neur, line width=0.9pt] coordinates {{{P(pool)}}};",
         rf"\addplot[neur, line width=1.3pt, mark=*, mark size=2.6pt] coordinates {{{P(rung)}}};"]
    names = {"C1a_top10": ("top 10", "north"), "C1a_top100": ("top 100", "south east"), "C1a_top1000": ("top 1,000", "east")}
    for q in rung:
        lab, anc = names[q["cond"]]
        L.append(rf"\node[font=\footnotesize\bfseries, neur, anchor={anc}] at (axis cs:{q['damage']:.6f},{f(100*q['drop'],2)}) {{\,{lab}\,}};")
    L += [rf"\node[font=\footnotesize, neur, anchor=north, align=center] at (axis cs:1.05e-3,{f(100*min(q['drop'] for q in pool)-3,2)})"
          r" {10 random draws of 100\\from the top 1,000};",
          r"\node[font=\footnotesize, ctrl, anchor=west, align=left] at (axis cs:1.5e-3,-10) {20 random sets of 100 from outside\\the top 1,000, same layers};",
          r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_m_neurons", "\n".join(L))


def fig_m_direction():
    """Weird-vs-normal direction: a direction fit on four behaviors reads behaviors it never saw."""
    D = C["dirs11"]; J = D["joint"]; seen = set(J["four_behaviors"])
    B = [b for b in D["behaviors"] if D["own"][b]["cv"] is not None]
    anc = {"claims-called-911": "east", "denying-ai-identity": "east", "unsolicited-sexual-advances": "east",
           "extreme-calorie-restriction": "west", "recommends-drunk-driving": "west", "chemtrails-assertion": "east",
           "false-physical-embodiment": "north west", "cutting-instructions": "north", "fabricated-code-execution": "north east",
           "laser-at-aircraft": "south east"}
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=8.2cm, height=8.2cm, xmin=0.5, xmax=1.0, ymin=0.5, ymax=1.0, xtick={0.5,0.6,0.7,0.8,0.9,1.0}, ytick={0.5,0.6,0.7,0.8,0.9,1.0},"
         r" xticklabel style={/pgf/number format/.cd, fixed, precision=1}, yticklabel style={/pgf/number format/.cd, fixed, precision=1},"
         r" axis x line*=bottom, axis y line*=left, grid=major,"
         r" xlabel={AUROC of the behavior's own direction}, ylabel={AUROC of one direction fit on four behaviors}]",
         r"\draw[black!40, dashed, line width=0.7pt] (axis cs:0.5,0.5) -- (axis cs:1,1) node[pos=0.2, anchor=south, font=\scriptsize, black!60, rotate=45] {same as own direction};"]
    for b in B:
        x, y = D["own"][b]["cv"], J["four_cv"][b]
        mk = "mark=*, dirc" if b in seen else "mark=o, dirc, line width=0.9pt, fill=white"
        L.append(rf"\addplot[only marks, {mk}, mark size=2.6pt] coordinates {{({f(x,3)},{f(y,3)})}};")
        L.append(rf"\node[font=\scriptsize, anchor={anc.get(b,'west')}, {'dirc' if b in seen else 'ink'}] at (axis cs:{f(x,3)},{f(y,3)}) {{\,{SHORTB[b]}\,}};")
    L += [r"\addlegendimage{only marks, mark=*, dirc, mark size=2.6pt}\addlegendentry{fit on this behavior}",
          r"\addlegendimage{only marks, mark=o, dirc, line width=0.9pt, mark size=2.6pt}\addlegendentry{never seen}",
          r"\legend{}",
          r"\node[anchor=north west, font=\footnotesize, align=left] at (axis cs:0.515,0.985) {\tikz\fill[dirc] (0,0) circle (2.6pt); fit on this behavior (4)\\[2pt]"
          r"\tikz\draw[dirc, line width=0.9pt] (0,0) circle (2.6pt); never seen by the fit (6)};",
          r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_m_direction", "\n".join(L))


def fig_m_probe():
    """Residual probe: at no depth does a probe trained on the outcome beat the transcript judge."""
    S = sharp_ci(); Ls = [16, 24, 32, 40, 42, 44, 48, 56]
    def band(key, colr, dash=""):
        v = [S[f"m3:pooled_L{l}"][key] for l in Ls]
        up = " ".join(f"({l},{f(x[2],3)})" for l, x in zip(Ls, v)); dn = " ".join(f"({l},{f(x[1],3)})" for l, x in zip(Ls, v))
        mid = " ".join(f"({l},{f(x[0],3)})" for l, x in zip(Ls, v))
        return (rf"\addplot[name path=u{key}, draw=none] coordinates {{{up}}};\addplot[name path=d{key}, draw=none] coordinates {{{dn}}};"
                rf"\addplot[{colr}, fill opacity=0.15] fill between[of=u{key} and d{key}];"
                rf"\addplot[{colr}, line width=1.3pt, mark=*, mark size=1.6pt{', ' + dash if dash else ''}] coordinates {{{mid}}};"), v[-1][0]
    t = S["text_b:text"]
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=9.5cm, height=6.8cm, xmin=14, xmax=58, xtick={16,24,32,40,48,56}, ymin=0.45, ymax=0.95, ytick={0.5,0.6,0.7,0.8,0.9},"
         r" yticklabel style={/pgf/number format/.cd, fixed, precision=1}, axis x line*=bottom, axis y line*=left, ymajorgrids,"
         r" xlabel={Block of the residual stream the probe reads}, ylabel={AUROC, weird vs.\ normal continuation}]",
         r"\draw[black!30, line width=0.5pt] (axis cs:14,0.5) -- (axis cs:58,0.5);"]
    for key, y0 in (("sharp", t["sharp"]), ("other", t["other"])):
        L.append(rf"\fill[black!8] (axis cs:14,{f(y0[1],3)}) rectangle (axis cs:58,{f(y0[2],3)});")
        L.append(rf"\draw[black, dashed, line width=0.9pt] (axis cs:14,{f(y0[0],3)}) -- (axis cs:58,{f(y0[0],3)});")
    b1, e1 = band("sharp", "dirc"); b2, e2 = band("other", "dirc!55", "densely dashed")
    L += [b1, b2,
          rf"\node[font=\footnotesize, anchor=west, align=left] at (axis cs:58.5,{f(t['sharp'][0],3)}) {{text judge,\\34 sharp forks}};",
          rf"\node[font=\footnotesize, anchor=west, align=left, dirc] at (axis cs:58.5,{f(e1,3)}) {{probe,\\34 sharp forks}};",
          rf"\node[font=\footnotesize, anchor=west, align=left] at (axis cs:58.5,{f(t['other'][0]+0.03,3)}) {{text judge, other\\466 cut points}};",
          rf"\node[font=\footnotesize, anchor=west, align=left, dirc!70!black] at (axis cs:58.5,{f(e2-0.03,3)}) {{probe, other\\466 cut points}};",
          r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_m_probe", "\n".join(L))


def rate_rows(groups, fname, width=10.5, xmax=50, note=None):
    """groups: [(header, clean, [(row label, color, mark, rate, lo, hi, ctrl_list or (min,max))])]. Behavior rate dot plot with controls."""
    rows, data = [], {}
    for g, clean, rr in groups:
        rows.append(("h", g))
        for i, r in enumerate(rr):
            key = f"{g}|{i}"; rows.append(("r", r[0], key)); data[key] = r + (clean,)
    ys, ticks, labels, ymin = rows_plot(rows)
    H = 0.5 * (len(ticks) + 2.5)
    L = [r"\begin{tikzpicture}",
         rf"\begin{{axis}}[width={width}cm, height={H:.2f}cm, ymin={ymin+0.4:.2f}, ymax=0.6, ytick={{{','.join(f(t,2) for t in ticks)}}},"
         rf" yticklabels={{{','.join('{' + l + '}' for l in labels)}}}, y tick label style={{font=\footnotesize}}, major y tick style={{draw=none}},"
         rf" axis y line*=left, axis x line*=bottom, xmajorgrids, xmin=0, xmax={xmax}, xticklabel={{$\pgfmathprintnumber{{\tick}}\%$}},"
         r" xlabel={Behavior rate with the edit}]"]
    # clean rate per group: a short dashed segment spanning the group's rows
    cur = None
    for g, clean, rr in groups:
        yy = [ys[f"{g}|{i}"] for i in range(len(rr))]
        top, bot = max(yy) + 1.0, min(yy) - 0.4
        L.append(rf"\draw[black!55, dashed, line width=0.8pt] (axis cs:{f(100*clean,2)},{f(bot,2)}) -- (axis cs:{f(100*clean,2)},{f(top-0.25,2)});")
        L.append(rf"\node[font=\scriptsize, black!60, anchor=west, inner sep=2pt] at (axis cs:{f(100*clean,2)},{f(top-0.1,2)}) {{no edit}};")
    for key, y in ys.items():
        lab, col, mk, m, lo, hi, ctrl, clean = data[key]
        if isinstance(ctrl, tuple):
            L.append(rf"\draw[ctrl!60, line width=3.2pt] (axis cs:{f(100*ctrl[0],2)},{f(y,2)}) -- (axis cs:{f(100*ctrl[1],2)},{f(y,2)});")
        elif ctrl:
            L.append(rf"\addplot[only marks, mark=o, mark size=1.5pt, ctrl] coordinates {{{' '.join(f'({f(100*c,2)},{f(y,2)})' for c in ctrl)}}};")
        L.append(rf"\addplot[only marks, mark={mk}, mark size=2.4pt, {col}, ci] coordinates {{({f(100*m,2)},{f(y,2)}) += ({f(100*(hi-m),2)},0) -= ({f(100*(m-lo),2)},0)}};")
    L.append(rf"\node[font=\scriptsize, anchor=south east, align=right, inner sep=1pt] at (axis cs:{xmax},0.75) "
             r"{\tikz\draw[ctrl!60, line width=3.2pt] (0,0) -- (0.4,0); range of matched random edits\quad"
             r" \tikz\draw[ctrl] (0,0) circle (1.5pt); one random edit\quad \tikz\draw[black!55, dashed, line width=0.8pt] (0,-0.1) -- (0,0.15); rate with no edit};")
    if note:
        L.append(note)
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    build(fname, "\n".join(L))


BEH4 = [("denying-ai-identity", "Denying being an AI"), ("false-physical-embodiment", "Claiming a body"),
        ("fabricated-code-execution", "Fake code execution"), ("laser-at-aircraft", "Laser at aircraft")]


def fig_m_axis():
    """Assistant Axis cap vs the neuron set, per behavior: the cap removes denying AI, the set the rest."""
    BB = by_beh(); E = evalc_data()
    groups = []
    for b, name in BEH4:
        n = BB[("C1a", b)]["n_prompts"]
        rr = []
        for arm, lab, col, mk in (("C1a", "top-100 neurons zeroed", "neur", "*"), ("C2", "Assistant Axis capped", "pers", "square*")):
            r = BB[(arm, b)]
            rr.append((lab, col, mk, float(r["arm4"]), float(r["arm4_lo"]), float(r["arm4_hi"]), (float(r["ctrl_min"]), float(r["ctrl_max"]))))
        groups.append((f"{name} ({n} prompts)", float(BB[("C1a", b)]["clean"]), rr))
    rr = [(lab, col, mk, E[a]["arm4"], E[a]["lo"], E[a]["hi"], E[a]["ctrl"]) for a, lab, col, mk in
          (("C1a", "top-100 neurons zeroed", "neur", "*"), ("C2", "Assistant Axis capped", "pers", "square*"))]
    groups.append(("All four (48 prompts)", E["C1a"]["clean"], rr))
    rate_rows(groups, "fig_m_axis", xmax=80)


def fig_m_jlens():
    """J-lens keyword direction: beats every random-word direction at a tiny damage, but removes a third."""
    BB = by_beh(); E = evalc_data()
    groups = []
    for b, name in BEH4:
        r = BB[("C6", b)]
        groups.append((f"{name} ({r['n_prompts']} prompts)", float(r["clean"]),
                       [("keyword direction capped", "read", "diamond*", float(r["arm4"]), float(r["arm4_lo"]), float(r["arm4_hi"]),
                         (float(r["ctrl_min"]), float(r["ctrl_max"])))]))
    groups.append(("All four (48 prompts)", E["C6"]["clean"],
                   [("keyword direction capped", "read", "diamond*", E["C6"]["arm4"], E["C6"]["lo"], E["C6"]["hi"], E["C6"]["ctrl"])]))
    rate_rows(groups, "fig_m_jlens", width=9.5, xmax=60)


def fig_m_olens():
    """O-lens: rises before the fork just as much when the words are read as a plain document."""
    F = C["fork"]; rel = F["rel"]
    def band(vals, colr, dash="", nm="x"):
        pts = [(r, v) for r, v in zip(rel, vals) if v and -20 <= r <= 10]
        up = " ".join(f"({r},{f(v[2],3)})" for r, v in pts); dn = " ".join(f"({r},{f(v[1],3)})" for r, v in pts)
        mid = " ".join(f"({r},{f(v[0],3)})" for r, v in pts)
        return (rf"\addplot[name path=u{nm}, draw=none] coordinates {{{up}}};\addplot[name path=d{nm}, draw=none] coordinates {{{dn}}};"
                rf"\addplot[{colr}, fill opacity=0.15] fill between[of=u{nm} and d{nm}];"
                rf"\addplot[{colr}, line width=1.3pt{', ' + dash if dash else ''}] coordinates {{{mid}}};"), pts
    b, a = F["P"]["before"], F["P"]["after"]
    tb, tp = band(F["text"], "black", "dashed", "t")
    lb, lp = band(F["lens"], "read", "", "l"); rb, rp = band(F["raw"], "read!50", "densely dashed", "r")
    L = [r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=1 by 2, vertical sep=0.5cm, x descriptions at=edge bottom},"
         r" width=10cm, xmin=-20, xmax=10, xtick={-20,-15,-10,-5,0,5,10}, axis x line*=bottom, axis y line*=left, ymajorgrids]",
         r"\nextgroupplot[height=4.2cm, ymin=0, ymax=1, ytick={0,0.5,1}, yticklabels={0\%,50\%,100\%}, ylabel={Chance of weird}]",
         rf"\addplot[meas, line width=1.6pt, const plot] coordinates {{(-20,{f(b[0],3)}) (0,{f(a[0],3)}) (10,{f(a[0],3)})}};",
         tb,
         r"\draw[black!50, dotted, line width=0.8pt] (axis cs:0,0) -- (axis cs:0,1);",
         rf"\node[font=\footnotesize, anchor=west] at (axis cs:10.3,{f(tp[-1][1][0],3)}) {{text judge}};",
         rf"\node[font=\footnotesize, meas!80!black, anchor=west] at (axis cs:10.3,{f(a[0]+0.08,3)}) {{measured}};",
         r"\nextgroupplot[height=4.8cm, ymin=0, ymax=8, ytick={0,2,4,6,8}, ylabel={O-lens behavior score},"
         r" xlabel={Tokens relative to the fork token (0 = the state that chooses it)}]",
         lb, rb,
         r"\draw[black!50, dotted, line width=0.8pt] (axis cs:0,0) -- (axis cs:0,8);",
         rf"\node[font=\footnotesize, read, anchor=west, align=left] at (axis cs:10.3,6.9) {{reply read\\in the chat}};",
         rf"\node[font=\footnotesize, read!60, anchor=west, align=left] at (axis cs:10.3,4.4) {{same words as\\a plain document}};",
         r"\end{groupplot}", r"\end{tikzpicture}"]
    build("fig_m_olens", "\n".join(L))


def fig_m_nla():
    """NLA: reconstructs chat states far worse than web text, and reads causal neurons as web-text genres."""
    import ast
    W = json.load(open(os.path.join(REPO, "runs/monitors/nla/val/fve_web.json")))
    V = json.load(open(os.path.join(REPO, "runs/monitors/nla/val/fve.json")))
    sets = json.load(open(os.path.join(REPO, "runs/evalc/prep/sets.json")))["C1a"]["fabricated-code-execution"]["sets"]["top10"]
    keys = [f"N|{l}|{n}|+1" for l, ns in sets.items() for n in ns]
    reads = {}
    for line in open(os.path.join(REPO, "runs/evalc/nla_c8/neurons_nla.jsonl")):
        r = json.loads(line)
        if r["key"] in keys:
            tx = ast.literal_eval(r["texts"]) if isinstance(r["texts"], str) else r["texts"]
            reads[r["key"]] = tx[0].split("\n")[0]
    def short(s, n=56):
        s = s.replace("Genre: ", "").split(";")[0].split(":")[0].split(",")[0].strip()
        s = s.split(" with ")[0].split(" — ")[0].split(" (")[0]
        if len(s) > n:
            s = s[:n].rsplit(" ", 1)[0] + "\u2026"
        s = s[0].upper() + s[1:]
        return esc(s.replace("#", "\\#").replace("$", "\\$")).replace("\u2026", r"\ldots")
    items = sorted(((int(k.split("|")[1]), short(v)) for k, v in reads.items()))
    Ls = [32, 44, 56]
    web = [W[f"web_L{l}_sample"]["FVE"] for l in Ls]; chat = [V[f"L{l}_sample"] for l in Ls]
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[name=a, width=5.6cm, height=6.6cm, xmin=-0.5, xmax=2.5, xtick={0,1,2}, xticklabels={32,44,56}, ymin=0, ymax=100,"
         r" ytick={0,25,50,75,100}, yticklabel={$\pgfmathprintnumber{\tick}\%$}, axis x line*=bottom, axis y line*=left, ymajorgrids,"
         r" xlabel={Block}, ylabel={Variance of the state reconstructed}, title={(a)}, title style={at={(-0.3,1.02)}, anchor=south west}]"]
    for i, (w, c) in enumerate(zip(web, chat)):
        L.append(rf"\draw[black!35, line width=1pt] (axis cs:{i},{f(100*w,2)}) -- (axis cs:{i},{f(100*c['FVE'],2)});")
    L.append(rf"\addplot[only marks, mark=*, mark size=2.6pt, black] coordinates {{{' '.join(f'({i},{f(100*w,2)})' for i, w in enumerate(web))}}};")
    cp = " ".join(f"({i},{f(100*c['FVE'],2)}) += (0,{f(100*(c['FVE_CI95'][1]-c['FVE']),2)}) -= (0,{f(100*(c['FVE']-c['FVE_CI95'][0]),2)})"
                  for i, c in enumerate(chat))
    L.append(rf"\addplot[only marks, mark=*, mark size=2.6pt, read, yci] coordinates {{{cp}}};")
    L += [rf"\node[font=\footnotesize, anchor=south, align=center] at (axis cs:1,{f(100*web[1]+4,2)}) {{web text\\(its training data)}};",
          rf"\node[font=\footnotesize, read, anchor=north, align=center] at (axis cs:1,{f(100*chat[1]['FVE']-7,2)}) {{our chat\\replies}};",
          r"\end{axis}"]
    lines = r"\\[1pt]".join(rf"{{\color{{black!55}}{l}}} & {t}" for l, t in items)
    L += [r"\node[anchor=north west, font=\footnotesize, align=left] at ($(a.north east)+(0.8cm,0.45cm)$) {"
          r"(b)\enspace The NLA's description of the output direction of each\\ of the ten neurons that most drive fake code execution\\[4pt]"
          r"\begin{tabular}{@{}r@{\enspace}l@{}}{\color{black!55}block} & {\color{black!55}NLA description (first phrase)}\\[2pt]" + lines + r"\end{tabular}};",
          r"\end{tikzpicture}"]
    build("fig_m_nla", "\n".join(L))


def fig_m_sae():
    """SAE: the signal at block 42 survives in the SAE's error term and is lost from its reconstruction."""
    S = sharp_ci()
    rows = [("r", "Raw block-42 state", "m6a:raw42_ridge"), ("r", "SAE error (state minus reconstruction)", "m6a:err_ridge"),
            ("r", "SAE reconstruction", "m6a:recon_ridge"), ("r", "Sparse probe, 256 SAE features", "m6a:k256"),
            ("r", "Sparse probe, 16 SAE features", "m6a:k16")]
    col = {"m6a:raw42_ridge": "dirc", "m6a:err_ridge": "read", "m6a:recon_ridge": "read", "m6a:k256": "read", "m6a:k16": "read"}
    ys, ticks, labels, ymin = rows_plot(rows)
    t = S["text_b:text"]
    L = [r"\begin{tikzpicture}",
         rf"\begin{{axis}}[width=9.6cm, height=6.0cm, ymin={ymin+0.4:.2f}, ymax=1.3, ytick={{{','.join(f(v,2) for v in ticks)}}},"
         rf" yticklabels={{{','.join('{' + l + '}' for l in labels)}}}, y tick label style={{font=\footnotesize}}, major y tick style={{draw=none}},"
         r" axis y line*=left, axis x line*=bottom, xmajorgrids, xmin=0.4, xmax=0.92, xtick={0.4,0.5,0.6,0.7,0.8,0.9},"
         r" xticklabel style={/pgf/number format/.cd, fixed, precision=1}, xlabel={AUROC, weird vs.\ normal continuation (ridge probe on each input)}]",
         rf"\draw[black!30, line width=0.5pt] (axis cs:0.5,{ymin+0.4:.2f}) -- (axis cs:0.5,1.3);"]
    for key, x, lab, anc in (("sharp", t["sharp"][0], r"text judge,\\sharp forks", "west"), ("other", t["other"][0], r"text judge,\\other cut points", "east")):
        L.append(rf"\draw[black, dashed, line width=0.8pt] (axis cs:{f(x,3)},{ymin+0.4:.2f}) -- (axis cs:{f(x,3)},1.3);")
        L.append(rf"\node[font=\scriptsize, anchor=north {anc}, align=left, inner sep=2pt] at (axis cs:{f(x,3)},1.3) {{{lab}}};")
    for _, lab, k in rows:
        y = ys[k]
        for key, mk, dy in (("sharp", "*", 0.13), ("other", "o", -0.13)):
            m, lo, hi = S[k][key][:3]
            L.append(rf"\addplot[only marks, mark={mk}, mark size=2.3pt, {col[k]}, line width=0.8pt, {'fill=white, ' if mk == 'o' else ''}ci] coordinates "
                     rf"{{({f(m,3)},{f(y+dy,2)}) += ({f(hi-m,3)},0) -= ({f(m-lo,3)},0)}};")
    L += [r"\node[font=\scriptsize, anchor=north west, align=left] at (axis cs:0.6,1.28) {\tikz\fill[read] (0,0) circle (2.3pt); 34 sharp forks\\ \tikz\draw[read, line width=0.8pt] (0,0) circle (2.3pt); 466 other cut points};",
          r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_m_sae", "\n".join(L))


def fig_steer():
    """Steering along the Assistant Axis at the forks: (a) dose-response per behavior, (b) the push with neuron sets zeroed."""
    P = json.load(open(os.path.join(REPO, "runs/persona_p1/analysis.json")))["c1"]["per_behavior"]
    X = json.load(open(os.path.join(REPO, "runs/persona_fwd/x1_analysis.json")))["groups"]
    doses = [("axis-0.5", -0.5), ("axis-0.25", -0.25), (None, 0.0), ("axis+0.1", 0.1), ("axis+0.25", 0.25), ("axis+0.5", 0.5)]
    hi = [("fabricated-code-execution", "Fake code execution", "*"), ("claims-called-911", "Called 911", "square*"),
          ("false-physical-embodiment", "Claiming a body", "triangle*"), ("denying-ai-identity", "Denying AI", "diamond*")]
    lo = [("laser-at-aircraft", "Laser"), ("recommends-drunk-driving", "Drunk driving"), ("chemtrails-assertion", "Chemtrails"),
          ("cutting-instructions", "Cutting")]
    val = lambda b, k: 0.0 if k is None else 100 * P[b][k]["dV"]
    import statistics as st
    band = []
    for k, x in doses:
        if k is None:
            band.append((x, 0.0, 0.0)); continue
        m = st.median(P[b][k]["rand_mean"] for b, *_ in hi + lo); s = st.median(P[b][k]["rand_sd"] for b, *_ in hi + lo)
        band.append((x, 100 * (m - 2 * s), 100 * (m + 2 * s)))
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[name=a, width=8.2cm, height=7cm, xmin=-0.55, xmax=0.55, xtick={-0.5,-0.25,0,0.25,0.5},"
         r" xticklabels={$-0.5$,$-0.25$,0,$+0.25$,$+0.5$}, ymin=-25, ymax=50, ytick={-20,-10,0,10,20,30,40,50},"
         r" yticklabel={$\pgfmathprintnumber{\tick}$}, axis x line*=bottom, axis y line*=left, ymajorgrids,"
         r" xlabel={Push along the Assistant Axis at block 32\\(times the median state norm)}, xlabel style={align=center},"
         r" ylabel={Change in chance of the weird token (points)}, title={(a)}, title style={at={(-0.25,1.02)}, anchor=south west}]",
         rf"\addplot[name path=bu, draw=none] coordinates {{{' '.join(f'({x},{f(h,2)})' for x, l_, h in band)}}};",
         rf"\addplot[name path=bd, draw=none] coordinates {{{' '.join(f'({x},{f(l_,2)})' for x, l_, h in band)}}};",
         r"\addplot[ctrl, fill opacity=0.25] fill between[of=bu and bd];",
         r"\draw[black!40, line width=0.5pt] (axis cs:-0.55,0) -- (axis cs:0.55,0);",
         r"\node[font=\scriptsize, black!60, anchor=south west] at (axis cs:-0.53,-24) {$\leftarrow$ away from the assistant end};",
         r"\node[font=\scriptsize, black!60, anchor=south east] at (axis cs:0.53,-24) {toward it $\rightarrow$};",
         r"\node[font=\scriptsize, ctrl!80!black, anchor=north, align=center] at (axis cs:-0.27,-7) {random directions,\\same size};"]
    vb = P["fabricated-code-execution"]["V_base_mean"]; vc = vb + P["fabricated-code-execution"]["axis-0.5"]["dV"]
    L.append(rf"\node[font=\scriptsize, pers!85!black, anchor=south west, align=left] at (axis cs:-0.48,{f(100*(vc-vb)+1,2)}) {{fake code execution: {100*vb:.0f}\% $\to$ {100*vc:.0f}\%}};")
    for b, lab in lo:
        L.append(rf"\addplot[ctrl, line width=0.7pt, mark=*, mark size=0.9pt] coordinates {{{' '.join(f'({x},{f(val(b,k),2)})' for k, x in doses)}}};")
    ends = []
    for b, lab, mk in hi:
        L.append(rf"\addplot[pers, line width=1.3pt, mark={mk}, mark size=2pt] coordinates {{{' '.join(f'({x},{f(val(b,k),2)})' for k, x in doses)}}};")
        ends.append([val(b, "axis+0.5"), lab])
    ends.sort(key=lambda e: -e[0])
    for i in range(1, len(ends)):
        if ends[i - 1][0] - ends[i][0] < 3.6:
            ends[i][0] = ends[i - 1][0] - 3.6
    for y, lab in ends:
        L.append(rf"\node[font=\scriptsize, pers!85!black, anchor=west] at (axis cs:0.53,{f(y,2)}) {{{lab}}};")
    L.append(r"\node[font=\scriptsize, ctrl!80!black, anchor=west, align=left] at (axis cs:0.53,6) {laser, drunk driving,\\chemtrails, cutting};")
    L.append(r"\end{axis}")
    # (b) the same push with each behavior's top-100 neurons zeroed (forks the sets were not chosen on)
    rows = [("911", "Called 911"), ("code", "Fake code execution"), ("dai", "Denying AI"), ("embodiment", "Claiming a body"), ("laser", "Laser")]
    L.append(r"\begin{axis}[at={($(a.east)+(6.6cm,0)$)}, anchor=west, width=7cm, height=6.4cm, xmin=-10, xmax=30, xtick={-10,0,10,20,30},"
             r" ymin=-4.6, ymax=0.9, ytick={0,-1,-2,-3,-4}, yticklabels={" +
             ",".join("{" + n + rf" ({X[k + '|outsample']['V']['n']})" + "}" for k, n in rows) + r"},"
             r" y tick label style={font=\footnotesize}, major y tick style={draw=none}, axis x line*=bottom, axis y line*=left, xmajorgrids,"
             r" xlabel={Change in chance of the weird token\\from a push of $-0.25$ (points)}, xlabel style={align=center}, title={(b)}, title style={at={(-0.62,1.12)}, anchor=south west}]")
    L.append(r"\draw[black!40, line width=0.5pt] (axis cs:0,-4.6) -- (axis cs:0,0.9);")
    for i, (k, n) in enumerate(rows):
        V = X[k + "|outsample"]["V"]; y = -i
        s0, s1, sr = 100 * V["S0_steer_clean"], 100 * V["S1_steer_under_top"], 100 * V["S_steer_under_rand100"]
        c0, c1 = [100 * v for v in V["ci"]["S0"]], [100 * v for v in V["ci"]["S1"]]
        L.append(rf"\addplot[only marks, mark=o, mark size=2.4pt, ctrl, line width=0.8pt] coordinates {{({f(sr,2)},{y+0.22})}};")
        L.append(rf"\addplot[only marks, mark=*, mark size=2.4pt, pers, ci] coordinates {{({f(s0,2)},{y+0.22}) += ({f(c0[1]-s0,2)},0) -= ({f(s0-c0[0],2)},0)}};")
        L.append(rf"\addplot[only marks, mark=*, mark size=2.4pt, neur, ci] coordinates {{({f(s1,2)},{y-0.18}) += ({f(c1[1]-s1,2)},0) -= ({f(s1-c1[0],2)},0)}};")
    L.append(r"\node[font=\scriptsize, anchor=south west, align=left] at (axis cs:-9.5,0.55) {\tikz\fill[pers] (0,0) circle (2.4pt); push alone\quad"
             r" \tikz\fill[neur] (0,0) circle (2.4pt); with the top-100 neurons zeroed\\ \tikz\draw[ctrl, line width=0.8pt] (0,0) circle (2.4pt); with 100 random neurons zeroed};")
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_steer", "\n".join(L))


def fig_axis_shift():
    """Zeroing the top-100 neurons removes the weird token with almost no movement along the Assistant Axis."""
    P = json.load(open(os.path.join(REPO, "runs/persona_p1/analysis.json")))["c1"]["per_behavior"]
    beh = [("fabricated-code-execution", "Fake code execution"), ("false-physical-embodiment", "Claiming a body"),
           ("denying-ai-identity", "Denying AI"), ("laser-at-aircraft", "Laser at aircraft"), ("claims-called-911", "Called 911")]
    L = [r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=5 by 1, horizontal sep=0.35cm, y descriptions at=edge left},"
         r" width=4.2cm, height=5.2cm, xmin=-0.1, xmax=1.65, xtick={0,0.5,1,1.5}, xticklabels={0,.5,1,1.5}, ymin=-3, ymax=22,"
         r" ytick={0,5,10,15,20}, axis x line*=bottom, axis y line*=left, grid=major, title style={font=\footnotesize, align=center, yshift=-2pt}]"]
    for i, (b, name) in enumerate(beh):
        v = P[b]; d = v["d"]
        sh = lambda k: (d["48"][k] + d["56"][k]) / 2
        steer = [(0, 0)] + [(sh(f"steer_axis+{x}_shift_gap"), -100 * v[f"axis+{x}"]["dV"]) for x in ("0.1", "0.25", "0.5")]
        L.append(rf"\nextgroupplot[title={{{name}\\[-1pt]{{\scriptsize\color{{black!55}}{v['n']} forks}}}}"
                 + (r", ylabel={Drop in chance of the weird token (points)}" if i == 0 else "") + "]")
        L.append(r"\draw[black!35, line width=0.5pt] (axis cs:-0.1,0) -- (axis cs:1.65,0);")
        L.append(rf"\addplot[pers, line width=1.1pt, mark=o, mark size=1.6pt] coordinates {{{' '.join(f'({f(x,3)},{f(y,2)})' for x, y in steer)}}};")
        L.append(rf"\addplot[only marks, mark=square*, mark size=2.2pt, pers] coordinates {{({f(sh('cap_shift_gap'),3)},{f(-100*v['e:cap_axis']['dV'],2)})}};")
        L.append(rf"\addplot[only marks, mark=*, mark size=1.8pt, ctrl] coordinates {{({f(sh('rand_shift_gap'),3)},{f(-100*v['top100']['dV_rand100'],2)})}};")
        L.append(rf"\addplot[only marks, mark=*, mark size=2.8pt, neur] coordinates {{({f(sh('top_shift_gap'),3)},{f(-100*v['top100']['dV_top'],2)})}};")
    L += [r"\end{groupplot}",
          r"\node[font=\small, anchor=north] at ($(group c3r1.south)+(0,-0.6cm)$) {Shift of the fork state toward the assistant end (fraction of the assistant-to-role distance, blocks 48 to 56)};",
          r"\node[font=\footnotesize, anchor=south, align=center] at ($(group c3r1.north)+(0,0.85cm)$) {\tikz\fill[neur] (0,0) circle (2.8pt); top-100 neurons zeroed\qquad"
          r" \tikz\fill[ctrl] (0,0) circle (1.8pt); 100 random neurons zeroed\qquad \tikz\fill[pers] (-2.2pt,-2.2pt) rectangle (2.2pt,2.2pt); Assistant Axis capped\\"
          r" \tikz{\draw[pers, line width=1.1pt] (-0.25,0) -- (0.25,0); \draw[pers] (0,0) circle (1.6pt);} pushed toward the assistant end ($+0.1$, $+0.25$, $+0.5$)};",
          r"\end{tikzpicture}"]
    build("fig_axis_shift", "\n".join(L))


FIGS = {"fig_evalc": lambda: fig_evalc(evalc_data()), "fig_steer": fig_steer, "fig_axis_shift": fig_axis_shift, "fig_monitors": fig_monitors, "fig_fork": fig_fork, "fig_dir_own": fig_dir_own,
        "fig_m_neurons": fig_m_neurons, "fig_m_direction": fig_m_direction, "fig_m_probe": fig_m_probe, "fig_m_axis": fig_m_axis,
        "fig_m_jlens": fig_m_jlens, "fig_m_olens": fig_m_olens, "fig_m_nla": fig_m_nla, "fig_m_sae": fig_m_sae,
        "fig_dir_transfer": fig_dir_transfer, "fig_dir_prefix": fig_dir_prefix, "fig_axis": fig_axis, "fig_lens": fig_lens_measure_pick,
        "fig_shared": fig_shared, "fig_chive": fig_chive}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for name, fn in FIGS.items():
        if not a.only or name in a.only.split(","):
            fn()
