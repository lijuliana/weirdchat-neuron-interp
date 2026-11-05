"""Exploratory figures about the attribution neuron sets that are not about eval C or the monitors.

  fig_x_neur_words   what the strongest top-10 neurons write: the fork-token z-score of each neuron's output direction
                     (data/agastya/results_neuron_logits.txt: output column -> final norm -> unembedding)
  fig_x_neur_overlap how many neurons the behaviors' top-1,000 sets share, against a layer-matched chance
                     (runs/evalc/prep/sets.json)
  fig_x_neur_push    pushing one neuron and asking the model who it is (runs/neuron_push: quick_identity.json, analysis.json,
                     gallery.md)

Builds through scripts/62_figures.py (shared style). Run: .venv/bin/python -I scripts/64_explore_neurons.py
"""
import ast, collections, importlib.util, json, os, re

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("figs62", os.path.join(REPO, "scripts/62_figures.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
build, f, esc = F.build, F.f, F.esc
OUT = {}
SH = {"claims-called-911": "Called 911", "denying-ai-identity": "Denying AI", "fabricated-code-execution": "Fake code execution",
      "false-physical-embodiment": "Claiming a body", "laser-at-aircraft": "Laser at aircraft"}
NPL = 17408  # MLP neurons per layer


def parse_logits(path):
    out, b, lines, i = {}, None, open(path).read().split("\n"), 0
    while i < len(lines):
        l = lines[i]
        m = re.match(r"== (\S+)\s+\(fork tokens: (.*)\)", l)
        if m:
            b = m.group(1); out[b] = {"fork": ast.literal_eval(m.group(2)), "neurons": []}; i += 1; continue
        m = re.match(r"\s+L(\d+) n\s*(\d+)\s+A_last mean ([+-][\d.]+)\s+fork-token z (.*)", l)
        if m:
            zs = dict((ast.literal_eval(t), float(z)) for t, z in re.findall(r"('(?:[^'\\]|\\.)*'|\"[^\"]*\"):([+-][\d.]+)", m.group(4)))
            out[b]["neurons"].append({"L": int(m.group(1)), "n": int(m.group(2)), "z": zs,
                                      "plus": ast.literal_eval(lines[i + 1].strip()[2:]), "minus": ast.literal_eval(lines[i + 2].strip()[2:])})
            i += 3; continue
        i += 1
    return out


DICT = None
# subword fragments and foreign words that pass the dictionary check, found by reading the lists in results_neuron_logits.txt
JUNK = {"park", "ulling", "pers", "live", "link", "ling", "ieres", "ILO", "IRES", "IPO", "IMO", "saya", "Legg"}


def real(w):
    global DICT
    if DICT is None:
        DICT = {x.strip().lower() for x in open("/usr/share/dict/words")}
    lw = w.lower()
    return lw in DICT or (lw.endswith("s") and lw[:-1] in DICT) or (w.isupper() and len(w) >= 2)


def latin(toks, k=3):
    seen, out = set(), []
    for t in toks:
        w = t.strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z']{1,}", w) and w.lower() not in seen:
            seen.add(w.lower()); out.append(w)
    return out[:k]


def words(toks, k=3):
    """English words in a token list, or [] when fewer than 3 of the tokens are English words."""
    kept = [t.strip() for t in toks if re.fullmatch(r"[A-Za-z][A-Za-z']{1,}", t.strip()) and real(t.strip()) and t.strip() not in JUNK]
    if len(kept) < 3:
        return []
    seen, out = set(), []
    for w in kept:
        if w.lower() not in seen:
            seen.add(w.lower()); out.append(w.lower() if not w.isupper() else w)
    return out[:k]


def tex_tok(t):
    return r"\texttt{" + esc(t.replace(" ", "\\textvisiblespace{}") if False else t).replace("'", "'") + "}"


# ------------------------------------------------------------------------------------------------ words
def fig_words():
    D = parse_logits(os.path.join(REPO, "data/agastya/results_neuron_logits.txt"))
    rows, allz = [], []
    for b in SH:
        for n in D[b]["neurons"]:
            t, z = max(n["z"].items(), key=lambda kv: abs(kv[1]))
            allz.append((b, n["L"], n["n"], t, z))
            if abs(z) >= 5:
                rows.append((b, n, t, z))
    strong = len(rows); total = len(allz)
    uniq = {(L, n) for _, L, n, _, _ in allz}
    lay_strong = sorted(n["L"] for _, n, _, _ in rows)
    lay_weak = sorted(L for _, L, n, t, z in allz if abs(z) < 5)
    OUT["words"] = {"strong": strong, "total": total, "unique": len(uniq), "strong_layers": [lay_strong[0], lay_strong[-1]],
                    "weak_layers": [lay_weak[0], lay_weak[-1]], "weak_late": sum(L >= 49 for L in lay_weak),
                    "writers": sum(z > 0 for *_, z in rows), "suppressors": sum(z < 0 for *_, z in rows)}
    # layout: grouped rows
    y, ticks, labels, items = 0.0, [], [], []
    order = sorted(SH, key=lambda b: -sum(1 for r in rows if r[0] == b))
    for b in order:
        rr = sorted([r for r in rows if r[0] == b], key=lambda r: -r[3])
        if not rr:
            continue
        if ticks:
            y -= 0.4
        ticks.append(y); labels.append(r"\textbf{" + SH[b] + "}"); y -= 1
        for _, n, t, z in rr:
            ticks.append(y); labels.append(f"L{n['L']} n{n['n']}")
            up, dn = words(n["plus"]), words(n["minus"])
            ft = t.strip().lower()
            if z > 0:
                other = ("also up", [w for w in up if w.lower() != ft]) if [w for w in up if w.lower() != ft] else ("", [])
            else:
                other = ("up instead", up) if up else ("also down", [w for w in dn if w.lower() != ft]) if [w for w in dn if w.lower() != ft] else ("", [])
            items.append((y, t, z, other, n)); y -= 1
    H = 0.44 * (len(ticks) + 3)
    L = [r"\begin{tikzpicture}",
         rf"\begin{{axis}}[width=7.2cm, height={H:.2f}cm, xmin=-50, xmax=40, xtick={{-40,-20,0,20,40}}, ymin={y+0.4:.2f}, ymax=0.6,"
         rf" ytick={{{','.join(f(v,2) for v in ticks)}}}, yticklabels={{{','.join('{' + l + '}' for l in labels)}}},"
         r" y tick label style={font=\footnotesize}, major y tick style={draw=none}, axis x line*=bottom, axis y line*=left, xmajorgrids,"
         r" xlabel={z-score of the fork word among all tokens}, xlabel style={align=center}]",
         rf"\fill[black!6] (axis cs:-5,{y+0.4:.2f}) rectangle (axis cs:5,0.6);",
         rf"\draw[black!45, line width=0.6pt] (axis cs:0,{y+0.4:.2f}) -- (axis cs:0,0.6);"]
    for yy, t, z, other, n in items:
        col = "neur" if z > 0 else "neur!40"
        L.append(rf"\fill[{col}] (axis cs:0,{f(yy-0.32,2)}) rectangle (axis cs:{f(z,2)},{f(yy+0.32,2)});")
        tok = r"\texttt{" + esc(t.strip()) + "}"
        L.append(rf"\node[font=\scriptsize, anchor={'west' if z > 0 else 'east'}, inner sep=2pt] at (axis cs:{f(z,2)},{f(yy,2)}) {{{tok}}};")
        txt = (r"{\color{black!55}" + other[0] + ":} " + ", ".join(r"\texttt{" + esc(o) + "}" for o in other[1])) if other[1] else ""
        L.append(rf"\node[font=\scriptsize, anchor=west, inner sep=0pt] at (axis cs:49,{f(yy,2)}) {{{txt}}};")
    L += [r"\node[font=\scriptsize, black!55, anchor=south] at (axis cs:-25,0.6) {pushes the fork word down};",
          r"\node[font=\scriptsize, black!55, anchor=south] at (axis cs:20,0.6) {pushes it up};",
          r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_x_neur_words", "\n".join(L))
    OUT["words_rows"] = [(SH[b], n["L"], n["n"], t, z, words(n["plus"]), words(n["minus"])) for b, n, t, z in rows]


# ------------------------------------------------------------------------------------------------ overlap
def fig_overlap():
    s = json.load(open(os.path.join(REPO, "runs/evalc/prep/sets.json")))["C1a"]
    B = list(SH)
    S = {(b, k): {(int(L), n) for L, ns in s[b]["sets"][k].items() for n in ns} for b in B for k in ("top100", "top1000")}
    def exp(a, b, k):
        ca = collections.Counter(L for L, _ in S[(a, k)]); cb = collections.Counter(L for L, _ in S[(b, k)])
        return sum(ca[L] * cb[L] / NPL for L in ca)
    cells = []
    for i, a in enumerate(B):
        for j, b in enumerate(B):
            if i == j:
                continue
            k = "top1000" if i > j else "top100"
            o, e = len(S[(a, k)] & S[(b, k)]), exp(a, b, k)
            cells.append((j, -i, k, o, e))
    t1 = [(o, e) for _, _, k, o, e in cells if k == "top1000"]
    OUT["overlap"] = {"top1000_range": [min(o for o, _ in t1), max(o for o, _ in t1)], "exp_range": [round(min(e for _, e in t1), 2), round(max(e for _, e in t1), 2)],
                      "dai_body": [len(S[("denying-ai-identity", "top1000")] & S[("false-physical-embodiment", "top1000")]),
                                   round(exp("denying-ai-identity", "false-physical-embodiment", "top1000"), 2)]}
    multi = collections.Counter(x for b in B for x in S[(b, "top100")])
    OUT["overlap"]["top100_shared"] = [(L, n, [SH[b] for b in B if (L, n) in S[(b, "top100")]]) for (L, n), v in multi.items() if v >= 2]
    stops = [(0, (247, 247, 247)), (10, (253, 208, 162)), (25, (241, 105, 19)), (40, (166, 54, 3))]
    def rgb(v):
        v = max(0, min(40, v))
        for (a, ca), (b_, cb) in zip(stops, stops[1:]):
            if v <= b_:
                t = (v - a) / (b_ - a); return tuple(int(round(ca[q] + t * (cb[q] - ca[q]))) for q in range(3))
        return stops[-1][1]
    n = len(B); short = [SH[b] for b in B]
    L = [r"\begin{tikzpicture}",
         rf"\begin{{axis}}[width=8.4cm, height=8.4cm, xmin=-0.5, xmax={n-0.5}, ymin={-(n-0.5)}, ymax=0.5, enlargelimits=false,"
         rf" xtick={{{','.join(str(j) for j in range(n))}}}, xticklabels={{{','.join(short)}}}, x tick label style={{rotate=35, anchor=north east, font=\footnotesize}},"
         rf" ytick={{{','.join(str(-i) for i in range(n))}}}, yticklabels={{{','.join(short)}}}, y tick label style={{font=\footnotesize}},"
         r" tick style={draw=none}, axis line style={draw=none}]"]
    for j, y, k, o, e in cells:
        ratio = o / e if e > 0 else 0
        r_, g_, b_ = rgb(ratio if k == "top1000" else 0) if k == "top1000" else (255, 255, 255)
        L.append(rf"\fill[fill={{rgb,255:red,{r_};green,{g_};blue,{b_}}}] (axis cs:{j-0.5},{y-0.5}) rectangle (axis cs:{j+0.5},{y+0.5});")
        if k == "top1000":
            L.append(rf"\node[font=\footnotesize, text={'white' if ratio > 22 else 'black'}, align=center] at (axis cs:{j},{y}) {{{o}\\[-2pt]{{\scriptsize {ratio:.0f}$\times$}}}};")
        else:
            L.append(rf"\node[font=\footnotesize, text=black!70] at (axis cs:{j},{y}) {{{o}}};")
    for i in range(n):
        L.append(rf"\fill[black!12] (axis cs:{i-0.5},{-i-0.5}) rectangle (axis cs:{i+0.5},{-i+0.5});")
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_x_neur_overlap", "\n".join(L))


# ------------------------------------------------------------------------------------------------ push atlas
def fig_push():
    Q = json.load(open(os.path.join(REPO, "runs/neuron_push/quick_identity.json")))["by"]
    A = json.load(open(os.path.join(REPO, "runs/neuron_push/analysis.json")))["sets"]
    al = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    kinds = [("set", "attribution-set neurons", "neur", "*", ""), ("rand", "random neurons, same layers", "ctrl", "o", ""),
             ("dir", "random directions", "black", "square*", "densely dashed")]
    OUT["push"] = {k: {"lost": [round(Q[f"{k}|{a}"]["lost"][0], 3) for a in al], "persona": [round(Q[f"{k}|{a}"]["persona"][0], 3) for a in al],
                       "incoherent": [round(Q[f"{k}|{a}"]["incoherent"][0], 3) for a in al]} for k, *_ in kinds}
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[name=a, width=6.6cm, height=6.2cm, xmin=0.15, xmax=1.6, xtick={0.25,0.5,0.75,1,1.25,1.5}, xticklabels={.25,.5,.75,1,1.25,1.5},"
         r" ymin=0, ymax=100, ytick={0,25,50,75,100}, yticklabel={$\pgfmathprintnumber{\tick}\%$}, axis x line*=bottom, axis y line*=left, ymajorgrids,"
         r" xlabel={Push size (times the layer's median state norm)}, ylabel={Replies to ``Who are you?''}, title={(a)}, title style={at={(-0.28,1.13)}, anchor=south west}]"]
    for k, lab, col, mk, dash in kinds:
        pts = " ".join(f"({a},{f(100*Q[f'{k}|{a}']['lost'][0],2)})" for a in al)
        L.append(rf"\addplot[{col}, line width=1.2pt, mark={mk}, mark size=1.8pt{', ' + dash if dash else ''}] coordinates {{{pts}}};")
        pts = " ".join(f"({a},{f(100*Q[f'{k}|{a}']['persona'][0],2)})" for a in al)
        L.append(rf"\addplot[{col}, line width=0.8pt, mark={mk}, mark size=1.3pt, densely dotted] coordinates {{{pts}}};")
    L += [r"\node[font=\scriptsize, anchor=north west, align=left] at (axis cs:0.2,98) {solid: no longer says it is an AI\\dotted: claims a human persona};",
          r"\node[font=\scriptsize, anchor=south, align=center] at (axis cs:0.875,101) {\tikz\draw[neur, line width=1.2pt] (0,0) -- (0.35,0); set neurons\enspace"
          r" \tikz\draw[ctrl, line width=1.2pt] (0,0) -- (0.35,0); random neurons\\ \tikz\draw[black, line width=1.2pt, densely dashed] (0,0) -- (0.35,0); random directions};",
          r"\end{axis}"]
    # (b) on-theme persona replies per set
    sets = [("A|claims-called-911", "Called 911", "emergency"), ("A|denying-ai-identity", "Denying AI", "a human"),
            ("A|false-physical-embodiment", "Claiming a body", "a body"), ("A|fabricated-code-execution", "Fake code execution", "coding"),
            ("A|laser-at-aircraft", "Laser at aircraft", "mischief")]
    OUT["push"]["ontheme"] = {lab: [A[k]["ontheme_either"][x] for x in ("set", "ctrl", "C", "lo", "hi", "p_holm")] for k, lab, _ in sets}
    L.append(r"\begin{axis}[at={($(a.east)+(3.4cm,0)$)}, anchor=west, width=5.4cm, height=6.2cm, xmin=0, xmax=15, xtick={0,5,10,15},"
             r" xticklabel={$\pgfmathprintnumber{\tick}\%$}, ymin=-4.6, ymax=0.6, ytick={0,-1,-2,-3,-4}, yticklabels={"
             + ",".join("{" + lab + rf"\\[-2pt]{{\scriptsize\color{{black!55}}theme: {th}}}" + "}" for _, lab, th in sets) + r"},"
             r" y tick label style={font=\footnotesize, align=right}, major y tick style={draw=none}, axis x line*=bottom, axis y line*=left, xmajorgrids,"
             r" xlabel={Replies on the set's own theme}, title={(b)}, title style={at={(-0.75,1.13)}, anchor=south west}]")
    for i, (k, lab, _) in enumerate(sets):
        o = A[k]["ontheme_either"]; y = -i
        L.append(rf"\addplot[only marks, mark=square*, mark size=1.8pt, black] coordinates {{({f(100*o['C'],2)},{y-0.22})}};")
        L.append(rf"\addplot[only marks, mark=o, mark size=2.2pt, ctrl, line width=0.8pt] coordinates {{({f(100*o['ctrl'],2)},{y})}};")
        L.append(rf"\addplot[only marks, mark=*, mark size=2.4pt, neur] coordinates {{({f(100*o['set'],2)},{y+0.22})}};")
    L.append(r"\node[font=\scriptsize, anchor=south east, align=right] at (axis cs:15,0.62) {\tikz\fill[neur] (0,0) circle (2.4pt); set neurons\enspace"
             r" \tikz\draw[ctrl, line width=0.8pt] (0,0) circle (2.2pt); random neurons\enspace \tikz\fill[black] (-1.8pt,-1.8pt) rectangle (1.8pt,1.8pt); random directions};")
    L.append(r"\end{axis}")
    # (c) quotes, allowed behaviors only, verbatim from runs/neuron_push/gallery.md
    gal = open(os.path.join(REPO, "runs/neuron_push/gallery.md")).read()
    want = [("A 911", "What are you doing right now?", "Called 911"), ("A 911", "Who are you?", "Called 911"),
            ("A code execution", "What are you doing right now?", "Fake code execution"), ("A laser", "What are you doing right now?", "Laser at aircraft"),
            ("A embodiment", "Who are you?", "Claiming a body"), ("A denying AI", "Who are you?", "Denying AI")]
    quotes = []
    for line in gal.split("\n"):
        m = re.match(r"\d+\. \*\*(A [^,]+), (L\d+N\d+)\*\*, (\w+), alpha ([\d.]+), \"([^\"]+)\": \"(.*)\"$", line)
        if not m:
            continue
        for w, pr, lab in want:
            if m.group(1) == w and m.group(5) == pr and lab not in [q[0] for q in quotes if q[0] == lab and lab != "Called 911"]:
                q = m.group(6).replace("**", "")
                q = q if len(q) < 150 else q[:150].rsplit(" ", 1)[0] + "\u2026"
                quotes.append((lab, pr, q))
                break
    seen, qs = collections.Counter(), []
    for lab, pr, q in quotes:
        if seen[lab] < (2 if lab == "Called 911" else 1):
            seen[lab] += 1; qs.append((lab, pr, q))
    OUT["push"]["quotes"] = qs
    lines = r"\\[3pt]".join(rf"{{\color{{neur}}{lab} neuron}}, ``{esc(pr)}''\\\enspace\textit{{``{esc(q).replace(chr(8230), r'\ldots')}''}}" for lab, pr, q in qs)
    L.append(r"\node[anchor=north west, font=\scriptsize, align=left, text width=13.6cm] at ($(a.south west)+(-1.3cm,-1.7cm)$) {"
             r"{\small (c)}\enspace Replies with one attribution-set neuron pushed (one layer, greedy decoding)\\[3pt]" + lines + "};")
    L.append(r"\end{tikzpicture}")
    build("fig_x_neur_push", "\n".join(L))


if __name__ == "__main__":
    fig_words(); fig_overlap(); fig_push()
    os.makedirs(os.path.join(REPO, "findings/explore"), exist_ok=True)
    json.dump(OUT, open(os.path.join(REPO, "findings/explore/neurons_numbers.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in OUT.items() if k != "words_rows"}, indent=1, default=str)[:3000])
    for r in OUT["words_rows"]:
        print(r)
