"""Exploratory SAE figures (block-42 SAE, ceselder/qwen36-27b-sae-l42-1b, 131,072 features), beyond monitor AUROC.

  fig_x_sae_split    which labelled features differ between the weird and the normal continuation of the same prefix,
                     4 tokens in (eval A set b), per behavior; sign-flip p-values over cuts
  fig_x_sae_chatweb  firing frequency of every feature on chat replies vs web text; chat-only features and what they
                     fire on (top-activating chat contexts)
  fig_x_sae_dir      the SAE features most aligned with each behavior's weird-vs-normal direction (m6d.json), marked
                     when the feature is chat-specific
  fig_x_sae_labels   how trustworthy the auto-labels are: held-out detection / fuzzing accuracy vs a shuffled-label
                     control, and the share of labels that restate visible text

Inputs: runs/monitors/sae/{codes.npz, labels.json, labels_m6d.json, chat_latent_stats.npz, chat_index.npz, m6d.json,
detfuzz.json, restate.json}, runs/monitors/capture/index.jsonl, data/tokenizer/tokenizer.json.
Writes figures/fig_x_sae_*.{pdf,svg} (via scripts/62_figures.build) and runs/monitors/sae/explore_sae.json.
Run: .venv/bin/python -I scripts/64_explore_sae.py
"""
import collections, importlib.util, json, os, sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("figs62", os.path.join(REPO, "scripts/62_figures.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
build, f, esc = F.build, F.f, F.esc
S = os.path.join(REPO, "runs/monitors/sae")
N = 131072
rng = np.random.default_rng(0)

LAB = json.load(open(os.path.join(S, "labels.json")))
LAB.update({k: v for k, v in json.load(open(os.path.join(S, "labels_m6d.json"))).items() if k not in LAB})
ST = np.load(os.path.join(S, "chat_latent_stats.npz"))
CHAT = ST["frac"]; WEB = ST["web_cnt"] / float(ST["web_n"])
MX = np.maximum(ST["max"], ST["web_max"]); MX = np.where(MX > 0, MX, 1.0)
OUT = {}


def tex_label(s, n=72):
    s = (s or "").replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace('"', "''")
    if len(s) > n:
        s = s[:n].rsplit(" ", 1)[0] + "…"
    s = s[0].upper() + s[1:] if s else s
    s = esc(s).replace("#", "\\#").replace("$", "\\$").replace("~", "\\textasciitilde{}").replace("^", "\\^{}")
    return s.replace("…", r"\ldots").replace("[", "{[}").replace("]", "{]}")


# ================================================================== (1) weird vs normal continuation, per behavior
def split_contrast():
    z = np.load(os.path.join(S, "codes.npz")); ptr, lat, act = z["ptr"], z["lat"], z["act"]
    R = [json.loads(l) for l in open(os.path.join(REPO, "runs/monitors/capture/index.jsonl"))]
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: {True: [], False: []}))
    for r in R:
        if r["set"] == "b" and r.get("k") == 4:
            by[r["b"]][r["cut"]][r["match"]].append(r["row"])
    res = {}
    for beh, cuts in by.items():
        rows = []
        for cut, g in cuts.items():
            if not g[True] or not g[False]:
                continue
            d = np.zeros(N)
            for side, sg in ((True, 1.0), (False, -1.0)):
                for row in g[side]:
                    a, b = ptr[row], ptr[row + 1]
                    np.add.at(d, lat[a:b], sg * act[a:b] / MX[lat[a:b]] / len(g[side]))
            rows.append(d)
        A = np.array(rows)  # cuts x features
        res[beh] = A
    return res


def pick_rows(A, k=3, fw=None):
    m = A.mean(0)
    lab = np.array([str(i) in LAB for i in range(N)])
    out = []
    for sign in (1, -1):
        o = np.argsort(-sign * m)
        picked = [i for i in o[:400] if lab[i]][:k]
        rank_unl = [int(i) for i in o[:5] if not lab[i]]
        for i in picked:
            col = A[:, i]
            obs = col.mean()
            fl = rng.choice([-1.0, 1.0], size=(20000, len(col)))
            p = float(((np.abs(fl @ col / len(col))) >= abs(obs) - 1e-12).mean())
            bo = [col[rng.integers(0, len(col), len(col))].mean() for _ in range(2000)]
            out.append({"lat": int(i), "diff": float(obs), "lo": float(np.percentile(bo, 2.5)), "hi": float(np.percentile(bo, 97.5)),
                        "p": p, "rank": int(np.where(o == i)[0][0]) + 1, "label": LAB[str(i)], "side": "weird" if sign > 0 else "normal",
                        "chat_frac": float(CHAT[i]), "web_frac": float(WEB[i]), "fw": bool(fw is not None and abs(obs) > fw)})
        out[-1]["unlabelled_above"] = rank_unl
    return out


BEH6 = [("denying-ai-identity", "Denying being an AI"), ("false-physical-embodiment", "Claiming a body"),
        ("claims-called-911", "Called 911"), ("fabricated-code-execution", "Fake code execution"),
        ("laser-at-aircraft", "Laser at aircraft")]
LABELLED = np.array(sorted(int(k) for k in LAB if int(k) < N))


def fw_threshold(A, n=1000):
    """95th percentile of the largest |mean| over the labelled features when each cut's sign is flipped at random."""
    B = A[:, LABELLED]; r = np.random.default_rng(1); mx = []
    for _ in range(n):
        sg = r.choice([-1.0, 1.0], size=B.shape[0]); mx.append(np.abs(sg @ B / B.shape[0]).max())
    return float(np.percentile(mx, 95))


def fig_split(res):
    rows, data = [], {}
    rec = {}
    for b, name in BEH6:
        A = res[b]
        fw = fw_threshold(A)
        pr = pick_rows(A, fw=fw)
        rec[b] = {"n_cuts": int(A.shape[0]), "fw95": fw, "n_labelled": int(len(LABELLED)), "rows": pr}
        rows.append(("h", f"{name} ({A.shape[0]} cut points)"))
        for j, r in enumerate(pr):
            key = f"{b}|{j}"; rows.append(("r", "@@" + key, key)); data[key] = r
    OUT["split"] = rec
    ys, ticks, labels, ymin = F.rows_plot(rows)
    labels = [(tex_label(data[l[2:]]["label"]) if l.startswith("@@") else l) for l in
              [r[1] if r[0] == "h" else "@@" + r[2] for r in rows]]
    labels = [(F.esc(l) if False else l) for l in labels]
    # rows_plot escaped the header labels already; rebuild header formatting
    hdr = iter([r[1] for r in rows if r[0] == "h"])
    labels = [(r"\textbf{" + F.esc(next(hdr)) + "}") if r[0] == "h" else tex_label(data[r[2]]["label"]) for r in rows]
    H = 0.42 * (len(ticks) + 2.5)
    L = [r"\begin{tikzpicture}",
         rf"\begin{{axis}}[width=7.2cm, height={H:.2f}cm, ymin={ymin+0.4:.2f}, ymax=0.6, ytick={{{','.join(f(t,2) for t in ticks)}}},"
         rf" yticklabels={{{','.join('{' + l + '}' for l in labels)}}}, y tick label style={{font=\scriptsize}}, major y tick style={{draw=none}},"
         r" axis y line*=left, axis x line*=bottom, xmajorgrids, xmin=-16, xmax=14, xtick={-15,-10,-5,0,5,10},"
         r" xlabel={Activation in the weird minus the normal continuation\\(\% of the feature's maximum, mean over cut points)}, xlabel style={align=center}]",
         rf"\draw[black!45, line width=0.6pt] (axis cs:0,{ymin+0.4:.2f}) -- (axis cs:0,0.6);",
         r"\node[font=\scriptsize, black!60, anchor=south east] at (axis cs:-0.4,0.6) {higher in the normal continuation};",
         r"\node[font=\scriptsize, black!60, anchor=south west] at (axis cs:0.4,0.6) {higher in the weird continuation};"]
    for key, y in ys.items():
        r = data[key]
        m, lo, hi = 100 * r["diff"], 100 * r["lo"], 100 * r["hi"]
        mk = "mark=*, read" if r["side"] == "weird" else "mark=o, read, fill=white"
        L.append(rf"\addplot[only marks, {mk}, mark size=2.2pt, line width=0.8pt, ci] coordinates {{({f(m,2)},{f(y,2)}) += ({f(hi-m,2)},0) -= ({f(m-lo,2)},0)}};")
        star = r"$\dagger$" if r["fw"] else ("**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else ""))
        if star:
            anc, xx = ("west", hi + 0.3) if r["side"] == "weird" else ("east", lo - 0.3)
            L.append(rf"\node[font=\scriptsize, anchor={anc}, inner sep=1pt] at (axis cs:{f(xx,2)},{f(y-0.05,2)}) {{{star}}};")
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_x_sae_split", "\n".join(L))


# ================================================================== (2) chat-only features
def contexts(lat_ids, n=4):
    import tokenizers
    tk = tokenizers.Tokenizer.from_file(os.path.join(REPO, "data/tokenizer/tokenizer.json"))
    z = np.load(os.path.join(S, "chat_index.npz")); lp, row, act, tok, seq = z["lat_ptr"], z["row"], z["act"], z["tok"], z["seq"]
    out = {}
    for L_ in lat_ids:
        a, b = lp[L_], lp[L_ + 1]; rr = row[a:b]; aa = act[a:b].astype(float)
        ex = []
        for j in np.argsort(-aa)[:n]:
            r = rr[j]; lo = r
            while lo > 0 and seq[lo - 1] == seq[r] and r - lo < 12:
                lo -= 1
            ex.append({"act": float(aa[j]), "before": tk.decode(tok[lo:r].tolist()), "tok": tk.decode([int(tok[r])])})
        out[int(L_)] = ex
    return out


CHAT_NOTES = {  # descriptions written from the top-activating chat contexts printed by contexts(); see explore_sae.json
    69151: "almost every reply token",
    57115: "almost every reply token",
    40688: "mid-sentence prose in replies",
    36086: "the model's own fiction narration",
    76037: "scene imagery in the model's fiction",
    117220: "bold list headings (Markdown)",
    13356: "citations: ``According to X (ABBR)''",
}


def fig_chatweb():
    lo = 2e-6
    c = np.maximum(CHAT, lo); w = np.maximum(WEB, lo)
    lc, lw = np.log10(c), np.log10(w)
    edges = np.linspace(np.log10(lo), 0, 25)
    H, _, _ = np.histogram2d(lw, lc, bins=[edges, edges])
    co = np.where((CHAT > 0.02) & (WEB < 0.002))[0]; wo = np.where((WEB > 0.02) & (CHAT < 0.002))[0]
    rho = float(np.corrcoef(np.argsort(np.argsort(CHAT)), np.argsort(np.argsort(WEB)))[0, 1])
    OUT["chatweb"] = {"n_chat_only_dense": int(len(co)), "n_web_only_dense": int(len(wo)), "fired_chat": int((CHAT > 0).sum()),
                      "fired_web": int((WEB > 0).sum()), "fired_chat_not_web": int(((CHAT > 0) & (WEB == 0)).sum()),
                      "spearman_all_latents": rho,
                      "highlight": {str(i): {"chat": float(CHAT[i]), "web": float(WEB[i])} for i in CHAT_NOTES}}
    OUT["chat_contexts"] = contexts(list(CHAT_NOTES))
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=8.6cm, height=8.6cm, xmin=-5.7, xmax=0, ymin=-5.7, ymax=0, xtick={-5,-4,-3,-2,-1,0}, ytick={-5,-4,-3,-2,-1,0},"
         r" xticklabels={$10^{-5}$,$10^{-4}$,$10^{-3}$,1\%,10\%,100\%}, yticklabels={$10^{-5}$,$10^{-4}$,$10^{-3}$,1\%,10\%,100\%},"
         r" axis x line*=bottom, axis y line*=left, xlabel={Share of web-text tokens the feature fires on}, ylabel={Share of chat-reply tokens the feature fires on}]"]
    mxh = np.log10(H.max())
    for i in range(24):
        for j in range(24):
            if H[i, j] > 0:
                t = 0.12 + 0.88 * np.log10(H[i, j]) / mxh
                L.append(rf"\fill[read!{int(100*t)}!white] (axis cs:{f(edges[i],3)},{f(edges[j],3)}) rectangle (axis cs:{f(edges[i+1],3)},{f(edges[j+1],3)});")
    L.append(r"\draw[black!50, dashed, line width=0.6pt] (axis cs:-5.7,-5.7) -- (axis cs:0,0);")
    L.append(r"\draw[neur, line width=0.7pt] (axis cs:-5.7,-1.699) -- (axis cs:-2.699,-1.699) -- (axis cs:-2.699,0);")
    L.append(rf"\node[font=\scriptsize, neur, anchor=north west, align=left] at (axis cs:-5.6,-0.85) {{{len(co)} features fire on $>$2\%\\of chat tokens and\\$<$0.2\% of web tokens}};")
    place = {69151: (0.1, 0.0), 57115: (0.1, -0.08), 40688: (0.1, 0.0), 36086: (0.1, 0.06), 76037: (0.1, -0.06),
             117220: (-0.1, 0.1), 13356: (0.1, -0.1)}
    key = []
    for n_, (i, txt) in enumerate(CHAT_NOTES.items(), 1):
        x, y = np.log10(max(WEB[i], lo)), np.log10(max(CHAT[i], lo))
        dx, dy = place[i]
        L.append(rf"\addplot[only marks, mark=*, mark size=1.8pt, neur] coordinates {{({f(x,3)},{f(y,3)})}};")
        L.append(rf"\node[font=\scriptsize\bfseries, neur!80!black, anchor={'west' if dx > 0 else 'east'}, inner sep=1pt] at (axis cs:{f(x+dx,3)},{f(y+dy,3)}) {{{n_}}};")
        key.append(rf"{n_}.\ {i}: {txt} ({100*CHAT[i]:.1f}\% of chat, {100*WEB[i]:.2f}\% of web tokens)")
    L += [r"\end{axis}",
          r"\node[anchor=north west, font=\scriptsize, align=left] at ($(current bounding box.north east)+(0.3cm,-0.2cm)$) {"
          r"\textbf{Chat-specific features, described from their}\\\textbf{top-activating chat contexts}\\[3pt]" + r"\\[1.5pt]".join(key) + "};",
          r"\end{tikzpicture}"]
    build("fig_x_sae_chatweb", "\n".join(L))


# ================================================================== (3) features aligned with the weird-vs-normal direction
def fig_dir():
    M = json.load(open(os.path.join(S, "m6d.json")))
    rmax = max(M["random_dir_max_abs_cos"])
    beh = [("embodiment", "Claiming a body"), ("code", "Fake code execution"), ("chemtrails", "Chemtrails"), ("dai", "Denying being an AI"),
           ("911", "Called 911"), ("laser", "Laser at aircraft"), ("drunk", "Drunk driving")]
    rows, data, rec = [], {}, {}
    for b, name in beh:
        top = [t for t in M["decoder_cos"][b]["L48"]["top"] if t.get("label")][:3]
        rec[b] = top
        rows.append(("h", name))
        for j, t in enumerate(top):
            key = f"{b}|{j}"; rows.append(("r", key, key)); data[key] = t
    OUT["dir"] = {"random_max": rmax, "top": rec}
    ys, ticks, _, ymin = F.rows_plot(rows)
    hdr = iter([r[1] for r in rows if r[0] == "h"])
    labels = [(r"\textbf{" + F.esc(next(hdr)) + "}") if r[0] == "h" else tex_label(LAB.get(str(data[r[2]]["lat"])) or data[r[2]]["label"]) for r in rows]
    H = 0.42 * (len(ticks) + 2.5)
    L = [r"\begin{tikzpicture}",
         rf"\begin{{axis}}[width=6.2cm, height={H:.2f}cm, ymin={ymin+0.4:.2f}, ymax=0.6, ytick={{{','.join(f(t,2) for t in ticks)}}},"
         rf" yticklabels={{{','.join('{' + l + '}' for l in labels)}}}, y tick label style={{font=\scriptsize}}, major y tick style={{draw=none}},"
         r" axis y line*=left, axis x line*=bottom, xmajorgrids, xmin=0, xmax=0.42, xtick={0,0.1,0.2,0.3,0.4},"
         r" xticklabel style={/pgf/number format/.cd, fixed, precision=1},"
         r" xlabel={Cosine of the feature's decoder vector\\with the weird-vs-normal direction}, xlabel style={align=center}]",
         rf"\fill[black!8] (axis cs:0,{ymin+0.4:.2f}) rectangle (axis cs:{f(rmax,3)},0.6);",
         rf"\node[font=\scriptsize, black!60, anchor=east, rotate=90, inner sep=2pt] at (axis cs:{f(rmax/2,3)},0.5) {{best random direction}};"]
    for key, y in ys.items():
        t = data[key]; i = t["lat"]
        chat_only = CHAT[i] > 0.02 and WEB[i] < 0.002
        mk = "mark=*, neur" if chat_only else "mark=*, read"
        L.append(rf"\addplot[only marks, {mk}, mark size=2.2pt] coordinates {{({f(t['cos'],3)},{f(y,2)})}};")
        L.append(rf"\draw[{'neur' if chat_only else 'read'}, line width=0.8pt] (axis cs:0,{f(y,2)}) -- (axis cs:{f(t['cos'],3)},{f(y,2)});")
    L.append(r"\node[font=\scriptsize, anchor=south east, align=right] at (axis cs:0.42,0.6) {\tikz\fill[neur] (0,0) circle (2.2pt); fires on $>$2\% of chat-reply tokens\\and $<$0.2\% of web tokens};")
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_x_sae_dir", "\n".join(L))


# ================================================================== (4) label quality
def fig_labels():
    D = json.load(open(os.path.join(S, "detfuzz.json")))
    Rr = json.load(open(os.path.join(S, "restate.json")))
    det = np.array(list(D["det"]["per_latent"].values())); fz = np.array(list(D["fuzz"]["per_latent"].values()))
    sh = np.array(list(D["shuf"]["per_latent"].values())) if "shuf" in D and "per_latent" in D["shuf"] else None
    OUT["labels"] = {"det_mean": float(det.mean()), "fuzz_mean": float(fz.mean()), "det_below_.6": float((det < 0.6).mean()),
                     "fuzz_below_.6": float((fz < 0.6).mean()), "shuf_mean": float(sh.mean()) if sh is not None else None,
                     "n_det": int(len(det)), "n_fuzz": int(len(fz)), "n_shuf": int(len(sh)) if sh is not None else 0,
                     "restate_real": Rr["real_all"], "restate_rand": Rr["rand_all"]}
    bins = np.linspace(0, 1, 11)
    def hist(v):
        h, _ = np.histogram(v, bins=bins); return 100 * h / len(v)
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[name=a, width=7.2cm, height=5.6cm, xmin=0, xmax=1, ymin=0, ymax=60, xtick={0,0.2,0.4,0.6,0.8,1},"
         r" xticklabel style={/pgf/number format/.cd, fixed, precision=1}, ytick={0,10,20,30,40,50,60}, yticklabel={$\pgfmathprintnumber{\tick}\%$},"
         r" axis x line*=bottom, axis y line*=left, ymajorgrids, xlabel={Held-out accuracy using the label (0.5 = chance)},"
         r" ylabel={Share of labels}, title={(a)}, title style={at={(-0.2,1.0)}, anchor=south west}]"]
    series = [(det, "read", "", "detection"), (fz, "read!50", "densely dashed", "fuzzing")]
    if sh is not None:
        series.append((sh, "ctrl", "", "shuffled label"))
    for v, col, dash, lab in series:
        h = hist(v)
        pts = " ".join(f"({f(bins[k],2)},{f(h[k],2)}) ({f(bins[k+1],2)},{f(h[k],2)})" for k in range(len(h)))
        L.append(rf"\addplot[{col}, line width=1.2pt{', ' + dash if dash else ''}] coordinates {{{pts}}};")
    L.append(r"\draw[black!50, dashed, line width=0.6pt] (axis cs:0.5,0) -- (axis cs:0.5,60);")
    L.append(r"\node[font=\scriptsize, ctrl!80!black, anchor=east] at (axis cs:0.39,45) {shuffled label};")
    L.append(r"\node[font=\scriptsize, read, anchor=west, align=left] at (axis cs:0.73,30) {detection};")
    L.append(r"\node[font=\scriptsize, read!60, anchor=west, align=left] at (axis cs:0.83,22) {fuzzing};")
    L.append(r"\end{axis}")
    rr, rn = Rr["real_all"], Rr["rand_all"]
    L += [r"\begin{axis}[at={($(a.east)+(2.4cm,0)$)}, anchor=west, width=4.4cm, height=5.6cm, ymin=0, ymax=50, xmin=-0.6, xmax=1.6,"
          r" xtick={0,1}, xticklabels={{active\\labels},{random\\features}}, xticklabel style={align=center, font=\footnotesize},"
          r" ytick={0,10,20,30,40,50}, yticklabel={$\pgfmathprintnumber{\tick}\%$}, axis x line*=bottom, axis y line*=left, ymajorgrids,"
          r" ylabel={Labels restating the last 64 tokens}, title={(b)}, title style={at={(-0.45,1.0)}, anchor=south west}]",
          rf"\fill[read] (axis cs:-0.3,0) rectangle (axis cs:0.3,{f(100*rr['rate'],2)});",
          rf"\fill[ctrl] (axis cs:0.7,0) rectangle (axis cs:1.3,{f(100*rn['rate'],2)});",
          rf"\draw[black, line width=0.8pt] (axis cs:0,{f(100*rr['ci'][0],2)}) -- (axis cs:0,{f(100*rr['ci'][1],2)});",
          rf"\draw[black, line width=0.8pt] (axis cs:1,{f(100*rn['ci'][0],2)}) -- (axis cs:1,{f(100*rn['ci'][1],2)});",
          r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_x_sae_labels", "\n".join(L))


if __name__ == "__main__":
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else ["split", "chatweb", "dir", "labels"]
    if "split" in only:
        fig_split(split_contrast())
    if "chatweb" in only:
        fig_chatweb()
    if "dir" in only:
        fig_dir()
    if "labels" in only:
        fig_labels()
    p = os.path.join(S, "explore_sae.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(OUT)
    json.dump(old, open(p, "w"), indent=1, default=float)
    print("wrote", p)
