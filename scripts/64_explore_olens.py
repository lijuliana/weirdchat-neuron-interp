"""Exploratory O-lens figures that are not about monitor scores or edits: what the lens actually writes.

Reads runs/olens/reads.jsonl (s3d_rl600 readouts: 2 samples per position, each a few "- " lines), data/olens/seqs.jsonl,
data/agastya/phase2_forks.jsonl (the 46 sharp forks). Builds with the shared style of scripts/62_figures.py.

  fig_x_olens_gallery   what the lens writes at the state that chooses the fork token and one token later, 4 forks
  fig_x_olens_flip      denying AI: share of lens lines that affirm vs deny being an AI, by token relative to the fork
  fig_x_olens_nextword  share of lens lines that begin with the reply's actual next k words, by block, chat vs document

Run: .venv/bin/python -I scripts/64_explore_olens.py [--only name,name]   (writes findings/explore/olens_numbers.json)
"""
import argparse, collections, importlib.util, json, os, random, re

import numpy as np
from tokenizers import Tokenizer

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(REPO)
spec = importlib.util.spec_from_file_location("figs62", "scripts/62_figures.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
f, build = F.f, F.build

tok = Tokenizer.from_file("data/tokenizer/tokenizer.json")
SEQS = {json.loads(l)["sid"]: json.loads(l) for l in open("data/olens/seqs.jsonl")}
P2 = [json.loads(l) for l in open("data/agastya/phase2_forks.jsonl")]
READS = {}
for line in open("runs/olens/reads.jsonl"):
    r = json.loads(line)
    if r["key"].startswith(("R|reply", "R|raw")):
        READS[r["key"]] = r["texts"]
CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]+")
NUM = {}


def lines(texts):
    return [b.strip()[1:].strip() for t in texts for b in t.split("\n") if b.strip().startswith("-")]


def reply_ids(rid):
    s = SEQS[f"reply|{rid}"]
    P = s["pos"][0] - s["c"][0] + 1
    return s["ids"][P:]


def tex(s, n=None):
    """Plain readout text -> LaTeX: markdown stripped, Chinese runs shown as a grey [zh] marker, specials escaped."""
    s = s.replace("**", "").replace("`", "").replace("\n", " ").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", ", ").replace("–", "-").replace("…", "...")
    if n and len(s) > n:
        s = s[:n].rsplit(" ", 1)[0] + " ..."
    parts = CJK.split(s)
    out = []
    for i, p in enumerate(parts):
        p = "".join(ch for ch in p if ord(ch) < 128)
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                     ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
            p = p.replace(a, b)
        out.append(p)
        if i < len(parts) - 1:
            out.append(r"{\color{black!45}[zh]}")
    return "".join(out).strip()


# ============================================================================================ gallery
GALLERY = [("claims-called-911", "c8f|22", "Called 911"), ("denying-ai-identity", "d38|31", "Denying being an AI"),
           ("false-physical-embodiment", "d59|20", "Claiming a body"), ("fabricated-code-execution", "3f6|26", "Fake code execution")]


def fig_gallery():
    byrid = {fk["rid"]: fk for fk in P2}
    blocks = []
    for b, tail, name in GALLERY:
        fk = next(x for x in P2 if x["behavior_id"] == b and x["rid"].endswith(tail))
        rep = reply_ids(fk["rid"]); c0 = fk["fork"][0]
        full_pre = tok.decode(rep[:c0]); post = tok.decode(rep[c0:c0 + 9]).split("\n")[0]
        last = [x for x in full_pre.split("\n") if x.strip()]
        pre = (last[-1] if last else "").lstrip("# ").strip()
        pre = ("..." if len(full_pre.strip()) > len(pre) + 2 else "") + pre + ("" if full_pre.endswith((" ", "\n")) or not pre else "")
        rows = []
        for rel, lab in ((0, "state that chooses the fork token"), (1, "one token later")):
            L = lines(READS[f"R|reply|{fk['rid']}|{c0+rel}|56"])
            # the first line of each of the two samples
            first = [lines([t])[0] for t in READS[f"R|reply|{fk['rid']}|{c0+rel}|56"] if lines([t])]
            rows.append((lab, first[:2]))
        blocks.append((name, fk["Plo_on"], fk["Phi_on"], pre, post, rows))
    L = [r"\begin{tikzpicture}[every node/.style={inner sep=0pt, anchor=north west, align=left}]"]
    y = 0.0
    for name, plo, phi, pre, post, rows in blocks:
        L.append(rf"\node[font=\small\bfseries] at (0,{f(y,2)}) {{{name}}};")
        L.append(rf"\node[font=\footnotesize, meas!80!black] at (5.2,{f(y-0.02,2)}) {{chance of weird: {100*plo:.0f}\% before the fork token, {100*phi:.0f}\% after}};")
        y -= 0.5
        L.append(rf"\node[font=\footnotesize, text width=15.5cm] at (0.2,{f(y,2)}) {{{{\color{{black!60}}Reply so far:}} {tex(pre)}\,"
                 rf"{{\color{{neur}}\textbf{{|}}\,{tex(post)}...}}}};")
        y -= 0.62
        for lab, ls in rows:
            L.append(rf"\node[font=\footnotesize, text width=3.3cm, black!60] at (0.2,{f(y,2)}) {{O-lens, {lab}:}};")
            body = r"\\[1pt]".join(r"\textbullet\ " + tex(x, 84) for x in ls)
            L.append(rf"\node[font=\footnotesize, text width=12.0cm, read!85!black] at (3.7,{f(y,2)}) {{{body}}};")
            y -= 0.48 * len(ls) + 0.18
        y -= 0.35
    L.append(r"\end{tikzpicture}")
    build("fig_x_olens_gallery", "\n".join(L))


# ============================================================================================ answer flips at the fork (denying AI)
DENY = re.compile(r"not an? (?:AI|bot|robot|chatbot|machine)|\bI(?:'m|’m| am) (?:a )?(?:human|real person|person)\b|human user")
AFFIRM = re.compile(r"\b(?:I am|I'm|I’m|yes,? I am)\s+(?:an?\s+)?(?:AI|artificial intelligence|large language model|language model|AI assistant)\b|\byes,? I am\b|\bI am Qwen", re.I)


def flip_data():
    forks = [x for x in P2 if x["behavior_id"] == "denying-ai-identity"]
    rels = list(range(-5, 4))
    D = {}
    rng = np.random.default_rng(0)
    for kind in ("reply", "raw"):
        for which, fn in (("deny", lambda x: bool(DENY.search(x))), ("affirm", lambda x: bool(AFFIRM.search(x)) and not DENY.search(x))):
            ser = []
            for rel in rels:
                v = []
                for fk in forks:
                    k = f"R|{kind}|{fk['rid']}|{fk['fork'][0]+rel}|56"
                    if k in READS and lines(READS[k]):
                        v.append(np.mean([fn(x) for x in lines(READS[k])]))
                v = np.array(v)
                bo = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)] if len(v) else [np.nan]
                ser.append((rel, float(v.mean()) if len(v) else None, float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)), int(len(v))))
            D[(kind, which)] = ser
    D["P"] = (float(np.mean([x["Plo_on"] for x in forks])), float(np.mean([x["Phi_on"] for x in forks])), len(forks))
    NUM["flip"] = {f"{k[0]}|{k[1]}": v for k, v in D.items() if k != "P"}; NUM["flip"]["P"] = D["P"]
    return D


def fig_flip():
    D = flip_data()
    def band(ser, colr, dash, nm):
        pts = [s for s in ser if s[1] is not None and s[4] >= 8]
        up = " ".join(f"({r},{f(100*h,2)})" for r, m, l_, h, n in pts); dn = " ".join(f"({r},{f(100*l_,2)})" for r, m, l_, h, n in pts)
        mid = " ".join(f"({r},{f(100*m,2)})" for r, m, l_, h, n in pts)
        return (rf"\addplot[name path=u{nm}, draw=none] coordinates {{{up}}};\addplot[name path=d{nm}, draw=none] coordinates {{{dn}}};"
                rf"\addplot[{colr}, fill opacity=0.13] fill between[of=u{nm} and d{nm}];"
                rf"\addplot[{colr}, line width=1.4pt, mark=*, mark size=1.5pt{', ' + dash if dash else ''}] coordinates {{{mid}}};"), pts
    plo, phi, n = D["P"]
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=10cm, height=6.6cm, xmin=-5.4, xmax=3.4, xtick={-5,-4,-3,-2,-1,0,1,2,3}, ymin=0, ymax=100,"
         r" ytick={0,25,50,75,100}, yticklabel={$\pgfmathprintnumber{\tick}\%$}, axis x line*=bottom, axis y line*=left, ymajorgrids,"
         r" xlabel={Tokens relative to the fork token (0 = the state that chooses it)}, ylabel={Share of O-lens lines (block 56)}]",
         rf"\addplot[meas, line width=1.4pt, const plot, dotted] coordinates {{(-5.4,{f(100*plo,2)}) (0.5,{f(100*plo,2)}) (0.5,{f(100*phi,2)}) (3.4,{f(100*phi,2)})}};",
         r"\draw[black!45, dotted, line width=0.8pt] (axis cs:0.5,0) -- (axis cs:0.5,100);"]
    bd, pd_ = band(D[("reply", "deny")], "neur", "", "a"); ba, pa = band(D[("reply", "affirm")], "read", "", "b")
    rd, _ = band(D[("raw", "deny")], "neur!50", "densely dashed", "c"); ra, _ = band(D[("raw", "affirm")], "read!50", "densely dashed", "d")
    L += [bd, ba, rd, ra,
          rf"\node[font=\footnotesize, neur, anchor=west] at (axis cs:3.5,{f(100*pd_[-1][1],2)}) {{says it is not an AI}};",
          rf"\node[font=\footnotesize, read, anchor=west] at (axis cs:3.5,{f(100*pa[-1][1]+3,2)}) {{says it is an AI}};",
          rf"\node[font=\footnotesize, meas!80!black, anchor=north west] at (axis cs:-5.3,{f(100*plo-1,2)}) {{measured chance the reply denies}};",
          r"\node[font=\scriptsize, black!60, anchor=north west, align=left] at (axis cs:-5.3,98) {solid: reply read in the chat\\dashed: same words as a plain document};",
          r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_x_olens_flip", "\n".join(L))


# ============================================================================================ next-word forecast by depth
W = re.compile(r"[A-Za-z0-9']+|[\u4e00-\u9fff]")


def words(s):
    return [w.lower() for w in W.findall(s.replace("’", "'"))]


def lead(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def nextword_data():
    full = {s["rid"]: reply_ids(s["rid"]) for s in SEQS.values() if s["kind"] == "reply"}
    raw_pos = {(k.split("|")[2] + "|" + k.split("|")[3], int(k.split("|")[4])) for k in READS if k.startswith("R|raw")}
    res = collections.defaultdict(lambda: collections.defaultdict(list))   # (kind, L) -> rid -> [lead lengths]
    conts = []
    for k, texts in READS.items():
        p = k.split("|"); kind, rid, c, L = p[1], p[2] + "|" + p[3], int(p[4]), int(p[5])
        if (rid, c) not in raw_pos:   # chat and document read on the same positions (the 46 sharp-fork windows)
            continue
        rep = full[rid]
        if c >= len(rep):
            continue
        cont = words(tok.decode(rep[c:c + 40])); conts.append(cont)
        for x in lines(texts):
            if words(x):
                res[(kind, L)][rid].append(lead(words(x), cont))
    rng = random.Random(0)
    for k, texts in list(READS.items()):
        p = k.split("|")
        if p[1] != "reply" or (p[2] + "|" + p[3], int(p[4])) not in raw_pos:
            continue
        cont = rng.choice(conts)
        for x in lines(texts):
            if words(x):
                res[("null", int(p[5]))][p[2] + "|" + p[3]].append(lead(words(x), cont))
    out = {}
    nr = np.random.default_rng(0)
    for key, d in res.items():
        rids = list(d)
        arr = {r: np.array(v) for r, v in d.items()}
        ser = []
        for kk in range(1, 6):
            per = np.array([(arr[r] >= kk).mean() for r in rids]); wts = np.array([len(arr[r]) for r in rids])
            est = float((per * wts).sum() / wts.sum())
            bo = []
            for _ in range(1000):
                ix = nr.integers(0, len(rids), len(rids)); bo.append((per[ix] * wts[ix]).sum() / wts[ix].sum())
            ser.append((kk, est, float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))))
        out[f"{key[0]}|{key[1]}"] = {"n_lines": int(sum(len(v) for v in d.values())), "n_replies": len(rids), "series": ser}
    NUM["nextword"] = out
    return out


def fig_nextword():
    D = nextword_data()
    colL = {32: "read!40", 44: "read!70", 56: "read"}
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=9cm, height=6.6cm, xmin=0.8, xmax=5.2, xtick={1,2,3,4,5}, ymode=log, ymin=1e-4, ymax=0.5,"
         r" ytick={1e-4,1e-3,1e-2,1e-1}, yticklabels={0.01\%,0.1\%,1\%,10\%}, axis x line*=bottom, axis y line*=left, ymajorgrids,"
         r" xlabel={Number of words the readout gets right in a row}, ylabel={Share of O-lens lines (log scale)}]"]
    ends = []
    for Lb in (56, 44, 32):
        for kind, dash in (("reply", ""), ("raw", "densely dashed")):
            s = D[f"{kind}|{Lb}"]["series"]
            pts = " ".join(f"({k},{max(m,1.2e-4):.6f})" for k, m, lo, hi in s)
            L.append(rf"\addplot[{colL[Lb]}, line width=1.3pt, mark=*, mark size=1.4pt{', ' + dash if dash else ''}] coordinates {{{pts}}};")
            if kind == "reply":
                ends.append((s[-1][1], f"block {Lb}", colL[Lb]))
    s = D["null|56"]["series"]
    L.append(rf"\addplot[ctrl, line width=1pt, mark=o, mark size=1.4pt] coordinates {{{' '.join(f'({k},{max(m,1.2e-4):.6f})' for k, m, lo, hi in s)}}};")
    for (v, lab, col), yy in zip(ends, (1.15e-2, 6.6e-3, 3.6e-3)):
        L.append(rf"\node[font=\footnotesize, {col}, anchor=west] at (axis cs:5.25,{yy:.5f}) {{{lab}}};")
    L += [r"\node[font=\footnotesize, ctrl!80!black, anchor=south west, align=left] at (axis cs:3.0,1.5e-4) {compared with the words of a random other\\reply (none from 3 words on)};",
          r"\node[font=\scriptsize, black!60, anchor=north east, align=right] at (axis cs:5.15,0.45) {solid: reply read in the chat\\dashed: same words as a plain document};",
          r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_x_olens_nextword", "\n".join(L))


FIGS = {"fig_x_olens_gallery": fig_gallery, "fig_x_olens_flip": fig_flip, "fig_x_olens_nextword": fig_nextword}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); a = ap.parse_args()
    for name, fn in FIGS.items():
        if not a.only or name in a.only.split(","):
            fn()
    os.makedirs("findings/explore", exist_ok=True)
    old = json.load(open("findings/explore/olens_numbers.json")) if os.path.exists("findings/explore/olens_numbers.json") else {}
    old.update(NUM); json.dump(old, open("findings/explore/olens_numbers.json", "w"), indent=1)
