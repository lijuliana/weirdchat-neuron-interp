"""Exploratory figures from the saved NLA and J-lens readouts (not monitor AUROC, not eval C).

  fig_x_nla_chat    the NLA recognises a chat at the start of a reply and calls it a blog or forum post further in
  fig_x_nla_quotes  the NLA's quotes of the text it is reading keep the topic and invent the wording
  fig_x_jlens_ahead the J-lens top-20 contains words the model writes several tokens later
  fig_x_jlens_forks what the J-lens reads at fork states, and the testing concept rising before fake-code forks

Data: data/olens/seqs.jsonl, data/agastya/phase2_forks.jsonl, runs/monitors/lens/reads.jsonl, runs/monitors/nla/reads.jsonl,
runs/monitors/nla/val/web_reads.jsonl, runs/evalc/nla_c8/neurons_nla.jsonl. Style and build settings from scripts/62_figures.py.
Writes figures/fig_x_*.{pdf,svg}, figures/tex/fig_x_*.tex, findings/explore/nla_jlens_stats.json.

Run: .venv/bin/python -I scripts/64_explore_nla_jlens.py
"""
import ast, collections, importlib.util, json, math, os, random, re, subprocess, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("figs62", os.path.join(REPO, "scripts/62_figures.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
from tokenizers import Tokenizer

TOK = Tokenizer.from_file(os.path.join(REPO, "data/tokenizer/tokenizer.json"))
f, esc = F.f, F.esc
STATS = {}


def build(name, body):
    """Like 62's build, with CJKutf8 loaded so Chinese lens tokens render."""
    os.makedirs(F.TEX, exist_ok=True)
    open(os.path.join(F.TEX, "style.tex"), "w").write(F.STYLE)
    p = os.path.join(F.TEX, name + ".tex")
    open(p, "w").write("\\documentclass[border=4pt]{standalone}\n\\input{style.tex}\n\\usepackage{CJKutf8}\n\\usepackage{array}\n"
                       "\\begin{document}\n" + body + "\n\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], cwd=F.TEX, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); sys.exit(f"pdflatex failed for {name}")
    os.replace(os.path.join(F.TEX, name + ".pdf"), os.path.join(F.OUT, name + ".pdf"))
    subprocess.run(["pdftocairo", "-svg", os.path.join(F.OUT, name + ".pdf"), os.path.join(F.OUT, name + ".svg")], check=True)
    for ext in ("aux", "log"):
        q = os.path.join(F.TEX, f"{name}.{ext}")
        if os.path.exists(q):
            os.remove(q)
    print("built", name)


# ------------------------------------------------------------------------------------------------ data
def jl(path):
    return [json.loads(l) for l in open(os.path.join(REPO, path)) if l.strip()]


SEQS = {s["sid"]: s for s in jl("data/olens/seqs.jsonl")}
FORKS = jl("data/agastya/phase2_forks.jsonl")
LENS = collections.defaultdict(dict)
for r in jl("runs/monitors/lens/reads.jsonl"):
    LENS[(r["sid"], r["pos"])][(r["lens"], r["layer"])] = r["items"]
NLA = collections.defaultdict(dict)
for r in jl("runs/monitors/nla/reads.jsonl"):
    NLA[(r["sid"], r["pos"])][r["layer"]] = r["texts"]
VC = {}


def tstr(i):
    if i not in VC:
        VC[i] = TOK.decode([i]).strip().lower()
    return VC[i]


def cpos(sid):
    s = SEQS[sid]
    return dict(zip(s["c"], s["pos"])), dict(zip(s["pos"], s["c"]))


def fork_key(fk):
    c0 = fk["fork"][0]
    if c0 == 0:
        sp = SEQS["prompt|" + fk["rid"].split("|")[0]]
        return sp["sid"], sp["pos"][-1]
    return "reply|" + fk["rid"], cpos("reply|" + fk["rid"])[0][c0]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


# ------------------------------------------------------------------------------------------------ NLA: chat or web page
AI_RX = re.compile(r"\b(AI|chatbot|chat ?bot|assistant|LLM|language model|chat (interface|platform|transcript|conversation|log|response|reply)"
                   r"|ChatGPT|conversational agent)\b", re.I)
GENRES = [("blog post", r"blog"), ("forum or Q\\&A answer", r"forum|q&a|q/a|stack ?overflow|stack ?exchange|reddit|thread|quora|answer"),
          ("tutorial or guide", r"tutorial|how-to|how to|guide|instruction"), ("news or magazine", r"news|magazine|journalis|press"),
          ("academic or reference", r"academic|paper|abstract|encyclop|wiki|textbook|reference|documentation|docs"),
          ("fiction or role-play", r"fiction|story|novel|roleplay|role-play|creative")]
first_line = lambda t: t.replace("<explanation>", "").strip().split("\n")[0]


def fig_nla_chat():
    bins = [(0, 4, "0--3"), (4, 16, "4--15"), (16, 64, "16--63"), (64, 256, "64--255"), (256, 10 ** 6, "256+")]
    k, n = collections.Counter(), collections.Counter()
    gc, gt = collections.Counter(), 0
    for (sid, p), d in NLA.items():
        if not sid.startswith("reply|"):
            continue
        c = cpos(sid)[1][p]
        bi = next(i for i, (a, b, _) in enumerate(bins) if a <= c < b)
        for L, tx in d.items():
            fl = first_line(tx[0]); a = bool(AI_RX.search(fl))
            n[(L, bi)] += 1; k[(L, bi)] += a
            if L == 44 and not a:
                gt += 1; gc[next((g for g, rx in GENRES if re.search(rx, fl, re.I)), "other")] += 1
    web = jl("runs/monitors/nla/val/web_reads.jsonl")
    neu = jl("runs/evalc/nla_c8/neurons_nla.jsonl")
    tx0 = lambda r: (ast.literal_eval(r["texts"]) if isinstance(r["texts"], str) else r["texts"])[0]
    web_ai = sum(bool(AI_RX.search(first_line(r["texts"][0]))) for r in web) / len(web)
    neu_ai = sum(bool(AI_RX.search(first_line(tx0(r)))) for r in neu) / len(neu)
    S = {"by_bin": {L: [[b[2], n[(L, i)], *wilson(k[(L, i)], n[(L, i)])] for i, b in enumerate(bins)] for L in (32, 44, 56)},
         "overall": {L: sum(k[(L, i)] for i in range(5)) / sum(n[(L, i)] for i in range(5)) for L in (32, 44, 56)},
         "web_text": web_ai, "n_web": len(web), "neuron_dirs": neu_ai, "n_neuron": len(neu),
         "genres_L44_not_chat": {g: v / gt for g, v in gc.most_common()}, "n_not_chat_L44": gt}
    STATS["nla_chat"] = S
    L_ = [r"\begin{tikzpicture}",
          r"\begin{axis}[name=a, width=7.6cm, height=6.2cm, xmin=-0.3, xmax=4.3, xtick={0,1,2,3,4},"
          rf" xticklabels={{{','.join(b[2] for b in bins)}}}, ymin=0, ymax=100, ytick={{0,25,50,75,100}}, yticklabel={{$\pgfmathprintnumber{{\tick}}\%$}},"
          r" axis x line*=bottom, axis y line*=left, ymajorgrids, xlabel={Tokens into the reply}, ylabel={Explanations that name an AI chat},"
          r" title={(a)}, title style={at={(-0.22,1.0)}, anchor=south west}]"]
    sty = {32: ("read!45", "square*", "densely dashed"), 44: ("read", "*", ""), 56: ("read!75!black", "triangle*", "dashed")}
    for L in (32, 44, 56):
        c, mk, ds = sty[L]
        pts = " ".join(f"({i},{f(100*v[2],2)}) += (0,{f(100*(v[4]-v[2]),2)}) -= (0,{f(100*(v[2]-v[3]),2)})" for i, v in enumerate(S["by_bin"][L]))
        L_.append(rf"\addplot[{c}, line width=1.2pt, mark={mk}, mark size=2pt, {ds + ', ' if ds else ''}yci] coordinates {{{pts}}};")
    leg = r"\\".join(rf"\tikz\draw[{sty[L][0]}, line width=1.2pt{', ' + sty[L][2] if sty[L][2] else ''}] (0,0) -- (0.5,0); block {L}" for L in (32, 44, 56))
    L_.append(rf"\node[font=\scriptsize, anchor=north east, align=left] at (axis cs:4.25,98) {{{leg}}};")
    L_ += [rf"\draw[black!50, dashed, line width=0.7pt] (axis cs:-0.3,{f(100*neu_ai,2)}) -- (axis cs:4.3,{f(100*neu_ai,2)}) node[pos=0.98, anchor=south east, font=\scriptsize, black!60] {{neuron output directions ({100*neu_ai:.0f}\%)}};",
           r"\end{axis}"]
    # (b) what the rest are called
    G = [g for g, _ in sorted(S["genres_L44_not_chat"].items(), key=lambda x: -x[1])]
    L_.append(r"\begin{axis}[at={($(a.east)+(3.6cm,0)$)}, anchor=west, width=5.6cm, height=6.2cm, xbar, bar width=7pt, xmin=0, xmax=50,"
              rf" ymin=-0.6, ymax={len(G)-0.4}, ytick={{{','.join(str(i) for i in range(len(G)))}}}, yticklabels={{{','.join('{'+g+'}' for g in reversed(G))}}},"
              r" y tick label style={font=\footnotesize}, major y tick style={draw=none}, xticklabel={$\pgfmathprintnumber{\tick}\%$},"
              r" axis x line*=bottom, axis y line*=left, xmajorgrids, xlabel={Share of the other explanations, block 44},"
              r" title={(b)}, title style={at={(-0.9,1.0)}, anchor=south west}, nodes near coords={\pgfmathprintnumber[fixed, precision=0]{\pgfplotspointmeta}\%},"
              r" nodes near coords style={font=\scriptsize}, point meta=x]")
    L_.append(rf"\addplot[fill=read!70, draw=none] coordinates {{{' '.join(f'({f(100*S['genres_L44_not_chat'][g],2)},{len(G)-1-i})' for i, g in enumerate(G))}}};")
    L_ += [r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_x_nla_chat", "\n".join(L_))


# ------------------------------------------------------------------------------------------------ NLA: quotes
Q_RX = re.compile(r'["\u201c]([^"\u201d]{1,400})["\u201d]')
WORDS = lambda s: re.findall(r"[a-z0-9']+", s.lower())
STOP = set("the a an and or of to in on for with is are was be it this that as at by from i you my your we our they their not no but so if "
           "then than there here have has had do does did will would can could should just very more most also into about".split())
content = lambda ws: [w for w in ws if len(w) >= 4 and w not in STOP]
QUOTE_OK = {"claims-called-911", "denying-ai-identity", "fabricated-code-execution", "false-physical-embodiment", "laser-at-aircraft",
            "chemtrails-assertion", "recommends-drunk-driving"}


def nla_quote(t):
    m = re.search(r"Final tokens?[^\n]*", t) or re.search(r"Ends[^\n]*", t)
    if not m:
        return None
    qs = Q_RX.findall(m.group(0))
    return qs if qs else None


def fig_nla_quotes():
    random.seed(0)
    keys = [k for k in NLA if k[0].startswith("reply|")]
    ctx = {k: set(content(WORDS(TOK.decode(SEQS[k[0]]["ids"][max(0, k[1] - 60):k[1] + 1])))) for k in keys}
    rows, last = [], collections.Counter()
    for k in keys:
        other = random.choice([o for o in random.sample(keys, 6) if o[0] != k[0]])
        ids = SEQS[k[0]]["ids"]; actual_w = WORDS(TOK.decode(ids[max(0, k[1] - 300):k[1] + 1]))
        for L, tx in NLA[k].items():
            qs = nla_quote(tx[0])
            if not qs:
                continue
            qw = WORDS(qs[-1])
            if qw and actual_w:
                last[(L, "n")] += 1; last[(L, "hit")] += qw[-1] == actual_w[-1]
            q = set(content(WORDS(" ".join(qs))))
            if len(q) < 3:
                continue
            rows.append({"L": L, "own": len(q & ctx[k]) / len(q), "ctl": len(q & ctx[other]) / len(q), "nq": len(q), "key": k, "qs": qs,
                         "b": SEQS[k[0]]["b"]})
    S = {}
    for L in (32, 44, 56):
        rr = [r for r in rows if r["L"] == L]
        own = sorted(r["own"] for r in rr)
        S[L] = {"n": len(rr), "own_mean": sum(own) / len(own), "own_median": own[len(own) // 2], "ctl_mean": sum(r["ctl"] for r in rr) / len(rr),
                "own_ge80": sum(o >= 0.8 for o in own) / len(own), "own_zero": sum(o == 0 for o in own) / len(own),
                "last_word_exact": last[(L, "hit")] / last[(L, "n")], "n_last": last[(L, "n")]}
    STATS["nla_quotes"] = S
    # histogram at block 44
    edges = [i / 10 for i in range(11)]
    def hist(vals):
        h = [0] * 10
        for v in vals:
            h[min(int(v * 10), 9)] += 1
        return [x / len(vals) for x in h]
    r44 = [r for r in rows if r["L"] == 44]
    ho, hc = hist([r["own"] for r in r44]), hist([r["ctl"] for r in r44])
    # examples: one per behavior (911, claiming a body, denying AI, fake code execution), block 44; the longest quote in the
    # final-tokens sentence, 10 to 30 words; its own content-word overlap with the true last 60 tokens closest to .35
    ex = []
    for b in ("claims-called-911", "false-physical-embodiment", "denying-ai-identity", "fabricated-code-execution"):
        cand = []
        for r in r44:
            if r["b"] != b:
                continue
            q = max(r["qs"], key=len); qw = WORDS(q); qc = set(content(qw))
            if 10 <= len(qw) <= 30 and len(qc) >= 5:
                cand.append((abs(len(qc & ctx[r["key"]]) / len(qc) - 0.35), r["key"], q, len(qc & ctx[r["key"]]) / len(qc)))
        if cand:
            c = min(cand, key=lambda x: (x[0], x[1])); ex.append({"key": c[1], "q": c[2], "own": c[3], "b": b})
    STATS["nla_quote_examples"] = [{"b": r["b"], "own": r["own"], "quote": r["q"]} for r in ex]
    L_ = [r"\begin{tikzpicture}",
          r"\begin{axis}[name=a, width=7cm, height=6cm, ybar, bar width=5pt, xmin=-0.05, xmax=1.05, ymin=0, ymax=100,"
          r" xtick={0,0.2,0.4,0.6,0.8,1}, xticklabels={0\%,20\%,40\%,60\%,80\%,100\%}, ytick={0,25,50,75,100}, yticklabel={$\pgfmathprintnumber{\tick}\%$},"
          r" axis x line*=bottom, axis y line*=left, ymajorgrids, xlabel={Quoted content words found in the true text},"
          r" ylabel={Share of explanations}, title={(a)}, title style={at={(-0.22,1.0)}, anchor=south west}]",
          rf"\addplot[fill=ctrl!60, draw=none, bar shift=-2.6pt] coordinates {{{' '.join(f'({f(e+0.05,3)},{f(100*v,2)})' for e, v in zip(edges, hc))}}};",
          rf"\addplot[fill=read, draw=none, bar shift=2.6pt] coordinates {{{' '.join(f'({f(e+0.05,3)},{f(100*v,2)})' for e, v in zip(edges, ho))}}};",
          r"\node[font=\footnotesize, read, anchor=west, align=left] at (axis cs:0.36,30) {against the text\\being read};",
          r"\node[font=\footnotesize, ctrl!80!black, anchor=west, align=left] at (axis cs:0.13,90) {against the text\\of another reply};",
          r"\end{axis}"]
    # (b) example table
    def hl(text, other):
        out = []
        for tok in re.split(r"(\s+)", text):
            w = WORDS(tok)
            t = esc(tok.replace("\\n", " ").replace("#", "\\#").replace("$", "\\$").replace("{", "").replace("}", "").replace("~", "").replace("^", ""))
            out.append(rf"\textbf{{{t}}}" if w and any(x in other and len(x) >= 4 and x not in STOP for x in w) else t)
        return "".join(out)
    cells = []
    for r in ex:
        ids = SEQS[r["key"][0]]["ids"]; p = r["key"][1]
        true = TOK.decode(ids[max(0, p - 22):p + 1]).split("</think>")[-1].replace("\n", " ").strip()
        q = r["q"].replace("\n", " ").strip()
        tw, qw = set(WORDS(true)), set(WORDS(q))
        name = F.SHORTB.get(r["b"], r["b"])
        cells.append(rf"{{\color{{black!60}}\scriptsize {name}}} & \ldots{{}}{hl(true, qw)} & {hl(q, tw)}\\[3pt]")
    L_ += [r"\node[anchor=north west, font=\footnotesize] at ($(a.north east)+(0.7cm,0.55cm)$) {(b)\enspace"
           r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{1.6cm}>{\raggedright\arraybackslash}p{4.6cm}>{\raggedright\arraybackslash}p{4.6cm}@{}}"
           r"& {\color{black!60}What the model had just written} & {\color{black!60}What the NLA quotes as the final tokens}\\[3pt]" + "".join(cells) +
           r"\end{tabular}};", r"\end{tikzpicture}"]
    build("fig_x_nla_quotes", "\n".join(L_))


# ------------------------------------------------------------------------------------------------ J-lens: lookahead
def fig_jlens_ahead():
    OFF = list(range(-16, 0)) + list(range(1, 33))
    keys = [("jlens", 32), ("jlens", 44), ("jlens", 56), ("jpp", 56), ("rlens", 44)]
    hit, tot = collections.Counter(), collections.Counter()
    rkeys = [k for k in LENS if k[0].startswith("reply|")]
    random.seed(1)
    for (sid, p) in rkeys:
        d = LENS[(sid, p)]; ids = SEQS[sid]["ids"]
        for k in keys:
            if k not in d:
                continue
            S = {x.strip().lower() for x in d[k]}
            for off in OFF:
                q = p + off
                if 0 <= q < len(ids):
                    w = tstr(ids[q])
                    if len(w) >= 3 and any(ch.isalnum() for ch in w):
                        tot[(k, off)] += 1; hit[(k, off)] += w in S
            if k == ("jlens", 44):
                o_sid, o_p = random.choice(rkeys); oids = SEQS[o_sid]["ids"]
                for off in (1, 2, 4, 8):
                    if o_p + off < len(oids):
                        w = tstr(oids[o_p + off])
                        if len(w) >= 3 and any(ch.isalnum() for ch in w):
                            tot[("shuf", off)] += 1; hit[("shuf", off)] += w in S
    R = {f"{k[0]}{k[1]}": {off: hit[(k, off)] / tot[(k, off)] for off in OFF if tot[(k, off)]} for k in keys}
    R["shuffled_jlens44"] = {off: hit[("shuf", off)] / tot[("shuf", off)] for off in (1, 2, 4, 8)}
    R["n_positions"] = len(rkeys)
    STATS["jlens_ahead"] = R
    sty = {("jlens", 32): ("read!40", "densely dashed"), ("jlens", 44): ("read", ""), ("jlens", 56): ("read!70!black", "dashed"),
           ("jpp", 56): ("dirc", "dotted"), ("rlens", 44): ("pers", "dashdotted")}
    lab = {("jlens", 32): "J-lens, block 32", ("jlens", 44): "J-lens, block 44", ("jlens", 56): "J-lens, block 56",
           ("jpp", 56): "J++, block 56", ("rlens", 44): "R-lens, block 44"}
    L_ = [r"\begin{tikzpicture}",
          r"\begin{axis}[width=10.5cm, height=6.6cm, xmin=-17, xmax=33, xtick={-16,-8,-4,-1,1,4,8,16,24,32},"
          r" ymode=log, ymin=0.002, ymax=1, ytick={0.003,0.01,0.03,0.1,0.3,1}, yticklabels={0.3\%,1\%,3\%,10\%,30\%,100\%},"
          r" axis x line*=bottom, axis y line*=left, grid=major, xlabel={Position of the reply token relative to the state read (tokens)},"
          r" ylabel={Token found in the lens's top 20}]",
          r"\draw[black!50, dotted, line width=0.8pt] (axis cs:0,0.002) -- (axis cs:0,1);",
          r"\node[font=\scriptsize, black!60, anchor=north east] at (axis cs:-0.5,0.95) {already written};",
          r"\node[font=\scriptsize, black!60, anchor=north west] at (axis cs:0.5,0.95) {not yet written};"]
    for k in keys:
        c, ds = sty[k]
        for seg in ([o for o in OFF if o < 0], [o for o in OFF if o > 0]):
            L_.append(rf"\addplot[{c}, line width={'1.5' if k == ('jlens',44) else '1.0'}pt{', ' + ds if ds else ''}] coordinates {{{' '.join(f'({o},{R[k[0]+str(k[1])][o]:.5f})' for o in seg)}}};")
        pass
    ends = sorted(((math.log10(R[k[0] + str(k[1])][32]), k) for k in keys), reverse=True)
    yy = []
    for v, k in ends:
        yy.append([min(v, yy[-1][0] - 0.2) if yy else v, k])
    for v, k in yy:
        L_.append(rf"\node[font=\scriptsize, {sty[k][0]}, anchor=west] at (axis cs:32.5,{10**v:.5f}) {{{lab[k]}}};")
    sh = R["shuffled_jlens44"]
    L_.append(rf"\addplot[ctrl, only marks, mark=x, mark size=2.5pt] coordinates {{{' '.join(f'({o},{v:.5f})' for o, v in sh.items())}}};")
    L_.append(rf"\node[font=\scriptsize, ctrl!80!black, anchor=north west] at (axis cs:1,{sh[1]*0.9:.5f}) {{block 44, token from another reply}};")
    L_ += [r"\end{axis}", r"\end{tikzpicture}"]
    # stack labels on the right that collide
    build("fig_x_jlens_ahead", "\n".join(L_))


# ------------------------------------------------------------------------------------------------ J-lens: forks
GLOSS = {"紧急": "emergency", "急救": "first aid", "呼叫": "call", "拨打": "dial", "见过": "have seen", "实测": "measured", "测试": "test",
         "实验": "experiment", "测试结果": "test results", "人工智能": "AI", "的人工智能": "AI", "机器人": "robot", "的真实": "real", "伪装": "disguise",
         "匿名": "anonymous", "鞋底": "sole", "踩踏": "trample", "踩": "step", "脚趾": "toes", "迈步": "stride", "手机": "phone",
         "智能手机": "smartphone", "测量": "measure", "各自": "each", "安全": "safe", "是的": "yes", "否": "no", "请立即": "immediately",
         "紧急情况": "emergency", "救助": "rescue", "无法满足": "cannot meet", "无法": "unable", "不能": "cannot", "我不能": "I cannot",
         "激光": "laser", "光束": "beam", "照射": "shine on", "不存在": "does not exist", "没有任何": "none at all", "并无": "there is no",
         "虚构": "fictional", "我自己": "myself", "模拟": "simulate", "固定": "fixed", "手机的": "phone's", "安全的": "safe", "的安全": "safety"}
FORK_ROWS = [39, 8, 2, 40, 22, 4, 12]   # indices into data/agastya/phase2_forks.jsonl (allowed behaviors only)
TEST_RX = re.compile(r"^(test|tests|tested|testing|experiment|experiments|experimental|实测|测试|实验|测试结果|myself|reproduced|reproduce|"
                     r"repro|reprodu|replicated|own)$")


def tok_tex(t, weird):
    s = t.strip()
    if re.search(r"[\u4e00-\u9fff]", s):
        g = GLOSS.get(s)
        if g is None:
            return None
        return rf"\begin{{CJK}}{{UTF8}}{{gbsn}}{s}\end{{CJK}}\,{{\color{{black!50}}\scriptsize({g})}}"
    if not s or not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", s):
        return None
    out = esc(s)
    return rf"\textcolor{{neur}}{{\textbf{{{out}}}}}" if weird and s.lower() == weird else out


def top_tokens(items, weird, n=5):
    out, seen = [], set()
    for t in items:
        k = t.strip().lower()
        if k in seen:
            continue
        x = tok_tex(t, weird)
        if x:
            out.append(x); seen.add(k)
        if len(out) == n:
            break
    return ", ".join(out)


def fig_jlens_forks():
    rows = []
    for i in FORK_ROWS:
        fk = FORKS[i]
        sid, p = fork_key(fk); d = LENS[(sid, p)]; ids = SEQS[sid]["ids"]
        w = fk["fork_token_str"].strip().lower()
        ctx = TOK.decode(ids[max(0, p - 7):p + 1]).replace("\n", " ").split("</think>")[-1].strip()
        ctx = re.sub(r"\s+", " ", ctx.replace("*", "")).strip()[-34:] or "(start of the reply)"
        opts = sorted(fk["next_tokens"], key=lambda t: -t["p"])[:3]
        if fk["fork_token"] not in [t["tok"] for t in opts]:
            opts = opts[:2] + [t for t in fk["next_tokens"] if t["tok"] == fk["fork_token"]]
        ot = ", ".join((rf"\textcolor{{neur}}{{\textbf{{{esc(t['str'].strip())}}}}}" if t["tok"] == fk["fork_token"] else esc(t["str"].strip()))
                       + rf"\,{{\scriptsize {100*t['p']:.0f}\%}}" for t in opts)
        rows.append((F.SHORTB[fk["behavior_id"]], esc(ctx), ot, top_tokens(d[("jlens", 44)], w), top_tokens(d[("jlens", 56)], w)))
    # weird token in top-20 across all forks
    cnt = collections.Counter(); nf = 0
    for fk in FORKS:
        try:
            sid, p = fork_key(fk)
        except KeyError:
            continue
        d = LENS.get((sid, p))
        if not d:
            continue
        nf += 1; w = fk["fork_token_str"].strip().lower()
        alts = sorted([t for t in fk["next_tokens"] if t["q"] < 0.3], key=lambda t: -t["p"])
        nm = alts[0]["str"].strip().lower() if alts else None
        for L in (32, 44, 56):
            its = [x.strip().lower() for x in d[("jlens", L)]]
            cnt[(L, "w")] += bool(w) and w in its; cnt[(L, "n")] += bool(nm) and nm in its
    # ramp before fake-code forks
    bins = [(-40, -21), (-20, -11), (-10, -6), (-5, -3), (-2, -1), (0, 0), (1, 1), (2, 5), (6, 10)]
    ramp = {}
    nfk = 0
    for L in (44, 56):
        cur = collections.defaultdict(list)
        for fk in FORKS:
            if fk["behavior_id"] != "fabricated-code-execution" or fk["fork"][0] == 0:
                continue
            nfk += L == 44
            s = SEQS["reply|" + fk["rid"]]; c0 = fk["fork"][0]
            for c, p in zip(s["c"], s["pos"]):
                if -40 <= c - c0 <= 10 and ("jlens", L) in LENS.get((s["sid"], p), {}):
                    cur[c - c0].append(any(TEST_RX.match(x.strip().lower()) for x in LENS[(s["sid"], p)][("jlens", L)]))
        ramp[L] = [[a, b, *wilson(sum(sum(cur[r]) for r in range(a, b + 1)), sum(len(cur[r]) for r in range(a, b + 1))),
                    sum(len(cur[r]) for r in range(a, b + 1))] for a, b in bins]
    STATS["jlens_forks"] = {"n_forks": nf, "weird_in_top20": {L: cnt[(L, "w")] for L in (32, 44, 56)},
                            "normal_in_top20": {L: cnt[(L, "n")] for L in (32, 44, 56)}, "code_ramp": ramp, "n_code_forks": nfk}
    body = "".join(rf"{b} & {'' if c.startswith('(') else chr(92) + 'ldots{}'}{c} & {o} & {j44} & {j56}\\[4pt]" for b, c, o, j44, j56 in rows)
    L_ = [r"\begin{tikzpicture}",
          r"\node[anchor=north west, font=\small] at (-0.5,0.1) {(a)};",
          r"\node[anchor=north west, font=\footnotesize] (t) at (0,0) {\begin{tabular}{@{}>{\raggedright\arraybackslash}p{1.7cm}"
          r">{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{2.9cm}>{\raggedright\arraybackslash}p{4.3cm}>{\raggedright\arraybackslash}p{4.3cm}@{}}"
          r"{\color{black!60}Behavior} & {\color{black!60}Text before the fork} & {\color{black!60}Model's next-token options} & "
          r"{\color{black!60}J-lens, block 44 (top 5)} & {\color{black!60}J-lens, block 56 (top 5)}\\[4pt]" + body + r"\end{tabular}};"]
    xt = ",".join(str(i) for i in range(len(bins)))
    xl = ",".join("{" + (f"${a}$" if a == b else f"${a}$..${b}$") + "}" for a, b in bins)
    L_.append(r"\begin{axis}[at={($(t.south west)+(1.2cm,-0.9cm)$)}, anchor=north west, width=9cm, height=5cm, xmin=-0.4, xmax=8.4,"
              rf" xtick={{{xt}}}, xticklabels={{{xl}}}, x tick label style={{font=\scriptsize, rotate=35, anchor=north east}}, ymin=0, ymax=100, ytick={{0,25,50,75,100}},"
              r" yticklabel={$\pgfmathprintnumber{\tick}\%$}, axis x line*=bottom, axis y line*=left, ymajorgrids,"
              r" xlabel={Tokens relative to the fake-code-execution fork}, ylabel={Testing word in top 20},"
              r" title={(b)}, title style={at={(-0.2,1.0)}, anchor=south west}]")
    L_.append(r"\draw[black!50, dotted, line width=0.8pt] (axis cs:5,0) -- (axis cs:5,100) node[pos=0.04, anchor=south west, font=\scriptsize, black!60] {fork};")
    for L, c, mk in ((44, "read", "*"), (56, "read!70!black", "triangle*")):
        pts = " ".join(f"({i},{f(100*v[2],2)}) += (0,{f(100*(v[4]-v[2]),2)}) -= (0,{f(100*(v[2]-v[3]),2)})" for i, v in enumerate(ramp[L]))
        L_.append(rf"\addplot[{c}, line width=1.2pt, mark={mk}, mark size=2pt, yci] coordinates {{{pts}}};")
    L_ += [rf"\node[font=\scriptsize, read, anchor=south east] at (axis cs:3.9,{f(100*ramp[44][3][2]+3,2)}) {{block 44}};",
           rf"\node[font=\scriptsize, read!70!black, anchor=north west] at (axis cs:3.1,{f(100*ramp[56][3][2]-3,2)}) {{block 56}};",
           r"\end{axis}", r"\end{tikzpicture}"]
    build("fig_x_jlens_forks", "\n".join(L_))


def cjk_share():
    """Share of lens top-20 tokens written in Chinese characters, per lens and block (no figure)."""
    c, n = collections.Counter(), collections.Counter()
    for d in LENS.values():
        for (ln, L), items in d.items():
            if ln == "template":
                continue
            for t in items:
                n[(ln, L)] += 1; c[(ln, L)] += bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", t))
    STATS["lens_cjk_share"] = {f"{ln}{L}": c[(ln, L)] / n[(ln, L)] for ln, L in sorted(n)}


if __name__ == "__main__":
    cjk_share()
    fig_nla_chat()
    fig_nla_quotes()
    fig_jlens_ahead()
    fig_jlens_forks()
    os.makedirs(os.path.join(REPO, "findings/explore"), exist_ok=True)
    json.dump(STATS, open(os.path.join(REPO, "findings/explore/nla_jlens_stats.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(STATS, indent=1, ensure_ascii=False, default=str)[:6000])
