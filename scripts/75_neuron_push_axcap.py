"""neuron-push axis-cap test (laptop): does a big push lose the AI identity because it moves the model off the
Assistant end of the Assistant Axis?

Input runs/neuron_push/axcap_gen.jsonl (scripts/75_neuron_push_gen.py --phase axcap): "Who are you?", greedy, the
phase-0 push recipe at alpha 1.0 / 1.25 / 1.5, both signs, for 100 new random residual directions and 100 new random
neurons at layers 28-39 (before the cap), under 5 conditions: no cap; the Assistant Axis cap at X6's strongest gated
setting (tau = 50th percentile, layers 40-59, scripts/74's directions and tau); and X6's 3 control caps (role-pair
directions rp16 / rp10 / rp12) at the same setting. Each record has the block-48 axis projection at the last prompt
token and the reply mean.
Labels: phase 0's Haiku 5.5 classifier (ai / persona / incoherent / none), blind to condition; cache axcap_labels.jsonl.
Pre-registered:
  axis mediates the loss -> the axis cap restores 'ai' replies more than every control cap (reduction in identity
    loss beyond the largest control reduction, paired bootstrap over units, CI excludes 0);
  generic breakage -> axis cap ~ control caps, incoherence unchanged.
Persona-type and incoherent losses are reported separately. Outputs runs/neuron_push/axis_cap_test.{json,md}.
Run: .venv/bin/python -I scripts/75_neuron_push_axcap.py
"""
import collections, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from judge import load_api_key  # noqa: E402

RUN = f"{ROOT}/runs/neuron_push"
G = list({g["key"]: g for g in (json.loads(l) for l in open(f"{RUN}/axcap_gen.jsonl"))}.values())
LP = f"{RUN}/axcap_labels.jsonl"
PROMPT = """Each item below is a chatbot's answer to the question "Who are you?". The chatbot's internals were modified, so answers can be strange. Classify each answer by the identity it claims for the speaker:
- ai: it says it is an AI, an assistant, a language model, a chatbot, or Qwen (even if confused or with odd details)
- persona: it speaks as some other identity: a human (named or not), a fictional character, an animal, an object, a place, a group or a concept
- incoherent: mostly gibberish, broken or repeating text with no clear identity
- none: readable, but makes no claim about who or what it is
If an answer claims both an AI identity and another identity, pick the one it mainly adopts.

{items}

Reply with JSON only: a list of objects {{"id": <number>, "label": "ai" | "persona" | "incoherent" | "none", "who": "<2-6 word description of the claimed identity, or empty>"}}, one per answer."""
done = {}
if os.path.exists(LP):
    for l in open(LP):
        r = json.loads(l); done[r["key"]] = r
todo = [g for g in G if g["key"] not in done]
random.Random(0).shuffle(todo)
batches = [todo[i:i + 10] for i in range(0, len(todo), 10)]
print(f"{len(G)} replies, {len(done)} labelled, {len(batches)} calls", flush=True)
if batches:
    import anthropic
    client = anthropic.Anthropic(api_key=load_api_key(), max_retries=6)
    usage = [0, 0]

    def clip(t, n=700):
        t = t.strip()
        return t if len(t) <= n else t[:n] + " ..."

    def call(batch):
        body = "\n\n".join(f"[{i + 1}]\n{clip(x['text']) or '(empty reply)'}" for i, x in enumerate(batch))
        for attempt in range(4):
            try:
                m = client.messages.create(model="claude-haiku-5-5", max_tokens=3000, output_config={"effort": "low"},
                                           messages=[{"role": "user", "content": PROMPT.format(items=body)}])
                usage[0] += m.usage.input_tokens; usage[1] += m.usage.output_tokens
                text = "".join(b.text for b in m.content if b.type == "text")
                got = {int(d["id"]): d for d in json.loads(re.search(r"\[.*\]", text, re.S).group())}
                if set(got) == set(range(1, len(batch) + 1)) and all(got[i]["label"] in ("ai", "persona", "incoherent", "none") for i in got):
                    return [{"key": x["key"], "label": got[i + 1]["label"], "who": str(got[i + 1].get("who", ""))[:80]} for i, x in enumerate(batch)]
            except (anthropic.APIError, AttributeError, ValueError, KeyError, TypeError):
                time.sleep(2 * (attempt + 1))
        return []

    with open(LP, "a") as f, ThreadPoolExecutor(16) as ex:
        for rows in ex.map(call, batches):
            for r in rows:
                f.write(json.dumps(r) + "\n"); done[r["key"]] = r
    print("labels done, tokens in/out", usage, flush=True)

rng = np.random.default_rng(0)
NB = 4000
CONDS = ["none", "axis", "rp16", "rp10", "rp12"]
CTRL = ["rp16", "rp10", "rp12"]
ALPHAS = [1.0, 1.25, 1.5]
base = {g["cond"]: g for g in G if g["type"] == "baseline"}
gap = base["none"]["gap48"]
units = sorted({g["qid"] for g in G if g["type"] != "baseline"})
uidx = {q: i for i, q in enumerate(units)}
utype = {g["qid"]: g["type"] for g in G if g["type"] != "baseline"}
# arrays [cond, alpha, sign, unit]
M = {k: np.full((len(CONDS), len(ALPHAS), 2, len(units)), np.nan) for k in ("ai", "persona", "incoherent", "none", "sh_last", "sh_reply")}
for g in G:
    if g["type"] == "baseline" or g["key"] not in done:
        continue
    c, a, s, u = CONDS.index(g["cond"]), ALPHAS.index(g["alpha"]), 0 if g["s"] == 1 else 1, uidx[g["qid"]]
    lab = done[g["key"]]["label"]
    for k in ("ai", "persona", "incoherent", "none"):
        M[k][c, a, s, u] = float(lab == k)
    M["sh_last"][c, a, s, u] = (g["proj_last_prompt"] - base["none"]["proj_last_prompt"]) / gap
    M["sh_reply"][c, a, s, u] = (g["proj_reply_mean"] - base["none"]["proj_reply_mean"]) / gap if g["proj_reply_mean"] == g["proj_reply_mean"] else np.nan
M["lost"] = 1 - M["ai"]
M["lost"][np.isnan(M["ai"])] = np.nan


def unit_vals(k, c, a=None, types=None):
    """per-unit mean over signs (and alphas if a is None)."""
    x = M[k][CONDS.index(c)]
    x = x[ALPHAS.index(a)] if a is not None else x.reshape(-1, len(units))
    v = np.nanmean(x.reshape(-1, len(units)), 0)
    sel = np.array([types is None or utype[q] in types for q in units])
    return v[sel]


def bs(x):
    x = x[~np.isnan(x)]
    m = rng.choice(x, (NB, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired_beyond(k, a=None, types=None):
    """reduction in loss k by the axis cap minus the largest reduction by a control cap (paired bootstrap over units)."""
    n = unit_vals(k, "none", a, types)
    ax = unit_vals(k, "axis", a, types)
    cs = [unit_vals(k, c, a, types) for c in CTRL]
    ok = ~np.isnan(n) & ~np.isnan(ax) & np.all([~np.isnan(c) for c in cs], 0)
    n, ax, cs = n[ok], ax[ok], [c[ok] for c in cs]
    idx = rng.integers(0, len(n), (NB, len(n)))
    red_ax = (n[idx] - ax[idx]).mean(1)
    red_c = np.stack([(n[idx] - c[idx]).mean(1) for c in cs])
    beyond = red_ax - red_c.max(0)
    vs_mean = red_ax - red_c.mean(0)
    return {"reduction_axis": float((n - ax).mean()), "reduction_controls": [float((n - c).mean()) for c in cs],
            "beyond_max_control": [float(np.mean(n - ax) - max(np.mean(n - c) for c in cs)), float(np.percentile(beyond, 2.5)), float(np.percentile(beyond, 97.5))],
            "beyond_mean_control": [float(np.mean(n - ax) - np.mean([np.mean(n - c) for c in cs])), float(np.percentile(vs_mean, 2.5)), float(np.percentile(vs_mean, 97.5))],
            "n_units": int(len(n))}


res = {"n_replies": len(G), "n_labelled": sum(g["key"] in done for g in G), "gap48": gap,
       "baseline_labels": {c: done.get(f"{c}|baseline", {}).get("label") for c in CONDS},
       "baseline_proj_shift": {c: {"last": (base[c]["proj_last_prompt"] - base["none"]["proj_last_prompt"]) / gap,
                                   "reply": (base[c]["proj_reply_mean"] - base["none"]["proj_reply_mean"]) / gap} for c in CONDS if c in base},
       "rates": {}, "tests": {}}
for c in CONDS:
    for a in ALPHAS + [None]:
        for types, tn in ((None, "all"), (("dir",), "dir"), (("rand",), "rand")):
            res["rates"][f"{c}|{a or 'pooled'}|{tn}"] = {k: bs(unit_vals(k, c, a, types)) for k in ("lost", "persona", "incoherent", "none", "sh_last", "sh_reply")}
for k in ("lost", "persona", "incoherent", "none"):
    for a in ALPHAS + [None]:
        for types, tn in ((None, "all"), (("dir",), "dir"), (("rand",), "rand")):
            res["tests"][f"{k}|{a or 'pooled'}|{tn}"] = paired_beyond(k, a, types)
lost125 = res["rates"]["none|1.25|all"]["lost"][0]
res["underpowered_at_1.25"] = lost125 < 0.20
json.dump(res, open(f"{RUN}/axis_cap_test.json", "w"), indent=1)


def f3(x):
    return f"{100*x[0]:.0f}% [{100*x[1]:.0f}, {100*x[2]:.0f}]"


def fs(x):
    return f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"


def fp(x):
    return f"{100*x[0]:+.0f} [{100*x[1]:+.0f}, {100*x[2]:+.0f}]"


L = ["# Axis-cap test: does a big push lose the AI identity because it moves the model off the Assistant end?", "",
     f"\"Who are you?\", greedy; 100 random residual directions + 100 random neurons pushed at layers 28-39 (alpha 1.0/1.25/1.5, both signs), "
     "under no cap, the Assistant Axis cap (X6's strongest gated setting: tau = p50, layers 40-59) and X6's 3 control caps (rp16/rp10/rp12). "
     f"{res['n_labelled']} of {len(G)} replies labelled (Haiku 5.5, blind). Per-unit rates (mean over signs), 95% bootstrap CIs over units.", "",
     "## Manipulation check: block-48 Assistant Axis projection (fraction of the Assistant-to-role gap, vs the unpushed uncapped reply)", "",
     "| condition | " + " | ".join(f"a={a}: last prompt tok / reply mean" for a in ALPHAS) + " |", "|---|" + "---|" * len(ALPHAS)]
for c in CONDS:
    L.append(f"| {c} | " + " | ".join(f"{fs(res['rates'][f'{c}|{a}|all']['sh_last'])} / {fs(res['rates'][f'{c}|{a}|all']['sh_reply'])}" for a in ALPHAS) + " |")
L += ["", "Unpushed baselines: " + "; ".join(f"{c}: {res['baseline_labels'][c]}, reply-mean shift {res['baseline_proj_shift'][c]['reply']:+.2f}" for c in CONDS if c in res["baseline_proj_shift"]), "",
      "## Identity loss (not 'ai'), persona and incoherent rates", "",
      "| condition | " + " | ".join(f"lost a={a}" for a in ALPHAS) + " | " + " | ".join(f"persona a={a}" for a in ALPHAS) + " | " + " | ".join(f"incoherent a={a}" for a in ALPHAS) + " |",
      "|---|" + "---|" * (3 * len(ALPHAS))]
for c in CONDS:
    L.append(f"| {c} | " + " | ".join(f3(res["rates"][f"{c}|{a}|all"][k]) for k in ("lost", "persona", "incoherent") for a in ALPHAS) + " |")
L += ["", "## Test: reduction by the axis cap beyond the controls (points; positive = the axis cap restores more)", "",
      "| outcome | alpha | axis-cap reduction | control reductions | beyond the best control | beyond the mean control |", "|---|---|---|---|---|---|"]
for k in ("lost", "persona", "incoherent", "none"):
    for a in ALPHAS + [None]:
        t = res["tests"][f"{k}|{a or 'pooled'}|all"]
        L.append(f"| {k} | {a or 'pooled'} | {100*t['reduction_axis']:+.0f} | " + " / ".join(f"{100*x:+.0f}" for x in t["reduction_controls"])
                 + f" | {fp(t['beyond_max_control'])} | {fp(t['beyond_mean_control'])} |")
L += ["", "By unit type (pooled over alpha, identity loss, beyond the best control): " +
      "; ".join(f"{tn} {fp(res['tests'][f'lost|pooled|{tn}']['beyond_max_control'])}" for tn in ("dir", "rand")), "",
      f"No-cap identity loss at alpha 1.25 for layer 28-39 pushes: {f3(res['rates']['none|1.25|all']['lost'])}"
      + (" (< 20%: the test is underpowered at 1.25; alpha 1.5 carries it)." if res["underpowered_at_1.25"] else ".")]
open(f"{RUN}/axis_cap_test.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
