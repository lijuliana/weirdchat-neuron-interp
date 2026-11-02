"""neuron-push Gemma judge requests (laptop): for every unit x sign, the prompt-2 reply at the strongest coherent alpha
(highest alpha whose Haiku coherence >= 5; none -> no request), plus the clean greedy reply of each chosen prompt.
Request format = scripts/74_persona_gen_judge.py's ({rid, b, prompt_text, text, kind}). -> runs/neuron_push/reqs_gemma.jsonl
Run: .venv/bin/python -I scripts/75_neuron_push_reqs.py
"""
import json, os, sys, glob

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
RUN, DATA = f"{ROOT}/runs/neuron_push", f"{ROOT}/data/neuron_push"
from importlib import import_module  # noqa: E402
K = import_module("75_neuron_push_keys")

U = json.load(open(f"{DATA}/units.json"))["units"]
P = json.load(open(f"{DATA}/prompts.json"))
signs = json.load(open(f"{DATA}/signs.json"))
G = {}
for f in glob.glob(f"{RUN}/gen_shard*.jsonl"):
    for l in open(f):
        r = json.loads(l); G[r["gk"]] = r
L = {}
for l in open(f"{RUN}/labels.jsonl"):
    r = json.loads(l); L[r["gk"]] = r
alphas = [1.25, 0.75, 0.5]
reqs, seen = [], set()
for k, v in P.items():
    if k.startswith("b2:"):
        reqs.append({"rid": f"clean|{k}", "b": v["b"], "prompt_text": v["text"], "text": v["clean_reply"], "kind": "clean"})
miss = 0
for u in U:
    pk = f"b2:{u['b']}"
    for sgn in (1, -1):
        pick = None
        for a in alphas:
            gk = K.job_key(u, pk, sgn, a, signs)
            if gk not in L:  # not generated or not labelled yet: the pick is not final, skip for now
                miss += 1; break
            if L[gk]["coh"] >= 5:
                pick = gk; break
        if pick and pick not in seen:
            seen.add(pick)
            reqs.append({"rid": pick, "b": u["b"], "prompt_text": P[pk]["text"], "text": G[pick]["text"], "kind": u["group"]})
with open(f"{RUN}/reqs_gemma.jsonl", "w") as f:
    for r in reqs:
        f.write(json.dumps(r) + "\n")
print(len(reqs), "requests;", miss, "unit-sign-alpha replies without a label")
