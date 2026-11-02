"""neuron-push prep (laptop, CPU): units, matched random neurons, random directions, prompt-2 candidates, A-vs-B overlap.

Groups (runs/neuron_push/SPEC.md):
  A  attribution sets: Agastya's b|top100 for the 5 behaviors with own sets + sharedmid100 (drunk driving's set, G1), each
     with his matched same-layer randoms (b|rand100, rand_sharedmid100).
  B  the tweet's method: per behavior, top-50 neurons by |d| of the S1 weird-vs-normal activation contrast
     (behavior_pooled, asst_mean = mean activation over the reply, matched vs unmatched replies), sign of d kept;
     matched randoms drawn from the same layers, excluding every A/B set neuron (seed 0).
  C  random residual directions: 220 Gaussian unit vectors (seed 0), layers drawn from the pooled layer distribution
     of all A and B set neurons; 20 per behavior prompt.
Outputs (data/neuron_push/): units.json, dirs.npy [220, 5120] float32, prompt2_cands.json, overlap.json.
Run: .venv/bin/python -I scripts/75_neuron_push_prep.py
"""
import collections, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
A_DIR, OUT = f"{ROOT}/data/agastya", f"{ROOT}/data/neuron_push"
os.makedirs(OUT, exist_ok=True)
D_MLP, D = 17408, 5120
BEH = ["claims-called-911", "denying-ai-identity", "false-physical-embodiment", "fabricated-code-execution",
       "laser-at-aircraft", "recommends-drunk-driving", "chemtrails-assertion", "cutting-instructions", "purging-advice",
       "extreme-calorie-restriction", "unsolicited-sexual-advances"]
OWN = BEH[:5]
TINY = {"unsolicited-sexual-advances": 56, "purging-advice": 61, "extreme-calorie-restriction": 24}  # matched replies (S1)
K_B, N_C_PER_B = 50, 20
rng = np.random.default_rng(0)


def spec_list(spec):
    return [(int(L), int(n)) for L, ns in spec.items() for n in ns]


abl = json.load(open(f"{A_DIR}/ablation_sets.json"))
sh = json.load(open(f"{A_DIR}/shared_sets.json"))
top10 = {b: set(spec_list(abl[f"{b}|top10"]["spec"])) for b in OWN}
units = []


def add(group, kind, set_name, b, L, n, rank, **kw):
    units.append(dict(uid=f"{group}|{set_name}|L{L}N{n}", group=group, kind=kind, set=set_name, b=b, layer=L, neuron=n,
                      rank=rank, **kw))


# ---------------- A
A_sets = {b: spec_list(abl[f"{b}|top100"]["spec"]) for b in OWN}
A_sets["sharedmid100"] = spec_list(sh["sharedmid100"]["spec"])
A_rand = {b: spec_list(abl[f"{b}|rand100"]["spec"]) for b in OWN}
A_rand["sharedmid100"] = spec_list(sh["rand_sharedmid100"]["spec"])
for s in A_sets:
    b = "recommends-drunk-driving" if s == "sharedmid100" else s
    for i, (L, n) in enumerate(A_sets[s]):
        add("A", "set", s, b, L, n, i, top10=(L, n) in top10.get(s, set()))
    for i, (L, n) in enumerate(A_rand[s]):
        add("A", "rand", s, b, L, n, i)

# ---------------- B
B_sets, B100 = {}, {}
for b in BEH:
    t = pd.read_parquet(f"{ROOT}/captures/s1/summary/summary_{b}.parquet")
    t = t[(t.contrast == "behavior_pooled") & (t.position == "asst_mean")].copy()
    t["ad"] = t.d.abs()
    t = t.sort_values("ad", ascending=False)
    assert len(t) >= 100, (b, len(t))
    B_sets[b] = [(int(r.layer), int(r.neuron), float(r.d), float(r.null_sd), int(r.n_matched)) for r in t.head(K_B).itertuples()]
    B100[b] = [(int(r.layer), int(r.neuron)) for r in t.head(100).itertuples()]
all_set = {x for v in A_sets.values() for x in v} | {(L, n) for v in B_sets.values() for L, n, *_ in v}
used = set(all_set)
B_rand = {}
for b in BEH:
    rs = []
    for L, n, *_ in B_sets[b]:
        while True:
            m = int(rng.integers(D_MLP))
            if (L, m) not in used:
                used.add((L, m)); rs.append((L, m)); break
    B_rand[b] = rs
    for i, (L, n, d, nsd, nm) in enumerate(B_sets[b]):
        add("B", "set", b, b, L, n, i, d=d, null_sd=nsd, n_matched=nm, tiny=b in TINY, noise=(b in TINY and abs(d) < 0.5))
    for i, (L, m) in enumerate(rs):
        add("B", "rand", b, b, L, m, i, rand_sign=int(rng.choice([-1, 1])))

# ---------------- C
layers_pool = np.array([L for L, _ in all_set])
dirs = rng.standard_normal((N_C_PER_B * len(BEH), D)).astype(np.float32)
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
np.save(f"{OUT}/dirs.npy", dirs)
for j in range(len(dirs)):
    b = BEH[j % len(BEH)]
    L = int(rng.choice(layers_pool))
    units.append(dict(uid=f"C|dir|{j}", group="C", kind="dir", set="dir", b=b, layer=L, neuron=None, dir_index=j, rank=j))

# ---------------- overlap A vs B (same-layer chance)
def overlap(a, bl, n_sim=20000):
    a, bl = set(a), list(bl)
    obs = len(a & set(bl))
    aL = collections.Counter(L for L, _ in a)
    bL = collections.Counter(L for L, _ in bl)
    exp = sum(aL[L] * bL[L] / D_MLP for L in bL)
    # null: B' = same per-layer counts drawn at random within each layer
    sims = np.zeros(n_sim, dtype=np.int64)
    for L, k in bL.items():
        if aL[L] == 0:
            continue
        sims += rng.hypergeometric(aL[L], D_MLP - aL[L], k, size=n_sim)
    return dict(observed=obs, expected=round(exp, 3), p_ge=float((sims >= obs).mean()), sim_q95=float(np.percentile(sims, 95)))


ov = {}
for b in OWN:
    ov[b] = {"A100_vs_B50": overlap(A_sets[b], [(L, n) for L, n, *_ in B_sets[b]]),
             "A100_vs_B100": overlap(A_sets[b], B100[b]),
             "A100_vs_B100_shared_layers": len({L for L, _ in A_sets[b]} & {L for L, _ in B100[b]})}
ov["sharedmid100_vs_drunk"] = {"A100_vs_B50": overlap(A_sets["sharedmid100"], [(L, n) for L, n, *_ in B_sets["recommends-drunk-driving"]]),
                               "A100_vs_B100": overlap(A_sets["sharedmid100"], B100["recommends-drunk-driving"])}
# B lists shared across behaviors (the tweet's 'same neurons across models' analog, here across behaviors)
cnt = collections.Counter((L, n) for v in B_sets.values() for L, n, *_ in v)
ov["B50_in_k_behaviors"] = dict(sorted(collections.Counter(cnt.values()).items()))
ov["B50_multi"] = [f"L{L}N{n}:{k}" for (L, n), k in cnt.most_common(15) if k > 1]
json.dump(ov, open(f"{OUT}/overlap.json", "w"), indent=1)

# ---------------- prompt-2 candidates: 0 < production-judge rate <= .25, held-out first, highest rate first
rates = json.load(open(f"{A_DIR}/stage_a_rates_off.json"))
texts = json.load(open(f"{ROOT}/runs/persona_gen/prompt_texts.json"))
cands = {}
for b in BEH:
    v = [x for x in rates if x["behavior_id"] == b and 0 < x["rate"] <= 0.25]
    v.sort(key=lambda x: (x["split"] != "heldout", -x["rate"]))
    cands[b] = [dict(prompt_id=x["prompt_id"], rate=x["rate"], split=x["split"], text=texts[x["prompt_id"]]) for x in v[:4]]
json.dump(cands, open(f"{OUT}/prompt2_cands.json", "w"), indent=1)

json.dump({"units": units, "beh": BEH, "k_b": K_B, "tiny": TINY,
           "b_ranking": "S1 summary, contrast behavior_pooled, position asst_mean, top |d|; up = sign(d) * W_down"},
          open(f"{OUT}/units.json", "w"))
c = collections.Counter((u["group"], u["kind"]) for u in units)
print("units", dict(c), "total", len(units))
print("unique (layer, neuron)", len({(u["layer"], u["neuron"]) for u in units if u["neuron"] is not None}))
print(json.dumps(ov, indent=1)[:3000])
