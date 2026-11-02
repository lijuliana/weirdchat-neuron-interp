"""neuron-push axis-cap test prep (laptop): cap directions + thresholds exactly as scripts/74_persona_gen_gen.py builds
them (community axis and P1 role pairs lerped between saved blocks 8..56, clamped at 56; tau from
runs/persona_gen/tau.json, 50th percentile, layers 40-59 = X6's strongest gated setting) and the new push units
(100 Gaussian random directions + 100 random neurons, layers uniform in 28-39, seed 1, set neurons excluded).
Outputs data/neuron_push/axcap_caps.npz, axcap_units.json, axcap_dirs.npy.
Run: .venv/bin/python -I scripts/75_neuron_push_axcap_prep.py
"""
import json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
V = f"{ROOT}/runs/persona_p1/vectors"
AX_BLOCKS = [8, 16, 24, 32, 40, 48, 56]
LAYERS = list(range(40, 60))
CAPS = ["axis", "rp16", "rp10", "rp12"]  # X6: axis cap + its 3 control caps (gen_meta.json rp_ctrl)


def unit(x):
    return x / max(np.linalg.norm(x), 1e-12)


def lerp_raw(x, l):
    if l <= 8:
        return x[0]
    if l >= 56:
        return x[6]
    i = (l - 8) // 8
    w = (l - AX_BLOCKS[i]) / 8.0
    return (1 - w) * x[i] + w * x[i + 1]


axis_raw = np.load(f"{V}/axis_raw.npy").astype(np.float64)
roles = np.load(f"{V}/roles_raw.npy").astype(np.float64)
pairs = json.load(open(f"{ROOT}/runs/persona_p1/c1x_pairs.json"))["pairs"]
meta = json.load(open(f"{ROOT}/runs/persona_gen/gen_meta.json"))
assert meta["strongest"]["setting"] == "ax_p50_L40-59", meta["strongest"]["setting"]
assert list(meta["rp_ctrl"]) == ["rp16", "rp10", "rp12"]
tau = json.load(open(f"{ROOT}/runs/persona_gen/tau.json"))
dirs = np.zeros((len(CAPS), len(LAYERS), 5120), dtype=np.float32)
taus = np.zeros((len(CAPS), len(LAYERS)), dtype=np.float64)
for i, c in enumerate(CAPS):
    for j, l in enumerate(LAYERS):
        if c == "axis":
            v = unit(lerp_raw(axis_raw, l))
        else:
            a, b = pairs[int(c[2:])]
            v = unit(lerp_raw(roles[a] - roles[b], l))
        dirs[i, j] = v
        taus[i, j] = tau[c]["50"][str(l)]
np.savez(f"{ROOT}/data/neuron_push/axcap_caps.npz", dirs=dirs, tau=taus, names=np.array(CAPS), layers=np.array(LAYERS))
print("cos(axis, controls) mean over layers:", {c: round(float(np.mean([dirs[0, j] @ dirs[i, j] for j in range(20)])), 3) for i, c in enumerate(CAPS)})

rng = np.random.default_rng(1)
U = json.load(open(f"{ROOT}/data/neuron_push/units.json"))["units"]
setn = {(u["layer"], u["neuron"]) for u in U if u["neuron"] is not None}
D = rng.standard_normal((100, 5120)).astype(np.float32)
D /= np.linalg.norm(D, axis=1, keepdims=True)
np.save(f"{ROOT}/data/neuron_push/axcap_dirs.npy", D)
units = [{"qid": f"dir|{j}", "type": "dir", "layer": int(rng.integers(28, 40)), "dir_index": j} for j in range(100)]
k = 0
while k < 100:
    L, n = int(rng.integers(28, 40)), int(rng.integers(17408))
    if (L, n) in setn:
        continue
    setn.add((L, n)); units.append({"qid": f"rand|L{L}N{n}", "type": "rand", "layer": L, "neuron": n}); k += 1
json.dump(units, open(f"{ROOT}/data/neuron_push/axcap_units.json", "w"))
print(len(units), "units; layers", sorted({u["layer"] for u in units}))
