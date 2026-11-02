"""neuron-push job keys, mirroring scripts/75_neuron_push_gen.py (phase gen): one generation per
(prompt, neuron or direction, signed push s = sgn * toward, alpha); units that share a neuron share its who / now jobs."""


def toward(u, signs):
    if u["group"] == "A":
        return signs[u["uid"]]["sign"]
    if u["group"] == "B":
        return (1 if u["d"] >= 0 else -1) if u["kind"] == "set" else u["rand_sign"]
    return 1


def job_key(u, pk, sgn, alpha, signs):
    s = sgn * toward(u, signs)
    who = "C" + str(u["dir_index"]) if u["group"] == "C" else "L%dN%d" % (u["layer"], u["neuron"])
    return f"{pk}|{who}|{s:+d}|{float(alpha)}"
