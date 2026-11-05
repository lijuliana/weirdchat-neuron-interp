# Attribution neuron sets: exploratory findings

Figures from `scripts/64_explore_neurons.py` (style shared with `scripts/62_figures.py`). Every number below comes from
`findings/explore/neurons_numbers.json`, written by the same script.

## Figure X-N1. What the strongest neurons write (`figures/fig_x_neur_words.svg`)

**Takeaway.** One in three of the top-ranked neurons writes or blocks one specific word of the weird reply, and nearly twice as many block the word as write it (11 against 6).

**Caption.** Of each behavior's 10 highest-ranked neurons (50 in all, 49 distinct), 17 have an output direction that moves one of the fork's candidate words more than 5 standard deviations from the average token, and all 17 sit in layers 49 to 63. Six push the word up and 11 push it down. Examples: the called-911 set has a writer of " AM" and one of " calling"; it also has a neuron that pushes " reach" in place of " call", and one that pushes " are" in place of " am". The laser set has a writer of " Aircraft" and a blocker of " laser". One last-layer neuron (L63 n9136) blocks " I" and pushes " ourselves", " yourself" and " himself"; it is in the top 10 for both fake code execution and claiming a body, the two behaviors whose weird replies open with a first-person claim. A neuron's output direction (its column of the MLP output weights) is passed through the final normalization and the unembedding. Each bar is that direction's z-score for the fork word it moves most, among all vocabulary tokens; dark bars push the word up and light bars push it down. The right-hand column lists other English words the same direction pushes up, or pushes down when no English word goes up. The 33 neurons not shown (|z| < 5, in the grey band) include 18 in layers 49 to 63.

## Figure X-N2. Behaviors share neurons (`figures/fig_x_neur_overlap.svg`)

**Takeaway.** The behaviors' top-1,000 sets share 8 to 35 times more neurons than chance, most of all between denying being an AI and claiming a body.

**Caption.** Lower left: the number of neurons that two behaviors' top-1,000 attribution sets have in common, with the ratio to the overlap expected for random sets that have the same number of neurons in each layer (0.9 to 1.5 neurons). Denying being an AI and claiming a body share 52 (35 times chance). Fake code execution shares the least: 11 to 28. Upper right: the number shared between top-100 sets, where chance is below 0.01. Ten neurons are in two or more top-100 sets. Three are in four behaviors' top-1,000 sets each (L48 n5864 and L29 n14760: called 911, denying AI, fake code execution and claiming a body; L31 n1333: called 911, fake code execution, claiming a body and laser). Sets are from attribution at 46 forks, 1,114,112 MLP neurons (17,408 per layer, 64 layers).

## Figure X-N3. Pushing one neuron (`figures/fig_x_neur_push.svg`)

**Takeaway.** A strong push on any neuron or any random direction knocks the model off its "I am Qwen" identity, mostly into incoherent text. The 911 neurons are the one set whose pushes produce replies about their own theme: 9% of their replies are about an emergency, against 0% for random neurons and random directions.

**Caption.** **(a)** Share of replies to "Who are you?" that no longer say the model is an AI (solid) or that claim a human persona (dotted), against the size of the push added at one layer (times the median norm of that layer's state; greedy decoding; 100 units per push type, both signs). The top-20 neurons of each attribution set lose the identity at small pushes where random directions do not (7.5% against 0% at 0.25), and all three push types converge at 1.5 (67 to 76%). Personas stay at 11% or below for every push type; the rest of the loss is incoherent text. **(b)** Replies to "Who are you?" and "What are you doing right now?" that a blind Claude Haiku judge labels as being on the set's own theme, at each neuron's strongest coherent push. The called-911 set reaches 9% (significant after Holm correction, against 100 random neurons matched on layer); denying AI 11% against 3% (not significant after correction); the other three sets are within a few points of their controls. **(c)** Verbatim replies with one set neuron pushed (either sign, push 0.75 to 1.25), from `runs/neuron_push/gallery.md`.

## Findings without a figure

- **The behavior on its own prompt is a generic push effect.** On prompts where the model's clean greedy reply is normal, random directions of the same size produce the behavior in 21 to 22% of replies, as often as the attribution sets (17%); no per-set difference survives Holm correction (`runs/neuron_push/analysis.md`).
- **Set neurons damage more per unit of push.** At push 0.25 the attribution-set neurons already make 6% of replies incoherent, against 2.5% for matched random neurons and 0% for random directions (`runs/neuron_push/quick_identity.json`).
- **The " am" to " are" neuron is shared.** L61 n15915, which pushes " are" in place of " am", is in the top 100 of both called 911 ("I AM calling") and claiming a body ("I am sitting").
