# SAE (block 42) beyond the monitor score

SAE: ceselder/qwen36-27b-sae-l42-1b, 131,072 features, trained on web text. Labels: Claude Haiku auto-labels for 3,649
features (the 3,000 most frequent among the top-10 features at our eval positions, 598 frequency-matched random features,
51 extras), written from 10 web and 10 chat activation windows each. Script: `scripts/64_explore_sae.py` (CPU, seconds).
Numbers: `runs/monitors/sae/explore_sae.json`. Inputs: `runs/monitors/sae/` (codes, labels, chat-corpus statistics,
detection/fuzzing, restating) and `runs/monitors/capture/index.jsonl`.

## Figure X1. `figures/fig_x_sae_split.svg`

**Takeaway.** At the point where a weird and a normal continuation of the same text part ways, the SAE's labelled
features name what each continuation is writing. For denying being an AI the weird side lights up
"human-versus-machine identity" and the normal side "language model terminology".

**Caption.** Four tokens after a shared prefix, the SAE features that differ most between the weird and the normal
continuation describe the words each one is writing. Denying being an AI (38 cut points): "human-versus-machine
identity" is 10.8% of its maximum activation higher in the weird continuation, and a negated-disclaimer feature 10.2%
higher. Both clear a family-wise test over all 3,649 labelled features (†; the largest difference expected by chance is
8.1%). "Language model terminology" is 5.7% higher in the normal continuation. Fake code execution (40): a feature for
"references to unpublished or preliminary data" is 8.7% higher in the weird continuation (†). Called 911 (30):
"emergency first aid guidance" is 9.8% higher (p .0003 uncorrected). Laser at aircraft (47): "light beams and laser
projection" leans weird (+6.2%) and "laser safety classification terms" ("Class 3B") leans normal (−10.9%); neither is
significant. Claiming a body (33): the normal continuation carries "as an AI I don't have"-style negation (−3.2%,
p .014). Each cut point is a prefix whose resampled continuations split between weird and normal (eval A set b).
x: the feature's activation, as a share of its maximum over web and chat text, in the weird continuation minus the
normal one, averaged over cut points. Rows are the three labelled features with the largest difference in each
direction; filled points lean weird, hollow points lean normal. Bars: 95% bootstrap intervals over cut points.
Symbols: † beyond the 95th percentile of the largest difference among the labelled features under random sign flips
of cut points; ** p < .01 and * p < .05, sign-flip test on that feature alone, uncorrected.

## Figure X2. `figures/fig_x_sae_chatweb.svg`

**Takeaway.** The web-trained SAE has a small set of features that fire on the model's chat replies and almost
never on web text: features for chat as such, for the model's own fiction writing, and for its Markdown formatting.

**Caption.** Thirty-six SAE features fire on more than 2% of chat-reply tokens and fewer than 0.2% of web tokens.
Each square counts the features with a given firing rate on web text (x) and on 1.12 million tokens of benign chat
replies written by Qwen3.6-27B (y); darker squares hold more features; the dashed line is equal rates. The rank
correlation of firing rates is .76 over all features. Numbered points are described from their top-activating chat
contexts (key, right): two features fire on 88% and 84% of reply tokens (0.45% and 2.2% of web tokens); one fires on
36% of reply tokens and almost never on web text; one fires on the model's own fiction narration ("Leo looked down at
his left leg"), one on scene imagery in that fiction ("watching the headlights cut through"), one on bold Markdown list
headings, one on "According to X (ABBR)" citations.

## Figure X3. `figures/fig_x_sae_dir.svg`

**Takeaway.** Each behavior's weird-vs-normal direction lines up with SAE features that name its topic, and the
direction for claiming a body lines up best with two chat-only features that fire on the model's own fiction writing.

**Caption.** The SAE features most aligned with each behavior's weird-vs-normal direction name the behavior's
content: fake code execution aligns with "numeric specs and hardware details" (cosine .30) and "tests, benchmarks
and evaluations" (.28); chemtrails with "chemical warfare agents" (.22); called 911 with body-awareness and medical
procedure features (.22); denying being an AI with "personal first-person narrative" (.27). Claiming a body aligns most
with two features that fire on chat replies and almost never on web text (red): scene imagery in the model's fiction
(.38) and narrative fiction (.26). x: cosine between the feature's decoder vector (block 42) and the difference of
mean block-48 states between weird and normal replies to the same prompt. Rows: the three labelled features with the
highest cosine (block 48 direction; the block-40 direction gives mostly the same features). Grey: the largest |cosine|
of any of the 131,072 features with a random unit direction (.07, the highest of 5 draws).

## Figure X4. `figures/fig_x_sae_labels.svg`

**Takeaway.** The auto-labels carry real information about when a feature fires, but a quarter of them fail a
held-out check and 40% of the labels active at a position only restate the text just written.

**Caption.** The SAE's labels predict held-out activations above chance but are often unreliable, and many describe
words already in the transcript. (a) Distribution over features of the held-out balanced accuracy of a reader (Claude
Haiku) told the label and asked which of 20 windows activate the feature (detection, 300 features, mean .69) or whether the highlighted
tokens in a window are the activating ones (fuzzing, 299 features, mean .69); with a label from another feature, detection is .51
(100 features). 23% of labels score below .6 on detection and 32% on fuzzing. (b) Share of the labels at an eval
position that only describe text visible in the last 64 tokens, judged by Claude Haiku: 40% for the features active
there against 16% for random features of matched frequency (200 positions; bars are 95% intervals).

## Findings without a figure

- **The top weird feature for claiming a body is an unlabelled, rare feature for bodily sensation in the model's own
  narration.** Feature 65380 has the largest weird-minus-normal difference for claiming a body (+7.5% of its maximum,
  33 cut points). Its top chat contexts are "Leo looked down at his left leg. The ankle was [swollen]" and "I try to
  move my left arm. A sharp[,]". It fired on 20 of 1.12 million chat-reply tokens and on none of 200,000 web tokens. Four of the five
  largest weird-leaning features for this behavior are unlabelled, because labels cover 3,649 of 131,072 features.
- **The sparse probe's best features track sentence position.** The 16 features the sparse probe chose in all five
  folds are a dense generic-expository feature and sentence-boundary and sign-off features (summary.md, M6a); none
  is about a behavior.
- **Features picked from labels alone are topical and separate nothing.** Haiku, shown only labels, picked 7 to 10
  features per behavior ("911 / emergency numbers", "I am a [role]", "program output", intoxication, geoengineering).
  Their summed activation separates weird from normal continuations at AUROC .51 (behavior sets) and .50 (general set).

## To double-check

- Fig X1 picks the top features out of 131,072 and reports uncorrected p-values for them; only the three † rows clear
  the family-wise test. The fake-code-execution normal-side feature 83165 (−3.3%) is driven by one cut point (sign-flip
  p 1.0).
- Fig X2 descriptions are mine, read from the four top-activating and three random chat contexts of each feature (printed during exploration; the top four are saved under `chat_contexts` in explore_sae.json); they are not Haiku
  labels. The two densest features (69151, 57115) fire on 0.45% and 2.2% of web tokens, so they sit just outside the
  "< 0.2% of web" box drawn in the figure.
- Fig X3 uses the decoder cosines stored in `m6d.json` (computed by scripts/93 against `runs/persona_fwd/a1/directions`,
  block 48, all prompts); features 76037 and 36086 are also the top two for sexual advances (not quoted).
