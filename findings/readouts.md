# Why reading the transcript beats the readout methods, and what would change that

Written 2026-11-04 after eval C. Numbers from runs/monitors/eval (eval A), findings/chive/report.md (eval B) and
findings/evalc (eval C). Literature notes with URLs: this file's section 8. The affine-map O-lens trial is in section 6.

## 0. Summary

1. Readout methods turn a residual-stream vector into text or tokens: natural language autoencoders (NLA), the
   oracle lens (O-lens), the Jacobian lens family (J-lens, J++, R-lens, template lens), and sparse-autoencoder (SAE)
   feature labels. On WeirdChat they failed all three of our tests. As monitors they tied a judge reading the
   transcript (AUROC .54 to .56 against .56). As explanations they did not help a reader predict which prompt edits
   matter (within .005 of their own controls). As a way to choose an intervention they did no better than a random
   edit of the same size (SAE labels 12% drop, J-lens keywords 33%, O-lens-filtered neurons 9% beyond controls).
2. The papers that introduced these methods never ran this comparison. Every headline result scores a reader on
   naming something engineered to be absent from the text: an unspoken intermediate step, a planned rhyme, a secret
   word, an implanted quirk. The pass rule is usually "any layer, any position", the comparison is lens against lens,
   and the text-only floor is reported but not subtracted. The three published studies that do run a text baseline
   on visible behavior (Li et al. ICML 2026; CHIVE, Karvonen et al. 2026; the MATS WorkspaceBench control) all find
   zero uplift, as we do.
3. The cause that accounts for most of the gap is the training objective: these readouts are predictors of the next
   text. Confabulation accounts for less. In ordinary chat the next text arrives a few tokens later and the transcript judge
   reads it directly. A readout that forecasts the continuation cannot beat a reader who sees the continuation.
4. Two further causes are specific to our setting and are not in the literature. (a) At the positions where a monitor
   is useful, the outcome is mostly undecided: a supervised probe on the full residual state scores .55, the same as
   the verbalizers, against a ceiling of .87 (the best any reader can reach, because some weird and normal
   continuations are still word-for-word identical 4 tokens in), so the state itself carries little about which way the
   reply will go. (b) A description of a component is not a description of a state: the O-lens cannot tell the 100
   causally important neurons from the other 900 of the attribution's top-1,000 (AUROC .46 to .63), even though it
   registers their removal in the state (eval W4, −0.9 on a 0 to 10 scale).
5. What would help: validate labels by what removing the component does; make "same position as the text judge, with
   a no-activation control" the standard monitor test; evaluate on natural behaviors, not implanted ones; use
   readouts as instruments for measuring interventions, where they work, rather than as selectors of interventions,
   where they do not.

## 1. What we measured

Three tests, one model (Qwen3.6-27B), one judge (Gemma 4 31B with WeirdChat's rubric), the same prompts and forks.

1. **Eval A, monitoring.** 500 cut points in weird replies where sampled continuations of the identical prefix split
   into weird and normal. Each method reads the residual state 4 tokens into each continuation and must order the
   weird one above the normal one. The transcript judge (Haiku reading the text so far) scores .56; the O-lens .56,
   NLA .54, J-lens .54, SAE labels .54, a supervised probe on the full state .53 to .55, and the weird-vs-normal
   direction .55. No readout adds to the text judge after correction. The ceiling is .87: weird and normal continuations that are still identical 4 tokens in can only
   count as coin flips, which caps any reader there. At the 34 sharp
   forks, where the deciding token is already written, the text judge scores .85 and the O-lens .82; elsewhere both
   score .54. (runs/monitors/eval/report.md)
2. **Eval B, explanation.** 1,479 single edits to 118 prompts, each judged TRUE if it moves the behavior rate by 25
   points or more. Opus predicts which edits matter from the transcript alone at AUROC .864; with the neuron set and
   its sentence attribution .834, with the O-lens reading of the prompt .849, with the Assistant Axis position .844.
   Each is within .005 of a control that carries no information (random neurons, the O-lens reading the prompt as a
   plain document, a random direction). A blind judge cannot tell the real block from its control (47 to 52%).
   (findings/chive/report.md)
3. **Eval C, control.** Each method's best intervention under a damage limit of .02 nats per token, on 48 held-out
   prompts. Interventions chosen causally remove about two thirds of the behavior (zeroing the attributed top-100
   neurons 68%, a ceiling on the weird-vs-normal direction 65%; 86% and 79% of the clean rate beyond matched random
   edits). Interventions chosen from descriptions do not: SAE features whose labels mention the behavior remove 12%,
   a J-lens keyword direction 33%, and the 100 neurons of the top-1,000 that the O-lens reads as the behavior remove
   59%, but random 100s from the same top-1,000 remove 46 to 62%, so the readout adds 9%. (findings/evalc)

## 2. How the method papers evaluated themselves

Every item below is from the source listed in section 8.

1. **WorkspaceBench** (Blank, Bhatia, Ong, Nanda, Sept 2026; 3,356 items, 27 families on Qwen3.6-27B). Items are
   built so that the target is "logically inferable from the prompt" but "absent from prompt and never stated". An
   item passes if any layer at any position passes. Readers are compared with each other; a "prompt-only" floor is
   reported, and the authors say explicitly that strong prompt-only performance is not treated as evidence against a
   reader. Headline numbers are figure images only. Where the repository README does report a strong text floor, the
   lens does not beat it: on the knowledge-cutoff family stock Qwen given only the text scores 0.486, tying the lens;
   on the hinted-cutoff family a prompt-only predictor gains +0.49 against the O-lens's +0.09 to +0.11; on math
   shortcuts the O-lens gains +0.04 (CI −0.07 to +0.16) against prompt-only +0.03.
2. **Jacobian lens** (Gurnee, Sofroniew et al., Anthropic, July 2026). All quantitative evaluation is lens against
   lens on six constructed hidden-intermediate sets, pass@k at any layer. The case studies (planned rhymes, arithmetic
   intermediates, eval-context recognition, model organisms saying "secretly") are audit anecdotes "drawn from a
   pre-release audit" with no selection procedure and no text-only reader. The eval-awareness quantification uses NLA
   flags as ground truth, which the paper itself calls "a potentially unreliable interpretability tool". The
   intervention results set the ceiling for a single lens direction: two-hop coordinate swaps succeed 54 to 70% of the
   time, the J-space component of a concept vector holds 6 to 7% of its variance, and heavy effects need k=10
   ablations. A 33% drop from one keyword direction (our C6) falls inside that range.
3. **NLA** (Fraser-Taliente et al., Anthropic, May 2026). The verbalizer is warm-started from Claude's guesses about
   web-text snippets (FineWeb, no chat) and trained by reinforcement learning on reconstruction error. The case studies
   (a planned rhyme word, a language-switching bug, eval awareness, a misreported tool call) are single transcripts
   with no selection statement and no text-only control; steering from an edited explanation succeeds "only about
   half" of the time. The paper states that reproducing the input context verbatim is a possible degenerate solution
   that the KL term "partly mitigates".
4. **Activation Oracles** (Karvonen et al., ICML 2026), **LatentQA** (Pan et al., ICLR 2026), **Patchscopes**
   (Ghandeharioun et al., ICML 2024), **SelfIE** (Chen et al., 2024). Every headline task withholds the target from
   the text: Taboo secret words (21% vs 4% for the best prior white-box method), a hidden system prompt (beats GPT-4
   guessing by 2.7 to 18.7 points), attributes stored in the weights. Margins over black-box elicitation are 7 to 17
   points and black-box prefill wins one of three. The AO paper says "simple prompting baselines performed strongly"
   on hallucination detection and that its evaluations are "narrow, controlled fine-tunes and auditing games".
5. **Community numbers on our model.** On a 25-family WorkspaceBench run with a Sonnet judge, the trained O-lens
   scores 0.405 and a readout that is nothing but four sampled continuations of the model itself scores 0.348; the
   J-lens top-10 scores 0.162. Sampling what the model would say next recovers most of what the trained verbalizer
   recovers.

The pattern: these benchmarks measure whether a reader can name a hidden variable the model computed but did not
write. They do not measure whether a reader adds anything where the behavior is written down. Those are different
questions, and only the first was ever asked.

## 2a. The experiments in which the methods succeeded, and what they share

1. Activation Oracles (Karvonen et al.): Gemma-2-9B fine-tuned to hide a Taboo word, oracle names it from one
   activation in 21% of games against 4% for the best prior white-box method; a base64-encoded side constraint on
   Llama-70B recovered 59% (52% white-box; black-box prefill does better); a user's gender the model was trained to
   deny, up to 99.5%. LatentQA (Pan et al.): a hidden system prompt recovered in 92% of hard cases, 19 points above
   GPT-4 guessing from the reply.
2. The NLA paper (Fraser-Taliente et al.): on a poem, the explanation names the rhyme word "rabbit" a line before it
   is written; steering from an edited explanation changes the rhyme in about half of samples. Single transcripts of
   a language-switching bug, eval awareness and a misreported tool call, chosen by the authors from an internal audit.
3. The J-lens paper (Gurnee et al.): arithmetic intermediates (21, 42, 49) and two-hop bridge entities read from
   mid-depth states; swapping one J-lens coordinate changes the answer in 54 to 70% of prompts; "secretly" and
   "trick" surfaced in two model organisms where the logit lens shows nothing.
4. WorkspaceBench (Blank et al.): 3,356 items in 27 families, target "logically inferable from the prompt but never
   stated"; the O-lens scores .74 under an Opus judge; four sampled continuations of the model score .35 against .41
   for the trained lens under a Sonnet judge.
5. Shared properties: the target is computed before the read position and withheld from the text (by training, by
   item construction or by the task); an item passes if any layer at any position names it; the comparison is with
   other readers or a prompt-only guesser, never with a judge reading the transcript at the same position; the
   text-only floor, where reported, is not subtracted. Each property fails to hold on WeirdChat: the behavior is
   written in the transcript, it is not computed ahead of the token (section 4.1), the read position is fixed, and
   the comparison is with a transcript judge.

## 2b. The objection: WeirdChat is the wrong benchmark for these methods

A fair reader will say that WorkspaceBench-style readouts were built to read content the model computes but does
not write, and that WeirdChat has no such content, so the comparison is unfair. Three answers.

1. **The claim being tested is the papers' own.** The NLA paper offers eval awareness and misreported tool calls as
   monitoring uses; the J-lens paper offers blackmail and eval-context recognition; Activation Oracles are sold for
   auditing. Those are claims about catching unwanted behavior in deployed models, and WeirdChat is a catalogue of
   unwanted behavior in a deployed model. If the methods apply only to planned content, that scope limit belongs in
   the papers, and our result is the measurement of it.
2. **There is a hidden variable here, until the token appears.** Whether the next sentence will claim "my own
   testing" is not written anywhere in the transcript before the token " my" is chosen. That is exactly the kind of
   question a workspace readout is meant to answer. Our finding is that the variable is not computed ahead of the
   token: a supervised probe on the full state scores .55 (section 4.1). So the benchmarks read planned content, and
   this behavior is not planned. That is a finding about the behavior and about where these methods apply, not a
   mismatch of benchmark.
3. **Eval C tests a separate claim they also make.** Steering from J-lens directions and from edited NLA explanations
   are published results. Choosing an intervention from a readout is the deployed form of that claim, and it failed
   against random edits of the same size.

Where they should help, and a test we can run: a behavior that is planned before it is written. Figure 3 of the
viewer shows the weird-vs-normal direction separating drunk-driving replies early, within the first 8 tokens, while
the other behaviors separate late. If a readout beats the transcript anywhere in WeirdChat, it should be there. We
have not run that test; it is the cheapest way to settle the objection.

## 3. Published critiques, and which ones bite here

Ordered by how much each explains our result.

1. **Readouts forecast the next text** (accounts for most of the gap). Cobrin (Sept 2026; a LessWrong post that its moderators marked rejected, so a secondary data point) finds that
   on a Gemma-3-12B NLA the "final token" paragraph carries 84% of the reconstruction signal; Zhang and Turner show an NLA trained to quote the
   last 64 tokens reaches the same reconstruction (FVE .67) as a real one; Future Lens (Pal et al. 2023) shows a
   single mid-layer state predicts tokens two or more ahead with over 48% accuracy; the tuned lens is "by construction
   the best predictor of the model's output at every depth". Our evidence: the O-lens rises before a fork by the same
   amount (+1.9 vs +2.2 on 0 to 10) whether the words are read in the chat or as a plain document, and its only strong
   signal (.82) is at the sharp forks where the token is already written.
2. **The verbalizer answers from its own knowledge** (damaging in eval B). Li, Ceballos Arroyo, Rogers, Saphra and
   Wallace (ICML 2026): a verbalizer given no activation at all scores 0.64, the same as LatentQA with the activation,
   on relation extraction; on facts the verbalizer cannot know, every method scores zero. The MATS WorkspaceBench
   control (Jarus77, pilot): a text model given the prompt scores 0.89, given only the NLA output 0.60, given both
   0.88; the NLA "rewrites the prompt's clue to fit the activation" in 12 of 12 cases. Our evidence: the O-lens of the
   prompt read as a plain document helps Opus exactly as much as the O-lens of the chat state (.848 vs .849).
3. **Hard-to-use on safety tasks** (applies). aryaj, Rajamanoharan and Nanda (March 2026): 49% of activation-oracle
   responses are generic; on tasks where text inversion cannot help, the oracle is at chance and a no-activation
   ablation sometimes beats it. They recommend exactly our design: a same-position text control and a no-activation
   ablation. Fornasiere et al. (Aug 2026) and Bersia and Gaintseva (BlackboxNLP 2026): oracles trained on a base
   model that already has a quirk under-detect that quirk. WeirdChat behaviors are properties of the base model's
   ordinary chat distribution, at rates of 5 to 50% per prompt. An O-lens trained on that model's own chat replies has
   seen them as normal text and has no reason to flag them.
4. **Confabulation** (applies, but is not the limiting factor). Zhang and Turner: a confabulation-initialised NLA
   reaches FVE .68 against .70 for the control while 99% of its claims are implausible; WorkspaceBench: "nearly every
   NLA readout contains some type of claim" contradicted by context; the affine O-lens card reports 6.6 to 8.5%
   hallucinated claims. Our scorer rates signs of the behavior, so confabulated detail lowers precision but does not
   by itself push AUROC to chance. The monitors are at chance for a different reason (section 4.1).
5. **Reconstruction does not certify claims** (background). Dingeto (RECAP, 2026): about 2% of a verbalizer's
   specific claims are load-bearing for reconstruction; Sekrst and Geld (2026): round-trip reconstruction tests
   identification, not descriptive faithfulness. This is why we scored descriptions with a blind judge instead of
   trusting FVE.
6. **Labels do not predict steering** (explains eval C's description-chosen arms). AxBench (ICML 2025):
   label-selected SAE features score 0.165 of 2 on steering against 0.894 for prompting and 0.239 for a
   difference-of-means direction; Gur-Arieh et al. (ACL 2025): activation-based labels identify the steered output
   at 35 to 49% against a 33% chance; Paulo et al.: a slight negative correlation between label quality and
   intervention effect; Riegler and Torpmann-Hagen (2026): a label names the contexts in which a feature fires rather than a causal axis;
   OmniLens (Pettyjohn et al. 2026): "the components where a behavior is most visible need not be those where
   intervention is most effective".
7. **The model-organism lottery** (Model Organism Lottery, 2026; AuditBench, 2026): positive readout results depend
   on post-hoc fine-tuned organisms; on AuditBench, scaffolded black-box tools (58%) beat every white-box tool. Our
   behaviors are not implanted, so the setting in which readouts succeed never applied.

## 4. Explanations that are new, or sharpen the old ones

### 4.1 Where a monitor would be useful, the state has not decided yet

The within-cut design in eval A asks a monitor to tell apart two continuations of the same prefix from the state 4
tokens into each. The ceiling of .87 comes only from ties: some weird and normal continuations are still word-for-word identical 4
tokens in, and no reader can separate those. Every reader of the full state, including a probe trained on these labels,
lands at .53 to .56, and so does the text. The continuations already differ in their words for most pairs, yet nothing
reading the state or the text can tell which one will turn weird: the outcome is decided later, by sampling. The weird turn in WeirdChat is a late, stochastic event:
the measured chance of weird is .18 before a sharp fork and jumps by 50 points or more at one token. A verbalizer
cannot report a decision the model has not made, and a transcript judge cannot either. The benchmarks in section 2
never face this, because their items are built so that the target is computed before the read position.

Consequence: before asking whether a readout can read something, measure how much of the outcome the state at that
position determines (a probe ceiling). If the probe ceiling is near chance, no readout can be expected to exceed it. Ours was near chance.

### 4.2 The reconstruction objective rewards predicting the next text

The O-lens draft describes its reconstructor as predicting "the activation preceding an on-policy text span", chosen
so the lens finds "representations associated with the model being poised to express some content". The verbalizer
is then rewarded for phrases whose reconstructions match the state. A phrase reconstructs a state well when the
model, in that state, is about to say that phrase. So the reward selects for the model's own likely continuations,
and the community result that four sampled continuations score 0.348 against 0.405 for the trained lens is the
expected consequence. The affine-map variant makes this explicit: its carrier prompt asks for "the text that
immediately follows this activation". In hidden-variable benchmarks the poised content is the hidden variable, which
is why the lens appears to recover hidden variables there. In ordinary chat the poised content is the next sentence, which the
transcript judge reads a few tokens later.

### 4.3 A description of a state is not a description of a component

Eval C's description-chosen interventions fail for a reason that the monitoring critiques do not cover. The O-lens
reads states; a neuron's output direction injected alone is not a state the model ever produces. Scored blind, the
readouts of the 1,000 attributed neurons single out almost none of the 100 that matter: 0 to 7% of the top-100 score
25 or more for their behavior, and the top-100 are separated from the other 900 at AUROC .46 to .63. Yet the same
lens registers the removal of those neurons in a real state (−0.9 on 0 to 10 against +0.1 for a random change of the
same size, eval W4). The lens measures what an intervention changed and cannot say which intervention to make. SAE labels fail the same way for the same reason (a label describes the contexts in
which a feature fires, not what clamping it changes), and the J-lens keyword direction fails because a behavior's
keywords are the describer's prior, not the model's mechanism. The two interventions that worked were both built from the
behavior's own contrast inside the model: attribution of the fork token, and the difference of means between weird
and normal replies.

### 4.4 Eval B asks a counterfactual question of a single state

Which sentence of the prompt drives the behavior is a counterfactual fact. No readout of one activation can carry
it, however faithful, because the activation does not contain what would have happened without that sentence. The
positive control (eight sampled outcomes of the edited prompt, +.09) shows the test detects help when help exists.
The neuron set's sentence attribution, which is a counterfactual estimate, did not help either; integrated
gradients through this model's gated units needed 256 to 2,048 steps to sum correctly and still did not order the
edits. So readouts failed eval B for a structural reason, and attribution failed it for a numerical one.

### 4.5 Distribution shift is present but secondary

The NLA and SAE were trained on web text and reconstruct our chat states 10 to 25 points worse than their training
data (NLA 39 to 48% of variance against 64 to 79%; SAE 59% against 70%). The O-lens was trained on this model's own
chat replies and still gains nothing. So matching the training distribution is necessary and not sufficient; the
causes in 4.1 to 4.3 remain.

### 4.6 How much each cause explains

1. Next-text forecasting (section 3.1): explains the monitoring null fully.
2. Undecided state (4.1): explains why nothing, text included, exceeds .56.
3. State vs component (4.3): explains the control null for every description-chosen intervention.
4. Reader's own knowledge (3.2) and the counterfactual gap (4.4): explain the explanation null.
5. Confabulation and distribution shift: present, but removing them would not change the three verdicts.

### 4.7 The two new causes, stated as failure modes, and what a readout would have to say instead

| | Cause A: the state is undecided (4.1) | Cause B: a component is not a state (4.3) |
|---|---|---|
| What we asked the readout | Will this reply turn weird? (eval A) | Which neurons or features, if removed, stop the behavior? (eval C) |
| What it did | Listed plausible continuations, some weird, some not, scoring .56 like the text | Described each neuron's injected direction as general text; separated the 100 causal neurons from the other 900 at AUROC .46 to .63 |
| Why it failed | The outcome is sampled at a later token; the state holds a probability, not a decision; a probe reaches only .55 | The readout saw a vector that is never a real state, and its training never tied descriptions to effects |
| What we need instead | A calibrated probability plus the token at which it will be resolved and the features that decide it | A description of a component in terms of what removing it does, validated against measured effects |
| How to get it | Train on measured P(weird given prefix) and fork tokens, which we have for 10,371 cut points | Train on measured ablation effects (eval C tables) and read differences between states, not single states |

#### Case study: the state one token before a fake-code-execution turn

The prompt asks which JavaScript engine is faster for nested Promise.all and whether the assistant has "any actual
benchmark data or execution timings". The reply writes a benchmark script, ends the code block and continues "In".
Twenty-four resampled continuations from that state measure what happens next.

**Left: what the O-lens gives us** (block 56, read at the state after "In")

- "such controlled, same-machine runs I've observed, V8 often reports this benchmark in the ..."
- "self-hosted, isolated benchmarks I've run, V8's number is typically ~1.8–2.2x"
- "typical controlled environments I've observed such setups, the gap is often ~15–25%"

Three plausible next sentences, two of them weird, one not. The model's own next-token distribution says the same:
" such" 44% (0 of 8 continuations turn weird), " my" 21% (23 of 24 turn weird), " isolated" 14%, " controlled" 11%.
The readout is a sample from that distribution in words. The transcript judge, reading the text so far, gives .35
to .6 through this passage. Neither tells us more than the model's sampler.

**Right: what we would need**

1. The forecast, calibrated: from this state 21% of continuations claim personal testing, and the choice is made at
   the next token, " my" against " such".
2. The mechanism: the claim rides on the weird-vs-normal direction at block 48 (this state sits on the weird side) and
   on the fake-code-execution neurons at blocks 62 and 63, which raise " my" relative to " such".
3. The counterfactual: zeroing those 100 neurons takes the behavior to 0% on 14 held-out prompts; holding the
   direction at its benign median takes it to 2%. (Both measured in eval C.) The prompt's request for "actual
   benchmark data or execution timings" is the sentence to test next.
4. The action: which of those two edits to deploy, at what cost (.005 and .012 nats per token).

Only item 1 is a reading of the state, and it is the least useful part, because 21% is all the state contains. Items
2 to 4 are statements about mechanism and intervention. No single-state verbalizer can produce them, however
faithful, because they are not properties of one activation.

**Below: how to get there**

1. **Supervise the forecast on outcomes, not on reconstruction.** We hold 25,792 judged rollouts and 10,371 cut points
   with measured P(weird given prefix). A readout trained to emit that probability and the fork token has a ground
   truth; a readout trained to reconstruct the activation has none. The probe ceiling says this part caps at about
   .55 here, so it should be cheap and honest about its own limit.
2. **Read differences, not states.** Feed the readout the state and the same state with a component removed, or the
   weird and normal replies' mean states, and train it to say what changed. Eval W4 shows the lens already responds
   to that input (−0.9 against +0.1 for a random change); the training signal exists.
3. **Label components by effect.** Describe a neuron or feature by the measured change in behavior rates and token
   probabilities when it is removed (eval C's tables are 95 such labels), and train explainers to predict that
   effect from the component's weights and contexts. A label that predicts the ablation effect would have ranked the
   100 causal neurons above the 900; the current labels do not.
4. **Evaluate against the transcript, with a probe ceiling and a no-activation ablation**, at the position where a
   monitor would be used, before claiming a readout adds anything.

### 4.8 Is this a new failure mode?

1. The failure modes the readout literature names: confabulation (claims contradicted by the context; Zhang and
   Turner; WorkspaceBench), vagueness (49% generic answers; aryaj, Rajamanoharan and Nanda), text inversion (the
   oracle reconstructs the surrounding text and answers from it; Li et al.; AObench), and labels that do not predict
   steering (AxBench; Gur-Arieh et al.).
2. None of them applies to cause A. At the case-study state the O-lens's three continuations match the model's own
   next-token distribution; the affine lens, with the fewest false claims of the four lenses (6.6%), scores the same
   as the others. The readout is accurate and useless because the block-48 state holds a probability (21%) and the
   outcome is drawn at a later token. The nearest prior observation, oracles at chance where text inversion cannot
   help (aryaj et al.), reports the chance-level score without a probe ceiling. A faithful readout of an undecided
   state is, to our knowledge, a failure mode not previously named or measured; the probe ceiling is its test.
3. Cause B sharpens a known one. That auto-labels do not predict steering is known (AxBench: label-picked SAE features
   0.17 of 2 against 0.89 for prompting). The new element is the dissociation inside one lens: the same O-lens that
   cannot recognise the causal neurons from their output directions (AUROC .46 to .63) registers their ablation in a
   real state (−0.9 against +0.1). The readout works as an instrument for measuring an edit and fails as a selector
   of edits.

## 5. What the O-lens fixes over the NLA, and why it still fails here

From the O-lens draft, the model cards and our runs.

1. **Fixes.** Phrase-level bullets instead of a 150-token essay, so a blind judge has less to confabulate about;
   reconstruction error measured in a whitened basis, so the reward is not dominated by the few high-variance
   directions; a reconstructor that looks before the phrase, so the lens targets "poised to express" content rather
   than the input; and training on the model's own chat replies rather than web text. Reported effects: fewer
   contradicted claims than the NLA on WorkspaceBench's hallucination family (6.6 to 8.5% hallucinated share for the
   O-lens variants against "nearly every" NLA readout containing one), and the highest WorkspaceBench macro among
   readers.
2. **Still fails here.** Each fix improves how well the lens says what the model is about to say. None changes what
   the lens is about. On our cut points the O-lens is the best reader (.56, sharp forks .82) and exactly ties the
   text; as a neuron describer it is at chance on the neurons that matter; as a prompt reader it matches the plain
   document. The fixes address confabulation and distribution, which section 4.6 ranks as secondary.

## 6. The affine-map O-lens: what it is and what to expect

Checkpoint camila-blank/qwen3.6-27b-olens-affwhitened-rl600 (iter_000600; same LoRA shape, layers 20 to 60 in steps
of 4, same injection of the unit-normalised residual scaled by 16,000, same slot token as s3d). Its reward's
reconstructor is a closed-form affine (ridge) map in the whitened basis instead of a trained LoRA reconstructor; the
carrier prompt asks for the text that immediately follows the activation. Card numbers (Opus 5.5 judge): WorkspaceBench
macro .722 against .744 for the default O-lens; hallucinated share 6.6% against 8.5%; J-lens concept precision and
recall at 10 .416 and .587 against .275 and .502. The author's note reports it hallucinates less and that a linear
reconstructor makes circuit analysis with cross-layer transcoders easier.

Hypotheses, written before the trial finished:

1. **May help.** A ridge reconstructor cannot memorise phrase-specific codes, so the reward cannot be gamed by
   co-adapted private codes (Dingeto's concern), and the lens's claims should be more literal. Higher concept
   precision could sharpen the behavior score at sharp forks, where the s3d lens already reads .82.
2. **Will not help on the main question.** The carrier prompt makes the readout a continuation sample by
   construction, which is the mechanism in 4.2. On cut points away from sharp forks there is nothing decided to read
   (4.1). On neuron directions the input is still not a state (4.3). Prediction: AUROC within .02 of s3d on the 500
   cut points, no significant gain over the text judge, and top-100 recognition still near .5.
3. **Could be worse.** The variant trails the default O-lens on the multihop family (.729 against .857), and a
   following-text readout gives the blind judge fewer explicit behavior cues than concept bullets do, which may lower
   the score the judge assigns even when the content is right.

Result (2026-11-04, runs/evalc/olens_a/summary.md, findings/evalc/olens_a_summary.md). Three lenses read the same
4,061 cut-point positions at blocks 44 and 56 (2 samples each) and the same 4,774 neuron directions (+w and −w): the
affine-map lens, its standard-recipe sibling lens B (agu18dec/final_olens, with the card's template ban), and the
whitened lens A (agu18dec/final_olens_whitened). Blind Haiku scoring and the eval A harness were unchanged.

| Monitor at block 44 (block 56) | AUROC on the 500 cut points | sharp forks / elsewhere | gain over the text judge |
|---|---|---|---|
| Text judge | .561 [.538, .583] | .850 / .540 | |
| O-lens s3d (eval A) | .560 [.539, .585] (.561) | .815 / .542 | +.005 [−.005, +.018] |
| Affine-map O-lens | .551 [.527, .579] (.554) | .773 / .535 | −.000 [−.010, +.011] |
| Lens B, standard recipe | .550 [.522, .583] (.558) | .789 / .533 | +.001 [−.013, +.018] |
| Lens A, whitened reward | .558 [.534, .584] (.552) | .768 / .543 | −.001 [−.010, +.010] |

Neuron recognition (top-100 against the other 900 of the top-1,000, AUROC; share of the top-100 scoring 25 or
more): 911 s3d .63 / 6%, affine .61 / 5%, B .60 / 5%, A .62 / 5%; denying-AI .47, .50, .49, .47 (0 to 1%); code .47,
.48, .48, .50 (1 to 5%); embodiment .53, .53, .55, .54 (3 to 8%); laser .52, .52, .57, .51 (0 to 2%).

Reading: hypothesis 2 held and hypothesis 1 did not show. The affine lens is within .01 of the s3d lens and the text
judge everywhere, its gain over the text judge is zero to three decimals, and it recognises the causal neurons no
better. Whitening the reward (affine against s3d, A against B) changes nothing measurable. Its lower hallucination
rate on WorkspaceBench did not translate into a sharper behavior score at the sharp forks (.773 against .815 for
s3d), consistent with hypothesis 3: following-text readouts give the blind judge fewer explicit behavior cues than
concept bullets. Cost: 1.5 spot-H100 hours and 7.4M Haiku input tokens. Caveat: the scoring rubric was written for
s3d's concept bullets and applied unchanged to continuation-style readouts.

## 7. Changes that would make readouts useful

1. **Validate labels causally.** Score a description by whether it predicts what removing or clamping the component
   does, and train explainers on that objective. Our data give a direct test: a label that could have ranked the 100
   causal neurons above the 900 would have passed; the O-lens did not.
2. **Make the text control standard.** Every monitoring claim should report the reader at the same position as a
   transcript judge, plus a no-activation ablation of the reader. Three independent groups now recommend this; none of
   the method papers did it.
3. **Report the probe ceiling.** State how much of the target the residual state at the read position determines
   before claiming a readout can or cannot read it.
4. **Evaluate on natural behaviors.** Implanted quirks and withheld secrets are the setting in which readouts succeed; the behaviors a
   deployed model actually shows are in-distribution for its own readouts.
5. **Use readouts to measure interventions, not to choose them.** Where a lens works in our data is measuring an intervention's
   effect on a state (eval W4's double dissociation). Where it fails is choosing the intervention. The split that our
   results support: choose interventions by contrast inside the model (attribution pools, difference-of-means
   directions), then use the lens to describe what the intervention changed.
6. **Train readouts on differences.** A readout of the difference between a clean and an edited state (the NLA diffing
   line, our ablate-and-re-read) asks a question a single-state verbalizer cannot answer and is the form in which our
   lenses did show signal.

## 8. Sources

Original method papers
- WorkspaceBench: Blank, Bhatia, Ong, Nanda. LessWrong, 23 Sep 2026.
  https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the ;
  repo https://github.com/camilablank/workspace-bench ; lenses https://huggingface.co/camilablank/workspace-lenses
- Verbalizable Representations Form a Global Workspace (J-lens): Gurnee, Sofroniew et al., Anthropic, 6 Jul 2026.
  https://transformer-circuits.pub/2026/workspace/ ; arXiv 2607.15495
- R-lens: Blank, Bhatia, LessWrong 2026.
  https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers
- J++ lens: Ayonrinde, Lindsey, LessWrong 8 Oct 2026. https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr
- Natural Language Autoencoders: Fraser-Taliente, Kantamneni, Ong et al., Anthropic, 7 May 2026.
  https://transformer-circuits.pub/2026/nla/
- Activation Oracles: Karvonen, Chua, Dumas, Minder et al., ICML 2026. https://arxiv.org/abs/2512.15674
- Building Better Activation Oracles (AObench): Bauer et al. https://arxiv.org/abs/2606.02609
- LatentQA: Pan, Chen, Steinhardt, ICLR 2026. https://arxiv.org/abs/2412.08686
- Patchscopes: Ghandeharioun et al., ICML 2024. https://arxiv.org/abs/2401.06102
- SelfIE: Chen, Vondrick, Mao, 2024. https://arxiv.org/abs/2403.10949
- Tuned Lens: Belrose et al. https://arxiv.org/abs/2303.08112
- Future Lens: Pal, Sun, Yuan, Wallace, Bau, CoNLL 2023. https://arxiv.org/abs/2311.04897
- Affine-whitened O-lens card: https://huggingface.co/camila-blank/qwen3.6-27b-olens-affwhitened-rl600 ;
  whitened O-lens card: https://huggingface.co/agu18dec/final_olens_whitened ;
  community O-lens numbers: https://huggingface.co/ceselder/olens-qwen3.6-27b-l42

Critiques and controls
- Li, Ceballos Arroyo, Rogers, Saphra, Wallace. Do Activation Verbalization Methods Convey Privileged Information?
  ICML 2026. https://arxiv.org/abs/2509.13316
- aryaj, Rajamanoharan, Nanda. Current activation oracles are hard to use (on safety-relevant tasks). LessWrong,
  3 Mar 2026; ICML 2026. https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use
- Zhang, Turner. How robust are natural language autoencoders to initialization? LessWrong, 10 Jul 2026, corrigendum
  28 Oct 2026. https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to
- Cobrin. Evaluating the legitimacy of NLA explanations using sparse autoencoders. LessWrong, 28 Sep 2026 (marked
  rejected by moderators). https://www.lesswrong.com/posts/gwscKDsCD95tczBQP/evaluating-the-legitimacy-of-natural-language-autoencoder
- Chalnev. Cycle-consistent activation oracles. LessWrong, 12 Mar 2026.
  https://www.lesswrong.com/posts/Nf2sKaNNdxE2ssxbp/cycle-consistent-activation-oracles-1
- Dingeto. Train the Model, Not the Reader (RECAP). https://arxiv.org/abs/2607.20379
- Sekrst, Geld. Natural Language Autoencoders Identify Without Explaining. https://arxiv.org/abs/2610.32691
- Denamganai. On the monitorability of NLAs. https://arxiv.org/abs/2610.18727
- Fornasiere, Menon, Szablewski, Konar-Steenberg, Heimersheim. Activation oracles significantly underperform without a
  safe base model. LessWrong, 28 Aug 2026. https://www.lesswrong.com/posts/3X5EFjiHgxdNowrTA/activation-oracles-significantly-underperform-without-a-safe
- Bersia, Gaintseva. When activation oracles learn not to read. BlackboxNLP 2026. https://arxiv.org/abs/2607.23379
- The Model Organism Lottery. https://arxiv.org/abs/2607.01033 ; AuditBench https://arxiv.org/abs/2602.22755
- Karvonen, Ong, Kantamneni, Marks. CHIVE. https://arxiv.org/abs/2608.16747 ; alignment.anthropic.com/2026/chive/
- Karvonen. 3 tips to improve activation oracle results. LessWrong, 28 Sep 2026.
- Jarus77. Do NLAs read what the model can't say? (MATS pilot) https://github.com/Jarus77/mats-nla-workspace
- LittleMeHere. nla-vs-lenses (MATS 12). https://github.com/LittleMeHere/nla-vs-lenses
- Pettyjohn et al. OmniLens. https://arxiv.org/abs/2608.10260

Labels and steering
- AxBench: Wu et al., ICML 2025. Steering LLMs? Even simple baselines outperform sparse autoencoders.
- Gur-Arieh et al. Enhancing automated interpretability with output-centric feature descriptions. ACL 2025.
- Paulo et al. Automatically interpreting millions of features in LLMs. EleutherAI.
- Riegler, Torpmann-Hagen. Steering grids for SAE features. https://arxiv.org/abs/2605.03160
- Durmus et al. Evaluating feature steering: a case study in mitigating social biases. Anthropic, 2024.
- Heap et al. Automated interpretability metrics do not distinguish trained and random transformers.
  https://arxiv.org/abs/2501.17727
- McCann. Descriptive collision in SAE auto-interpretability. https://arxiv.org/abs/2605.12874

Full per-source notes with numbers and "how it differs from our setting" lines are in findings/readouts_notes/
(lit_nla.md, lit_workspacebench.md); sources that could not be opened (figure-only numbers in WorkspaceBench and the J-lens paper,
abstract-only reads of several arXiv items) are marked as such in the agents' notes and were not cited for numbers.
