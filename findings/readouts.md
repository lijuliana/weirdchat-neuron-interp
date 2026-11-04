# Why the readouts did not beat reading the text, and what would

Written 2026-11-04 after eval C. Numbers from runs/monitors/eval (eval A), findings/chive/report.md (eval B) and
findings/evalc (eval C). Literature notes with URLs: this file's section 8. The affine-map O-lens trial (section 6)
was running when this was written; its result is appended at the end.

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
3. The most damaging explanation here is not confabulation. It is that these readouts are, by training objective,
   predictors of the next text. In ordinary chat the next text arrives a few tokens later and the transcript judge
   reads it directly. A readout that forecasts the continuation cannot beat a reader who sees the continuation.
4. Two further causes are specific to our setting and are not in the literature. (a) At the positions where a monitor
   is useful, the outcome is mostly undecided: a supervised probe on the full residual state scores .55, the same as
   the verbalizers, against a judge-noise ceiling of .87, so the state itself carries little about which way the
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
   direction .55. No readout adds to the text judge after correction. The judge-noise ceiling is .87. At the 34 sharp
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
   ablations. A 33% drop from one keyword direction (our C6) sits inside that range.
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

## 3. Published critiques, and which ones bite here

Ordered by how much each explains our result.

1. **Readouts forecast the next text** (most damaging here). Cobrin (Sept 2026; a LessWrong post that its moderators marked rejected, so a secondary data point) finds that
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
   intervention effect; Riegler and Torpmann-Hagen (2026): a label names an activation regime, not a causal axis;
   OmniLens (Pettyjohn et al. 2026): "the components where a behavior is most visible need not be those where
   intervention is most effective".
7. **The model-organism lottery** (Model Organism Lottery, 2026; AuditBench, 2026): positive readout results depend
   on post-hoc fine-tuned organisms; on AuditBench, scaffolded black-box tools (58%) beat every white-box tool. Our
   behaviors are not implanted, so the easy regime never applied.

## 4. Explanations that are new, or sharpen the old ones

### 4.1 Where a monitor would be useful, the state has not decided yet

The within-cut design in eval A asks a monitor to tell apart two continuations of the same prefix from the state 4
tokens into each. The ceiling of .87 accounts for judge noise only. Every reader of the full state, including a probe
trained on these labels, lands at .53 to .56, and so does the text. The missing .3 is not a readout failure. It is
the share of the outcome that is decided later, by sampling. The weird turn in WeirdChat is a late, stochastic event:
the measured chance of weird is .18 before a sharp fork and jumps by 50 points or more at one token. A verbalizer
cannot report a decision the model has not made, and a transcript judge cannot either. The benchmarks in section 2
never face this, because their targets are computed before the read position by construction.

Consequence: before asking whether a readout can read something, measure how much of the outcome the state at that
position determines (a probe ceiling). If the probe ceiling is near chance, the readout question is moot. Ours was.

### 4.2 The reconstruction objective is a next-text objective in disguise

The O-lens draft describes its reconstructor as predicting "the activation preceding an on-policy text span", chosen
so the lens finds "representations associated with the model being poised to express some content". The verbalizer
is then rewarded for phrases whose reconstructions match the state. A phrase reconstructs a state well when the
model, in that state, is about to say that phrase. So the reward selects for the model's own likely continuations,
and the community result that four sampled continuations score 0.348 against 0.405 for the trained lens is the
expected consequence. The affine-map variant makes this explicit: its carrier prompt asks for "the text that
immediately follows this activation". In hidden-variable benchmarks the poised content is the hidden variable, which
is why the lens looks like it reads minds there. In ordinary chat the poised content is the next sentence, which the
transcript judge reads a moment later for free.

### 4.3 A description of a state is not a description of a component

Eval C's description-chosen interventions fail for a reason that the monitoring critiques do not cover. The O-lens
reads states; a neuron's output direction injected alone is not a state the model ever produces. Scored blind, the
readouts of the 1,000 attributed neurons single out almost none of the 100 that matter: 0 to 7% of the top-100 score
25 or more for their behavior, and the top-100 are separated from the other 900 at AUROC .46 to .63. Yet the same
lens registers the removal of those neurons in a real state (−0.9 on 0 to 10 against +0.1 for a random change of the
same size, eval W4). The lens is a good thermometer and a bad map: it measures what an intervention did, and cannot
say which intervention to make. SAE labels fail the same way for the same reason (a label describes the contexts in
which a feature fires, not what clamping it changes), and the J-lens keyword direction fails because a behavior's
keywords are the describer's prior, not the model's mechanism. The two handles that worked were both built from the
behavior's own contrast inside the model: attribution of the fork token, and the difference of means between weird
and normal replies.

### 4.4 Eval B asks a counterfactual question of a single state

Which sentence of the prompt drives the behavior is a counterfactual fact. No readout of one activation can carry
it, however faithful, because the activation does not contain what would have happened without that sentence. The
positive control (eight sampled outcomes of the edited prompt, +.09) shows the test detects help when help exists.
The neuron set's sentence attribution, which is a counterfactual estimate, did not help either; integrated
gradients through this model's gated units needed 256 to 2,048 steps to sum correctly and still did not order the
edits. So readouts failed eval B for a structural reason, and attribution failed it for a numerical one.

### 4.5 Distribution shift is real but secondary

The NLA and SAE were trained on web text and reconstruct our chat states 10 to 25 points worse than their training
data (NLA 39 to 48% of variance against 64 to 79%; SAE 59% against 70%). The O-lens was trained on this model's own
chat replies and still gains nothing. So matching the training distribution is necessary and not sufficient; the
causes in 4.1 to 4.3 remain.

### 4.6 Which critique matters most, in one line each

1. Next-text forecasting (section 3.1): explains the monitoring null fully.
2. Undecided state (4.1): explains why nothing, text included, exceeds .56.
3. State vs component (4.3): explains the control null for every description-chosen handle.
4. Reader's own knowledge (3.2) and the counterfactual gap (4.4): explain the explanation null.
5. Confabulation and distribution shift: present, but removing them would not change the three verdicts.

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

Result: appended below when the trial completes.

## 7. What would make readouts useful

1. **Validate labels causally.** Score a description by whether it predicts what removing or clamping the component
   does, and train explainers on that objective. Our data give a direct test: a label that could have ranked the 100
   causal neurons above the 900 would have passed; the O-lens did not.
2. **Make the text control standard.** Every monitoring claim should report the reader at the same position as a
   transcript judge, plus a no-activation ablation of the reader. Three independent groups now recommend this; none of
   the method papers did it.
3. **Report the probe ceiling.** State how much of the target the residual state at the read position determines
   before claiming a readout can or cannot read it.
4. **Evaluate on natural behaviors.** Implanted quirks and withheld secrets are the easy regime; the behaviors a
   deployed model actually shows are in-distribution for its own readouts.
5. **Use readouts as instruments, not selectors.** Where a lens works in our data is measuring an intervention's
   effect on a state (eval W4's double dissociation). Where it fails is choosing the intervention. The division of
   labour that our results support: pick handles by contrast inside the model (attribution pools, difference-of-means
   directions), then use the lens to describe what the handle changed.
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
