# Literature review: verbalizer / readout methods (NLA, AO, LatentQA, Patchscopes, SelfIE, J-lens/O-lens, SAE labels), how they were evaluated, and critiques

Compiled 2026-11-04 from web sources (WebSearch + WebFetch). Every number below was read from the cited page unless marked "(secondary)". Where a page could not be opened or was truncated, that is stated. Our setting, for the "vs ours" lines: no hidden information; behaviour emerges in ordinary chat (WeirdChat, Qwen3.6-27B); the readout is taken at the same position a text judge reads; descriptions are scored blind by a judge; interventions selected from readouts are compared with random edits.

## 0. Bottom line for the write-up

1. Every headline success of these methods was obtained on a task with hidden information not present in the text (Taboo secret words, base64 side constraints, fine-tuned personas, implanted quirks, user gender the model denies knowing), or on a task framed as an anomaly (an implanted motivation). None of the original method papers reports a "judge reads the transcript" baseline at all; the closest baselines are probes, logit/tuned lens, Patchscopes, or black-box elicitation of the target model.
2. The one prior work with our exact framing (naturally occurring behaviours, read-only tools, predictor also sees the transcript) is CHIVE (Karvonen, Ong, Kantamneni, Marks, Aug 2026): AO, NLA and SAE tools give zero uplift over a transcript-only predictor (AUROC 0.814 vs 0.810/0.808/0.812 on Gemma-3-27B-IT; every sweep delta in [-0.012, +0.004] with CIs straddling zero). The authors' explanation is that tool outputs restate features already visible in the transcript and "almost never explicitly state the causal relationship" (explicit causal statements in 0.6% of NLA outputs).
3. Independent critiques converge on a single mechanism: these readouts are LLMs reading tokens. Li et al. (ICML 2026) show a zero-shot verbalizer with no activations matches LIT and beats Patchscopes on the standard feature-extraction benchmarks, and that inverting the input from the activation then reading the text matches LIT; Nanda's group's "hard to use" post shows AOs at chance on three tasks where text inversion cannot help; Zhang and Turner show an NLA initialised on confabulated explanations reaches FVE 0.68 vs 0.70 while 99.3% of its claims are implausible, and an NLA trained to quote the last 64 tokens reaches the same FVE; Cobrin shows the NLA's grounded content is concentrated in the "final token" paragraph. Reconstruction quality (FVE) does not predict plausibility (r = -0.02 in Cobrin).
4. On label-selected interventions: AxBench (ICML 2025) finds SAE features chosen from auto-interp labels score 0.165/2 on steering vs 0.894 for prompting; Gur-Arieh et al. (ACL 2025) find MaxAct labels identify the steered output only 35-49% of the time on Gemma-2 (chance 33%); Paulo et al. find a slight negative correlation between context-based explanation scores and intervention scores; Riegler and Torpmann-Hagen find labels name an activation regime, not a causal axis; Durmus et al. (Anthropic) report off-target effects even for successful bias features. Golden Gate Claude is a demonstration, not an evaluation.
5. Cherry-picking pattern: the NLA paper's case studies (poetry, language switching, misreported tool calls, Mythos audit, blackmail) are single transcripts with no statement of how they were chosen and no text-only control; the J-lens paper's headline figures are illustrative single prompts, with aggregate numbers only for swap experiments; LatentQA and SelfIE do not state how qualitative examples were selected; AO's hallucination-detection example is acknowledged in an appendix to be matched by prompting.

---

## 1. The methods and how they were originally evaluated

### 1.1 Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations
- Fraser-Taliente, Kantamneni, Ong, Marks et al. (Anthropic), Transformer Circuits, 7 May 2026. https://transformer-circuits.pub/2026/nla/ ; LessWrong linkpost https://www.lesswrong.com/posts/oeYesesaxjzMAktCM/natural-language-autoencoders-produce-unsupervised ; code https://github.com/kitft/natural_language_autoencoders ; Anthropic news page https://www.anthropic.com/research/natural-language-autoencoders
- Note: the transformer-circuits page served only ~30k characters to WebFetch, so the quantitative-evaluation section (five prediction tasks, plausibility audit) was not readable; what follows combines the summary page, the LW post and comments, and the GitHub README.
- Method: two LLM modules initialised as copies of the target: an activation verbalizer (AV) that writes text from one residual-stream activation, and an activation reconstructor (AR) that maps the text back to an activation. Warm start: Claude Opus 4.5 writes guesses of "what the model is thinking" for ~100k web-text snippets (the released open-model runs use FineWeb, config qwen7b_fineweb_1M); SFT on these; then joint GRPO with reward = negative normalised MSE (vectors L2-normalised, so MSE = 2(1 - cos)); KL penalty to keep the AV fluent. FVE ~0.3-0.4 after warm start, 0.6-0.8 after RL; grows ~linearly in log(steps). Released open NLAs at ~2/3 depth: Qwen2.5-7B-Instruct L20, Gemma-3-12B-IT L32, Gemma-3-27B-IT L41, Llama-3.3-70B-Instruct L53 (Qwen 7B reference run ~0.75 FVE). Training data is pretraining-style web text, not chat.
- Headline evaluations: (a) "five prediction tasks" on Haiku 3.5, Haiku 4.5, Opus 4.6 showing explanations get more informative over training (example: inferring user occupation from context); comparison condition is earlier checkpoints, not a text judge. (b) Case studies on Opus 4.6 / Mythos Preview: poetry rhyme planning ("rabbit" planned early; steering flips rabbit to mouse in ~half of samples), an early Opus 4.6 answering English prompts in Russian traced via NLA to malformed training data, a Mythos coding run where NLAs read "avoiding detection", a blackmail scenario where NLAs read "constructed scenario". (c) Evaluation awareness: NLAs flag eval awareness in 16% of destructive-code-action test transcripts and 26% of SWE-bench Verified problems vs <1% of opted-in claude.ai traffic. (d) Automated auditing game on a toy model with an implanted hidden motivation: auditors with NLAs win 12-15% of runs vs <3% without (even with other interp tools), without training-data access.
- Stated limitations: confabulation (claims about the context that are "verifiably false"; claims repeated across adjacent tokens are more reliable; the authors read for themes and corroborate); no mechanistic grounding; excess expressivity (AV can add inferences); degenerate solutions possible, including "reproducing the input context verbatim", which KL "partly mitigates" and which they say "doesn't appear to be a significant problem".
- Plausibility audit: Zhang and Turner's post (1.2 below) cites Anthropic's Opus 4.6 NLA as having claims judged plausible 64% (theme), 28% (entities), 24% (details) at FVE 0.61 (secondary; I could not reach the section of the paper that states this).
- Commenter result (Ryan Greenblatt, Gemma-3-27B NLA, easy math): NLA contains the model's eventual answer 80% of the time when correct and 46% when wrong; on 125 wrong-answer records the model's own wrong output appears verbatim 40.7% vs 3.3% for another problem's wrong output; "model-specific wrong reasoning" category 20.8% real vs 0.8% fake-output control; no sign of a hidden chain of thought.
- Cherry-pick flags: all case studies are single transcripts with no statement of selection, no text-only reader, no random-activation control; the only aggregate numbers (eval awareness, auditing game) involve hidden or implanted information.
- vs ours: hidden motivation / eval-awareness that the model does not verbalise; no transcript-reading baseline; explanations read by the authors, not blind-scored.

### 1.2 Cycle-Consistent Activation Oracles (independent NLA)
- Slava Chalnev, LessWrong, 12 Mar 2026. https://www.lesswrong.com/posts/Nf2sKaNNdxE2ssxbp/cycle-consistent-activation-oracles-1
- Qwen3-8B mid-layer; decoder is an AO-style LoRA (activation injected at layer 1), encoder a LoRA with pooling head; 50k-example warmup (LatentQA QA + FineWeb token prediction), then 4k GRPO steps with encoder cosine similarity as reward, K=4, KL penalty.
- Numbers: retrieval of the right activation from 1,000 candidates top-1 95.7%, top-5 99.1%, cosine ~0.8; linear probes beat the decoder on every classification task (SST-2 88.5 vs 72.5; language ID 99.5 vs 50.7, 78.5 with a multilingual checkpoint); arithmetic exact match 46% simple, 0% medium/hard while a probe gets 100% on the first digit, so the information is present but not verbalised.
- Author's own critique: "cycle consistency rewards the decoder for guessing context, not for producing good descriptions"; the decoder can reason forward and report conclusions not in the activation; descriptions substitute entities (Sarah/marbles for Alice/apples).
- vs ours: no behaviour prediction at all; relevant as the earliest statement of the "guesses the context" failure mode.

### 1.3 LatentQA: Teaching LLMs to Decode Activations Into Natural Language
- Pan, Chen, Steinhardt (Berkeley), arXiv 2412.08686, Dec 2024; ICLR 2026. https://arxiv.org/abs/2412.08686
- Latent Interpretation Tuning: decoder = copy of Llama-3-8B-Instruct, reads layer-15 activations written into layer 0; 16,732 training points (4,670 goals, 3,359 personas, 8,703 extractive QA) with controls/dialogs/QA written by o1-preview; some control tokens masked so the decoder cannot read them directly. Chat-formatted synthetic data.
- Evaluations: hidden system prompts (GPT-4 judge): LIT 100% easy / 92% hard, beating GPT-4 prompting by 2.7 / 18.7 points (the intro says 10.8) and SelfIE by ~77 points. Relational extraction: 32.2 points over probes and 38.2 over Patchscopes on average (Country_Curr 86.9 vs 17.7 probe vs 24.3 Patchscope), probe numbers copied from prior work. Control: CrowS-Pairs stereotype 60.9% vs 64.3% no control; harmful elicitation 100% on harmful prompts, 87% on benign ones; Golden-Gate-style steering is qualitative only.
- Limitations stated: three control types only; "with no ground truth for latents, the decoder may hallucinate" (argued unlikely from reading/steering results). Qualitative examples: selection not stated.
- vs ours: the reading task is recovering a hidden system prompt the decoder cannot see; the "prompting" baseline is GPT-4 guessing the persona from the response, i.e. there is a text baseline but the information is deliberately hidden.

### 1.4 Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers
- Karvonen, Chua, Dumas, Fraser-Taliente, Kantamneni, Minder, Ong, Sen Sharma, Wen, Evans, Marks; arXiv 2512.15674, Dec 2025 (v2 Jan 2026). https://arxiv.org/abs/2512.15674 ; blog https://alignment.anthropic.com/2025/activation-oracles/ ; code https://github.com/adamkarvonen/activation_oracles
- Targets: Llama-3.3-70B-Instruct, Gemma-2-9B-IT, Qwen3-8B, Claude Haiku 3.5; activations at 25/50/75% depth for training, ~50% for eval. Data ~1M examples (~65M tokens): LatentQA SPQA, seven binary classification tasks, and self-supervised context prediction (predict tokens before/after an activation span) on FineWeb and LMSYS-style chat. LoRA.
- Four auditing tasks, all with information absent from the input text: Taboo secret word (Gemma-2-9B-IT) AO 21% vs best white-box 4%; Secret Side Constraint (base64 instruction, Llama-70B) AO 59% vs 52% white-box, but black-box prefill beats AO on this task; User Gender (model trained to deny knowing) up to 91-99.5% depending on token selection; emergent-misalignment diffing 2/5 rubric, tied with Activation Difference Lens, slightly above black-box. Claim: matches/exceeds white-box on 4/4 and best overall on 3/4. PersonaQA (fine-tuned biographies): ~14% full mixture vs ~5% base; Haiku 3.5 35%, "within error of each other".
- Baselines: logit lens, SAEs, Patchscopes, probes, black-box prefill/continuation. No no-activation control reported. Appendix F: on a hallucination-detection example "a black-box prompting baseline performs similarly".
- Limitations stated: often wrong, poorly calibrated; auditing benchmarks may not reflect deployment; fine-tunes are narrow; not mechanistic; "too expressive", may "confabulate plausible answers".
- vs ours: all four tasks are secret-elicitation or diffing against a known clean model; nothing like ordinary chat behaviour.

### 1.5 Building Better Activation Oracles (AObench)
- Bauer, De Schamphelaere, Karvonen, Luick, Nanda; arXiv 2606.02609, May/Jun 2026. https://arxiv.org/abs/2606.02609 ; LW https://www.lesswrong.com/posts/heXwuDRfbQQgB5JLP/building-better-activation-oracles
- AObench: 11 evals (Activation Sensitivity / "Not Just Reading Tokens", Hallucination, Missing Information, Sycophancy, Taboo, PersonaQA, MMLU Prediction, Number Prediction, Vagueness, Domain Confusion, Backtracking), reported chance-adjusted; "work in progress". Text-inversion confound defined as the oracle reconstructing the surrounding text and answering from it "as any black-box system could"; only the Activation Sensitivity task controls it by design (same tokens, different upstream context); the authors ran an early text-only baseline with similar performance but call themselves "sufficiently uncertain" to treat it as strong evidence.
- Numbers: chance-adjusted AObench +0.244 baseline, +0.310 with conversational data, +0.435 full recipe; layer 22 vs 18 +0.025, five contiguous layers +0.05; hallucination rate rose 68.8% to 84.6% across the ablation ladder as answers became more specific; vagueness 0.076 to 0.205; Taboo consensus sampling keeps 19.4% of items at 94.3% precision vs 46.6% unfiltered; sycophancy accuracy 0.50 vs AUC 0.83; backtracking 1.26/5 with final token only, 2.10/5 with last 50 tokens (above a black-box baseline given full text). Narrow-task AOs match but do not beat linear probes. Abstract: "capability improvements are marginal."
- vs ours: a benchmark built explicitly because off-the-shelf AOs hallucinate, are vague and are confounded by text inversion; still mostly hidden-information tasks.

### 1.6 Patchscopes
- Ghandeharioun, Caciularu, Pearce, Dixon, Geva (Google), ICML 2024, arXiv 2401.06102. https://arxiv.org/abs/2401.06102
- Patch one hidden state into a template prompt; models LLaMA2-13B, Vicuna 7B/13B, GPT-J, Pythia-12B. Next-token identity vs logit/tuned lens (better from layer 10 up, "up to 98%" gain in layers 18-22); attribute extraction vs logistic probe on GPT-J, better on 6/12 tasks (fruit colour 71.0 vs 35.5; CEO 47.8 vs 5.0), probe wins on object superclass; entity descriptions scored by ROUGE-L vs Wikipedia, declining with layer, with "contamination from the placeholder token" noted; multi-hop correction 50% vs 35.7% CoT vs 19.6% vanilla on 46 filtered samples.
- Caveat the paper itself states: post-patch computation "can add information not present in the patched representation". No text-only baseline for the open-ended description tasks.
- vs ours: factual-attribute recovery, not behaviour; the information is in the entity token.

### 1.7 SelfIE: Self-Interpretation of Large Language Model Embeddings
- Chen, Vondrick, Mao (Columbia), ICML 2024, arXiv 2403.10949. https://arxiv.org/abs/2403.10949
- Embedding inserted five times at layer 3 of a "summarize your message" prompt on LLaMA-2-70B-chat. Evaluations: TextWorld entity state 60-80% recovery, similar to a 100-shot probe (3,400 samples); relevance measured by a treatment-effect score (next-token probability with vs without the embedding), not an LLM judge; harmful-knowledge and prompt-injection results shown mainly as figures; Reinforcement Control reduces injection success 89.06% to 4.4%; ethics override 100% to 2% vs random-embedding control. Smaller models fail instruction following 27% (7B) / 18% (13B) of the time. No limitations section; headline edits are single examples.
- vs ours: illustrative single cases; the embeddings read are at the token carrying the concept.

### 1.8 Verbalizable Representations Form a Global Workspace (Jacobian lens, template lens, oracle lens)
- Gurnee, Sofroniew, ..., Lindsey (Anthropic), 6 Jul 2026. https://transformer-circuits.pub/2026/workspace/ ; arXiv 2607.15495; code https://github.com/anthropics/jacobian-lens
- J-lens: averaged Jacobian from layer l to the final residual (over positions and ~1000 prompts), then unembed; yields a ranked single-token list per activation. Models Sonnet 4.5 (default), Haiku 4.5, Opus 4.5/4.6. Aggregate numbers are for swap interventions, not readout accuracy: 14-category swaps move the target into top-5 on 88% of trials; two-hop answer swaps succeed 54/70/70% (Haiku/Sonnet/Opus) on 50 prompts; function-template swaps 76/192 (101 at double strength); J-space is ~6-10% of activation variance. Baseline comparison to logit/tuned lens is qualitative in the main text. Auditing/eval-awareness sections report surfaced tokens ("fake", "fictional", "secretly", "trick") on misaligned models, no text-judge baseline.
- Template lens (App. A.9.1): per-word LDA directions from Claude-written passages, ~12,700 words, no training; final-layer top-10 contains the sampled next word only 67% of the time; early layers "skip to the answer".
- Oracle lens (App. A.9.2, the "O-lens"): built on Haiku 4.5: reconstructor maps 1-32-token phrases to the preceding activation; 3.4M-phrase dictionary; teacher labels ~1M activations by non-negative OMP with up to 16 phrases; oracle SFT on teacher labels then RL with reward = whitened variance explained. After RL explains on average 31% of whitened variance on held-out Assistant-turn positions. Nine prompts shown (N=4, K=10). Compared to NLAs the authors "suspect" less confabulation because the reconstructor is frozen and linear, "argued, not quantified". A community oracle-lens for Qwen3.6-27B (https://huggingface.co/agu18dec/oracle-lens, returned HTTP 401 to me; secondary) reports held-out FVE ~0.18 vs teacher ceiling 0.483, described as "signs of life".
- Stated limitations: single-token vocabulary; bag of concepts; "some positions and layers yield tokens the authors could not make sense of"; averaging a first-order approximation.
- Community replications: willkn, "Anthropic's J-Lens: A Research Engineer's Analysis", LW 24 Jul 2026 (https://www.lesswrong.com/posts/vHxGD5HKsFuBStirq/anthropic-s-j-lens-a-research-engineer-s-analysis): on GPT-2-medium the raw Jacobian underperforms the logit lens on next-token faithfulness; shrinkage J+lambda*I reaches 0.294 vs 0.275 logit lens at layer 12; softmax over a watched dictionary yields false positives (weather prompt flagged for violence); evidence base is "one small model, one metric, n=32 in places". Louapre, "J-Space: Yet Another LLM Mind Reader?", HF blog 13 Jul 2026 (https://huggingface.co/blog/dlouapre/j-space): mostly expository; notes partial swap effects and that J-space is sparse directions rather than a workspace.
- Cherry-pick flags: headline readouts are single prompts chosen for illustration; no random-direction control on readouts; no reader-only baseline for auditing claims.
- vs ours: vocabulary-token readout designed for unverbalised intermediates; the behaviours in WeirdChat are verbalised in the transcript, so J-lens tokens largely duplicate the text.

### 1.9 Community NLA on our target model
- ceselder, "Qwen3.6-27B Natural Language Autoencoder (layer 42)", HF model card (undated, 2026). https://huggingface.co/ceselder/qwen3.6-27b-nla-L42
- EasyNLA on 6xB200; layer 42/64; AV LoRA r128, AR = 43-block truncation + linear head; warm start reuses FineFineWeb texts and Claude Sonnet 4.6 gold explanations from a Qwen3-8B NLA; 400 GRPO steps; held-out FVE 74.5% (AR-only warm start 59.4%). Web text only. Qualitative: on the agentic-misalignment blackmail honeypot (12 rollouts) the NLA described the scenario as "roleplay / simulation / constructed fiction" and read out eval-awareness clearly only when the model had already verbalised it.
- vs ours: same model class; the author's own example shows the NLA reflecting verbalised content.

---

## 2. Critiques of verbalizers / activation readouts

### 2.1 Do Activation Verbalization Methods Convey Privileged Information?
- Li, Ceballos Arroyo, Rogers, Saphra, Wallace; arXiv 2509.13316 (Sep 2025, v4 May 2026); ICML 2026. https://arxiv.org/abs/2509.13316
- Tests Patchscopes and LIT (LatentQA) with Llama-3.1-8B-Instruct and Ministral-8B-Instruct as both target and verbalizer. Shortcut baseline: the verbalizer gets the input text and question, no activations. Feature extraction (6 relation datasets): Llama zero-shot 0.64 = LIT 0.64 > Patchscopes 0.38; Ministral 0.68 / 0.61 / 0.27. "Inversion then interpretation" (reconstruct the input from the activation, read it) roughly matches LIT (count_curr 0.79 = 0.79) and beats Patchscopes everywhere. Cross-model LIT (Llama activations, Ministral verbalizer) collapses to 0.00-0.20. PersonaQA-Fantasy (facts the verbalizer cannot know): all methods 0.00. When the verbalizer knows more than the target, both methods still answer above zero (fav_food Patchscopes 0.38, LIT 0.45), so answers come from the verbalizer's own knowledge.
- Conclusion: popular benchmarks "can be solved well without access to the target model's internals"; verbalizations "often reflect the verbalizer LLM's own parametric knowledge".
- vs ours: directly supports the "verbalizer says what the input says" explanation; their shortcut baseline is the analogue of our text judge.

### 2.2 How robust are natural language autoencoders to initialization?
- Michael Zhang and Alex Turner (MATS summer 2026, team shard), LW/AF 10 Jul 2026, corrigendum 28 Oct 2026. https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to
- Retrains Anthropic's Qwen2.5-7B NLA with 20k Ultra-FineWeb docs (control FVE 0.70 vs Anthropic 0.75), varying only Claude's warm-start guesses. Carthage addendum removed by RL step ~100 in the original run; "I love Carthage" metaphors fall 9% -> 0.6% -> 2/1000. Confabulated initialisation: post-SFT FVE 0.33 vs 0.61 control, but after RL 0.68 vs 0.70; plausibility (Opus 4.8 judge, 1,000 snippets) 0.08% -> 0.7% for the confabulated NLA while the plausible-initialised NLA fell from 21% to 7.6%. Claude almost always still quotes the correct final token. An NLA trained only on the "final token" paragraph reaches FVE 0.67; Zhang's follow-up NLA trained to quote the last 64 tokens reaches FVE 0.67, "indistinguishable from baseline", while repeating the input's surface text.
- Corrigendum: the released code had a broken KL term (effectively no KL); with KL 0.01 the Carthage addendum persists in ~100% of rollouts, "I love Carthage" drops only to 58%, confabulated plausibility 0.16% at FVE 0.66. Authors: results "strengthen the original findings".
- Comparison they cite: Anthropic's Opus 4.6 NLA plausible 64% theme / 28% entities / 24% details at FVE 0.61.
- vs ours: shows FVE is satisfiable by copying tokens or by confabulation; explains why a reconstruction-trained readout can match but not beat the text.

### 2.3 Evaluating the Legitimacy of NLA Explanations Using Sparse Autoencoders
- Jared Cobrin, LW 28 Sep 2026 (page shows a "Rejected" moderation notice). https://www.lesswrong.com/posts/gwscKDsCD95tczBQP/evaluating-the-legitimacy-of-natural-language-autoencoder
- Gemma-3-12B-IT NLA at L32, 200 Gemma Scope 2 activations, explanations split into the first two paragraphs vs the "Final token" paragraph. FVE 0.69 full / 0.58 final paragraph / 0.27 first two; SAE latents recovered 70% / 59% / 30%; auto-interp features stated in the text 43% / 42% / 20% (mismatched-feature baseline 4.6%); 81% of latents recovered by the first paragraphs are also recovered by the final paragraph; FVE vs groundedness r = -0.02.
- vs ours: small, single model, SAE as imperfect ground truth; but it quantifies that the token-level paragraph carries the grounded content.

### 2.4 On the Monitorability of NLAs: Identification vs Discrimination
- Kevin Yandoka Denamganai, arXiv 2610.18727, Oct 2026, "work in progress, open draft". https://arxiv.org/abs/2610.18727
- Uses the NLA's own reconstructor as a "maximally sympathetic" monitor. True accounts beat other activations' accounts in 320/320 comparisons, but beat fluent single-claim rivals only 60.2% (SE 6.7%, n=35, chance 50%); meaning-preserving rewrites shift distance by only 2.1%, so this is not a private code. Rival claims are transcript-level truths, "the easiest case".
- vs ours: no behaviour prediction; relevant for why blind judges cannot use NLA text to discriminate between close hypotheses.

### 2.5 Current Activation Oracles Are Hard to Use (on Safety-Relevant Tasks)
- aryaj, Senthooran Rajamanoharan, Neel Nanda (MATS 9.0), LW 3 Mar 2026; ICML 2026 listing. https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use ; https://icml.cc/virtual/2026/79576
- Released Qwen3-8B/32B AOs. 49.4% of 77 probes across 33 rollouts judged generic (Opus). Hallucination examples ("BANANA" on a permutation problem). Three tasks where text inversion cannot help: number prediction (same guesses "10"/"12" for every problem, even 1+1); sycophancy AUC ~0.60 (control tokens 0.55) vs dense probe 0.77; missing-information 0/54 correct, same answer in 53/54 token-matched pairs, 59% false flags on complete prompts. Backtracking: ~10% of 2,300 flagged turns correct and specific, ~5% after removing what a text-only LLM could answer; edit experiments 8/15 "correct" updates all restated a keyword right before the probe. Signal: 4-way multiple choice 66% vs 47% text-only; next/previous CoT token F1 0.38/0.45 vs 0.30/0.20; subtle pirate steering at coefficient 0.15 detected by AO but not by a text reader. No-activation ablation on Chinese censorship: AO without activations matched or beat AO with activations (Falun Gong 1/3 vs 2/3; Tiananmen 2/3 vs 3/3; n=3).
- Recommendations: text-inversion-controlled tasks and a no-activation ablation "should become standard checks".
- vs ours: the two controls they recommend are exactly our design; they found the same null on tasks without hidden information.

### 2.6 Activation Oracles significantly underperform without a safe base model
- Fornasiere, Menon, Szablewski, Konar-Steenberg, Heimersheim (LASR Labs), LW 28 Aug 2026. https://www.lesswrong.com/posts/3X5EFjiHgxdNowrTA/activation-oracles-significantly-underperform-without-a-safe
- OLMo-2-1B organisms (Italian food, military submarines; seven training variants each); an AO trained on a quirked base under-detects that quirk in both diffing and non-diffing injection, while detecting the other family's quirk normally; Taboo (gemma-2-9b-it) MO-AO scores significantly below the base oracle in "direct" and "standard" settings. Numbers are in figures only. Proposed mechanism: the AO learns to ignore any activation component present across its whole training distribution.
- vs ours: WeirdChat behaviours are properties of the base model itself, which is this paper's worst case: there is no clean reference and the readout is trained on the same model.

### 2.7 When Activation Oracles Learn Not to Read: Concept-Specific Blind Spots
- Bersia and Gaintseva, arXiv 2607.23379 (Jul 2026), BlackboxNLP 2026; LW post 1 Sep 2026. https://arxiv.org/abs/2607.23379 ; https://www.lesswrong.com/posts/9yETjcrbH7p8x2tLT/when-activation-oracles-learn-not-to-read-concept-specific-2
- Qwen3-8B Taboo subjects (leaf, moon, wave, flag, book); AOs fine-tuned on a subject hiding word W become "anti-readers" for W while cross-word AOs recover it; a linear probe on the AO's L33 states still decodes W; suppression localises to ~L18-23; failing oracles are confident. Numbers in figures only.
- vs ours: readouts inherit the target's reporting habits; silence is not evidence of absence.

### 2.8 Faithful-Patchscopes
- Gong, Yang, Cao, Billard, Wang; arXiv 2602.00300, Jan 2026. https://arxiv.org/abs/2602.00300
- Patchscopes decoding is overridden by linguistic priors (representation encodes "purple broccoli", output says "green"); average faithfulness decrease 18.84% on biased cases (the 11.84-28.47% range is from a secondary summary); BALOR logit recalibration gives up to 33% relative gain.
- vs ours: a prior-override failure that also applies to any LLM-based verbalizer.

### 2.9 Verbalizing Activation Differences with NLAs
- Blasi Nunez, Barmina, Torrielli, Savarimuthu, Schneider-Kamp, Galke Poech; arXiv 2610.30934, 22 Oct 2026, under review. https://arxiv.org/abs/2610.30934
- Uses Anthropic's released NLAs. Analogy composition: target word named at 80-87% at the probe position; Taboo secret-word recovery is best at prefix/template positions, within-prompt hit rates mostly <30%; "book" and "song" named for nearly all organisms (inflating hits); EM audit rubric (1-5): ADL 3.55, AO 3.18, NLA 2.67, black-box agent 1.80; NLAs "open with genre/format labels that often crowd out the finetuning domain"; continued RL raises Qwen secret-word mention 9.2% -> 35.8%, Gemma 5.9% -> 10.0%. Transfer FVE: Llama Taboo finetunes -0.24 to -0.39 in-domain.
- vs ours: diffing against a clean base; its observation that verbalizations begin with format labels regardless of content matches the "generic description" failure.

### 2.10 Retrieval is Enough: Training-Free Interpretability with a Tool-Using Agent
- Balasubramanian and Feizi, arXiv 2607.16448, Jul 2026. https://arxiv.org/abs/2607.16448
- HARP: an agent with a vector DB of activations paired with text, probes and activation arithmetic, no training; claimed to outperform AOs and SAE agents on concept discovery, detection, steering and secret elicitation; concludes training-based methods "do not yet extract insights beyond their training data". Numbers not on the abstract page.
- vs ours: it still uses activations; its point is that trained verbalizers add nothing beyond their training corpus.

### 2.11 The Model Organism Lottery
- Szablewski, Konar-Steenberg, Fornasiere, Menon, Heimersheim; arXiv 2607.01033 (Jul 2026, v2 Oct 2026). https://arxiv.org/abs/2607.01033
- 61 organisms (OLMo2-1B, gemma-3-1b-it, seven training techniques) plus 96 distillation variants; AO, steering, logit lens and J-lens results depend strongly on how the organism was trained; integrated training and distillation give less interpretable organisms than post-hoc SFT. Numbers in body only.
- vs ours: the positive AO/NLA results above come from post-hoc-SFT organisms, the easiest regime; WeirdChat behaviours arose in ordinary post-training.

### 2.12 AuditBench
- Sheshadri, Ewart, Kempf, Fronsdal, Gupta, Schrodi, Conmy, Bowman, Price, Marks, Wang (Anthropic / MATS), arXiv 2602.22755 (v4 Oct 2026). https://arxiv.org/abs/2602.22755
- 56 Llama-3.3-70B organisms (14 behaviours x SDF/TD x SFT/KTO), Haiku 4.5 investigator; default 35.5%. White-box tools average 40.2 -> 46.9% on SDF+SFT, flat on TD (34.4 vs 33.5), 3.5-3.8 pp below default on KTO organisms. AOs and prefill steering are the strongest white-box tools (beat default on 8/14 behaviours); SAEs and logit lens "little benefit"; honesty steering harmful (22.3%). Scaffolded black-box text completion 58.0%, user sampling 53.2%, prefilling 51.6% are the three best tools. "Tool-to-agent gap": agents never called text completion in ~2/3 of runs offering it; agents given noisy tools probed a third as often.
- vs ours: implanted behaviours with adversarial denial training; even there the best tools are black-box.

### 2.13 3 Tips to Improve Activation Oracle Results
- Adam Karvonen, LW 28 Sep 2026. https://www.lesswrong.com/posts/XL3Rdq8mZEnqYym4B/3-tips-to-improve-activation-oracle-results
- Final-token-only AO at chance on backtracking; 20 tokens matches a text baseline; 50 tokens exceeds it. "Open-ended AO answers can confidently hallucinate"; consensus sampling (10 samples, threshold 0.8) keeps 19.4% at 94.3% precision vs 46.6% unfiltered. Accuracy swings ~20 points with question wording; AUC stable (0.50 acc vs 0.83 AUC). States that "recent work found that AOs, NLAs, and SAEs gave no uplift over a strong LLM reading the transcript when predicting model behavior" (CHIVE) and speculates either that current LLMs do little computation not inferable from the text, or that the information is not cleanly in activations.
- vs ours: author of AOs conceding the no-uplift result in our setting.

### 2.14 CHIVE: Would this change your answer? Evaluating Explanations of LLM Behavior in the Wild with Counterfactual Experiments
- Karvonen, Ong, Kantamneni, Marks (Anthropic Fellows / Anthropic), arXiv 2608.16747, 17 Aug 2026; blog 21 Aug 2026. https://arxiv.org/abs/2608.16747 ; https://alignment.anthropic.com/2026/chive/
- Pipeline: WildChat prompts (+~9% agentic), 30 samples per prompt, Opus 4.6 screens for unexpected behaviour, an investigator runs 5-15 counterfactual prompt edits, an independent judge verifies; a claim is "true" if the edit moves the behaviour rate >= 50 points and "false" if <= 15. Targets Gemma-3-27B-IT, Qwen3-8B, Qwen3-32B, Qwen3.5-397B-A17B, Llama-3.1-8B.
- Tool evaluation: Opus 4.8 predictor sees transcript + claim; tool agents get up to 5 read-only calls to an AO, an NLA, or an SAE (top-15 features per position, <= 32 positions per call); intervention methods excluded. Gemma-3-27B-IT AUROC (n=1,294): transcript-only 0.814, AO 0.810, NLA 0.808, SAE 0.812; GPT-5.5 0.819 / 0.810 / 0.812 / 0.809; Gemini-3.1-Pro 0.793 / 0.784 / 0.654 / 0.746. Sweep over tool budget {2,5,10} and positions {16,32,64}: every delta in [-0.012, +0.004], all 95% bootstrap CIs straddle zero. Qwen3-8B "matches". Tools mostly flip predictions toward "no effect". Explicit "X causes Y" statements in 0.6% of NLA outputs for claims whose edit mattered. Wins and losses are qualitative pairs (MiniLM, decompiler vs Twitch "xqc", Gilligan's Island). Filter ablation (n=4,433 unfiltered) unchanged.
- Authors' explanations: natural behaviours are caused by specific context features rather than anomalous concepts, so anomaly-detection strategies from fine-tuned quirks do not apply; activations may be insufficient for causal claims; or current tools do not surface the relation.
- vs ours: nearly identical (naturally occurring behaviours, read-only readouts, predictor has the transcript); differences: they score counterfactual-edit claims generated by an agent, not blind-judged descriptions, and they do not test readout-selected interventions against random edits. Our intervention-selection null extends their result.

---

## 3. Using readouts / labels to select interventions

### 3.1 AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders
- Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts (Stanford), ICML 2025, arXiv 2501.17148. https://arxiv.org/abs/2501.17148
- Concepts taken from Neuronpedia auto-interp labels (gpt-3.5/4o-mini generated); SAE feature = the GemmaScope feature carrying the label; SAE-A picks the feature with best training-set AUROC. Steering score (0-2), average over Gemma-2-2B L10/L20 and 9B L20/L31: prompting 0.894, LoReFT 0.741, SFT 0.676, LoRA 0.615, ReFT-r1 0.543, DiffMean 0.239, SAE 0.165, SAE-A 0.157, probe 0.098. "Better classification does not directly lead to better steering"; "the poor performance of SAEs is at least partially a reflection of the limitations of auto-interpretability".
- vs ours: label-selected features fail to steer even for simple concepts, with a prompting baseline present.

### 3.2 Enhancing Automated Interpretability with Output-Centric Feature Descriptions
- Gur-Arieh, Mayan, Agassy, Geiger, Geva, ACL 2025, arXiv 2501.08319. https://arxiv.org/abs/2501.08319
- Output eval: amplify the target feature and two random features; a judge must pick which generation matches the description (chance 33%). MaxAct (input-centric) scores 49.2 (Gemma-2 res SAE), 35.1 (Gemma-2 MLP), 71.8 (Llama-3.1 res), 36.9 (Llama-3.1-Instruct MLP); VocabProj/TokenChange 54.9-56.5 / 37-40 / 64-72 / 44-46; ensemble up to 66.9. Transluce's MaxAct++ scores 89.8 on the input eval but 39 on the output eval. Input-based descriptions "fail to capture the causal effect of the feature on outputs". Output eval acknowledged as noisy.
- vs ours: this is the closest controlled "does the label predict the intervention" test; their random-feature control is the analogue of our random edits.

### 3.3 Automatically Interpreting Millions of Features in LLMs
- Paulo, Mallen, Juang, Belrose (EleutherAI), arXiv 2410.13928 (v3 Aug 2025). https://arxiv.org/abs/2410.13928
- Introduces intervention scoring (surprisal reduction about the explanation given intervened text, KL-normalised); reports "a slight negative correlation between the fuzzing score and the intervention score", and that many features uninterpretable from contexts are interpretable from their output effects. No coefficient given in text.
- vs ours: context-based labels and causal effects are different objects.

### 3.4 Steering grids for SAE features: when a top-context label names an activation regime rather than a causal axis
- Riegler and Torpmann-Hagen, arXiv 2605.03160 (May/Jul 2026). https://arxiv.org/abs/2605.03160
- Qwen3-1.7B-Instruct, Gemma-2-2B-it, Llama-3.1-8B-Instruct. "AI self-disclaimer" features switch to a contemplative voice (Qwen) or collective "we" (Gemma) under steering; three near-orthogonal features individually substitutable but jointly necessary; feature directions damage unrelated tasks while magnitude-matched random directions do not (non-overlapping 95% CIs on three models). No fraction of revised labels on the abstract page.
- vs ours: a label is one cell of a steering grid; recommends geometry-matched random controls.

### 3.5 Look Before You Steer: Geometry Predicts SAE Feature Steerability
- Khan, Channawar, Gurugubelli, Gupta, Shah; arXiv 2609.22782, Sep 2026; MI workshop ICML 2026. https://arxiv.org/abs/2609.22782
- Steerability (coefficient needed for a fixed effect) is predicted by decoder-space geometry with rho up to -0.546 on Gemma-2, -0.266 on Llama-3.1-8B, binary-only on Qwen3-8B BatchTopK; labels are not evaluated as predictors.
- vs ours: steerability varies widely across features and is not something a label tells you.

### 3.6 Evaluating Feature Steering: A Case Study in Mitigating Social Biases
- Durmus, Tamkin, Clark, ..., Ganguli (Anthropic), 25 Oct 2024. https://www.anthropic.com/research/evaluating-feature-steering ; appendix PDF https://assets.anthropic.com/m/6a464113e31f55d5/original/Appendix-to-Evaluating-Feature-Steering-A-Case-Study-in-Mitigating-Social-Biases.pdf
- 29 label-selected social-bias features from Claude 3 Sonnet on BBQ, steering factors -5..5. "Gender bias awareness" raises gender bias on target and also age bias off target; "Multiple perspectives" at +5 reduces overall BBQ bias ~3%; a neutrality feature reduces bias on nine dimensions. Appendix: effects "are not always predictable" and each method "can have unintended impacts".
- vs ours: label-chosen features have systematic off-target effects even in a success case.

### 3.7 Towards Unifying Interpretability and Control: Evaluation via Intervention
- Bhalla, Srinivas, Ghandeharioun, Lakkaraju; arXiv 2411.04430 (Nov 2024, v2 Feb 2025). https://arxiv.org/abs/2411.04430
- Proposes intervention success rate and coherence trade-off; intervention effectiveness "inconsistent across features and models"; lens-based methods beat SAEs and probes on simple interventions; mechanistic interventions often reduce coherence and "can underperform simpler approaches such as prompting". The "predict/control discrepancy" phrase is secondary (search summary); the abstract does not use it.

### 3.8 Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers
- Heap, Lawson, Farnik, Aitchison; arXiv 2501.17727 (Jan 2025, v2 Jan 2026). https://arxiv.org/abs/2501.17727
- SAEs on randomly initialised Pythia models yield auto-interp and reconstruction scores similar to trained models; recommends randomised baselines before treating labels as evidence of meaning.

### 3.9 Descriptive Collision in SAE Auto-Interpretability
- McCann, arXiv 2605.12874, May 2026. https://arxiv.org/abs/2605.12874
- Marks et al. human annotations (722 features, Gemma-2-2B and Pythia-70M): mean annotation reused across 3.07 features; 82.1% share an annotation; "plural nouns" labels 101 features; detection scores are invariant to collision so cannot certify that a label picks out a feature.
- vs ours: a label that fits many features cannot tell you which feature to intervene on.

### 3.10 Golden Gate Claude / Scaling Monosemanticity
- Anthropic, 23 May 2024. https://www.anthropic.com/news/golden-gate-claude ; paper https://transformer-circuits.pub/2024/scaling-monosemanticity/ (page exceeded fetch size; not re-read).
- Feature clamped to 10x max activation; "most" replies mention the bridge. Presented as a research demonstration; no quantitative steering evaluation; feature selection method not stated on the news page.
- vs ours: the canonical success is a single, strongly token-anchored concept selected by hand, not a behaviour.

### 3.11 The Necessity for Intervention Fidelity: Unintended Side Effects When Steering LLMs
- ICML 2025 MOSS workshop poster. https://icml.cc/virtual/2025/50956 (abstract only). Aggregate metrics hide side effects of steering; proposes "intervention fidelity". Related: SteeringControl (arXiv 2509.13450) and Steering Externalities (arXiv 2602.04896) report secondary-behaviour shifts and increased jailbreak risk from benign steering. Not read in detail.

---

## 4. Cross-cutting comparison with our setting

| Source | Information hidden from text? | Behaviour emerges in ordinary chat? | Text-only / no-activation baseline? | Random control for interventions? | Examples selected? |
|---|---|---|---|---|---|
| NLA (Anthropic 2026) | yes (implanted motivation, unverbalised eval awareness) | partly (Opus transcripts) but behaviours chosen post hoc | no | n/a | single transcripts, selection unstated |
| LatentQA | yes (hidden system prompt) | no (synthetic) | GPT-4 guesses from response | no | unstated |
| Activation Oracles | yes (Taboo, SSC, gender, EM) | no (fine-tuned organisms) | no (black-box elicitation only) | n/a | hallucination example matched by prompting (App. F) |
| Building Better AOs | mostly yes | no | one early text-only run, "uncertain" | n/a | benchmark |
| Patchscopes / SelfIE | no, factual attributes | no | no | random-embedding control (SelfIE ethics) | unstated |
| J-lens / O-lens | aimed at unverbalised intermediates | no (constructed prompts) | no | no random-direction readout control | illustrative single prompts |
| Li et al. 2026 | no | no | yes: zero-shot verbalizer matches LIT | n/a | benchmark |
| Nanda group "hard to use" | no for 3 tasks | partly (CoT, Scruples) | yes: no-activation ablation | n/a | counted failures |
| Zhang and Turner | n/a | n/a | n/a | n/a | 1,000 held-out snippets |
| CHIVE | no | yes (WildChat) | yes: transcript-only predictor | not tested (interventions excluded) | n=1,294 claims |
| AxBench | no | no | prompting | random not needed; probe baseline | benchmark |
| Gur-Arieh | no | no | n/a | yes: two random features | benchmark |

Our evaluation sits in the bottom rows (no hidden information, ordinary chat, text baseline, random-edit control). Every source in the top rows that reports a positive result lacks at least one of those four properties, and the three sources that have all or most of them (Li et al., the Nanda-group post, CHIVE) report nulls.

## 5. Sources I could not open
- https://huggingface.co/agu18dec/oracle-lens (HTTP 401); figures quoted are from a search summary.
- transformer-circuits.pub/2026/nla/ served only a 30k-character summary; the quantitative-evaluation and plausibility-audit sections were not readable. The 64/28/24% plausibility figures are quoted via Zhang and Turner.
- transformer-circuits.pub/2024/scaling-monosemanticity/ exceeded the fetch size limit.
- arXiv abstract pages only (no full text read) for: 2607.16448 (Retrieval is Enough), 2607.01033 (Model Organism Lottery), 2609.22782 (Look Before You Steer), 2605.03160 (Steering grids), 2605.12874 (Descriptive collision), 2411.04430 (Bhalla), 2501.17727 (Heap), 2602.00300 (Faithful-Patchscopes), 2510.01246 (Xie, SAE vs activation-difference steering), 2610.18727 (Monitorability).
- "Huang et al." on explanation scores vs intervention effects, cited in the descriptive-collision paper, could not be identified.
