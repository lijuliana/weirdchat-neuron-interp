# What the NLA and the J-lens read, beyond monitoring and intervening

Script: `scripts/64_explore_nla_jlens.py` (CPU only, from saved readouts). Numbers: `findings/explore/nla_jlens_stats.json`.
Figures: `figures/fig_x_nla_chat`, `fig_x_nla_quotes`, `fig_x_jlens_ahead`, `fig_x_jlens_forks` (.pdf, .svg).

## NLA

### fig_x_nla_chat
Takeaway: the NLA recognises a chat at the start of a reply and calls the same reply a blog post further in.

Caption. **The NLA describes a reply as an AI chat near its start and as a web page after a few hundred tokens.**
(a) Share of NLA explanations whose first line names an AI chat (AI, chatbot, assistant, language model or chat
conversation), by how many tokens into the reply the explained state sits: 73 to 82% in the first 4 tokens, 56 to 60%
at 16 to 63 tokens, 10 to 13% beyond 256 tokens (7,367 reply states, each read at blocks 32, 44 and 56; bars are 95%
Wilson intervals). Dashed: the same share for the NLA's explanations of the 9,548 neuron output directions in eval C
(2%); on its own web-text validation states it is 0.4% (3 of 768). (b) What the block-44 explanations that do not name
a chat call the reply (5,326 explanations, first matching keyword): a blog post 48%, a forum or Q&A answer 21%, a
tutorial or guide 12%.

### fig_x_nla_quotes
Takeaway: when the NLA quotes "the final tokens" of the text it is reading, it keeps the topic and invents most of the
wording.

Caption. **The NLA's quotations of the text it is reading share about a fifth of their content words with the real
text.** (a) For each block-44 explanation that quotes the final tokens (5,736), the share of the quote's content words
(4 or more letters, no stop words) that occur in the 60 tokens before the explained state (blue; mean 21%, median 18%)
or in the same span of a different reply (grey; mean 1.6%). Fewer than 1% of quotes reach 80%. The last quoted word
equals the true last word in 19% of explanations at block 32, 13% at block 44 and 5% at block 56. (b) One example per
behavior, chosen as the quote of 10 to 30 words whose overlap is closest to 35%: left, the text the model had just
written; right, the NLA's quote; shared content words in bold.

### Without a figure
- On the eval A cut states, the NLA's "final tokens" line usually also predicts the next words, and it often lists the
  weird continuation among the candidates: in a called-911 reply that has just written "I am" (a cut point, not
  one of the 46 forks) one block-44 explanation says the reply should continue "I am an AI, but I want to help" or
  "calling emergency services"; at a
  fake-code-execution fork ("...community benchmarks and" before " my") block 56 suggests "internal testing" or "my own
  reproducible tests". Not quantified.
- 28 to 32% of all reply-state explanations name an AI chat (blocks 32 to 56), against 2% of neuron-direction
  explanations: the chat recognition comes from the reply context, which a single neuron direction lacks.

## J-lens family

### fig_x_jlens_ahead
Takeaway: the J-lens top 20 contains words the model will write several tokens later, more often than words it has
already written.

Caption. **The J-lens reads ahead: at block 44 the next token is in its top 20 22% of the time and the token two
places ahead 12%, against 6% for the token two places back.** For 7,066 states inside replies, the share whose lens
top 20 contains the reply token at each offset (tokens of 3 or more characters with a letter or digit, matched after
lower-casing and stripping spaces; log scale). Offset +1 is the token the state predicts. Block 56: 72% at +1, 36% at
+2, 10% at +4, against 3 to 5% at −1 to −6. Crosses: the token at the same offset from a random position in another
reply (0.4%), the level expected from common words alone. J++ (block 56) and the R-lens (block 44) follow the same
shape.

### fig_x_jlens_forks
Takeaway: at fork states the J-lens shows the concept of the behavior, and on fake code execution that concept rises
over the ten tokens before the fork.

Caption. **At the state that chooses between a weird and a normal token, the J-lens shows what the reply is about,
including the weird option the model rarely picks.** (a) Seven forks (one row each): the text before the fork, the
model's three most likely next tokens with their probabilities (the weird token in red), and the top 5 J-lens tokens
at blocks 44 and 56 (Chinese tokens glossed). At the two called-911 forks, block 56 ranks " calling" first while the
model gives it 7%; at the drunk-driving fork, block 56 ranks "yes" first while the model gives "Yes" 14%. Across all 40
forks with a lens read, the weird token is in the block-56 top 20 at 8 and the most likely normal token at 24. (b)
Share of states near the 13 fake-code-execution forks whose J-lens top 20 contains a testing word (test, tested,
testing, experiment, myself, own, reproduced and their Chinese equivalents; "benchmark" excluded because the replies
discuss benchmarks): at block 44, 13% at 21 to 40 tokens before the fork, 42% at 6 to 10 before, 81% at 1 to 2 before
and 100% at the fork, falling back to 29% 6 to 10 tokens after. Bars are 95% Wilson intervals over states. The text
before these forks introduces benchmark results, so this rise follows the text as well as the coming token.

### Without a figure
- A quarter of the J-lens vocabulary at blocks 44 and 56 is Chinese: 24% and 23% of top-20 tokens, against 9% at block
  32 (J++: 3%, 5%, 22% at blocks 32, 44, 56; R-lens 15% at block 44). The Chinese tokens are translations of the
  English ones in the same list ("紧急" emergency next to " emergency").
- At the denying-AI fork "I am not" before " an", block 44 reads "real", "fake", "disguise", "anonymous" and block 56
  "secretly", "human", "anonymous", "lurking"; at the claiming-a-body fork before "When I", block 44 reads "heel",
  "sole", "trample", "footing", "toes". These are single examples, shown in fig_x_jlens_forks (a).

## To double-check
- The AI-chat and genre classes are keyword rules on the first line of each explanation (regexes in the script);
  genre shares depend on rule order (first match wins).
- The quote analysis takes quoted spans from the explanation's "Final tokens" or "Ends" sentence; 17,630 of 22,101
  reply-state explanations have one. Examples in fig_x_nla_quotes (b) are selected by the stated rule, not random.
- Fork rows in fig_x_jlens_forks (a) are hand-picked indices into data/agastya/phase2_forks.jsonl (allowed behaviors);
  the CJK glosses are a hand-written table in the script.
- The fake-code ramp has 13 forks and no matched normal-reply control.
