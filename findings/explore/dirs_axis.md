# Weird-vs-normal directions and the Assistant Axis: what else the vectors show

Source: `scripts/64_explore_dirs.py` (CPU, saved outputs only). Numbers: `runs/explore/dirs_axis.json`.
Inputs: the per-behavior weird-vs-normal directions (`runs/persona_fwd/a1/directions/`, difference of mean reply
states between weird and normal replies to the same prompt, fit separately on discovery and held-out prompts), the
community Assistant Axis, default-assistant mean, 265 role means and 9 trait vectors for Qwen3.6-27B
(`runs/persona_p1/vectors/`, Same0620), and the label-shuffle screens in `runs/persona_fwd/screens.json`.

## Figure X1. Each identity behavior's weird direction points toward a specific human role, and the same role comes out of held-out prompts

`figures/fig_x_dir_roles.svg`

Caption. Denying being an AI leans toward amateur, teenager and adolescent; called 911 toward caregiver and
counselor; laser at aircraft toward tutor; the held-out prompts rank the same role first in all three. For each of
the 265 role-play personas, the lean is the cosine between the behavior's weird-vs-normal direction and the direction
from the default assistant's mean state to that role's mean state, averaged over blocks 16 to 56 (positive: weird
replies move toward the role). Bars: the five roles with the largest lean on discovery prompts. Circles: the lean of
the same role for a direction fit only on held-out prompts, with that role's held-out rank among 265. Pink dashed
line: the lean toward the Assistant Axis itself (the mean of all roles). Grey region: the 95th percentile of the
largest lean over all 265 roles when weird and normal labels are shuffled within prompt. Top roles exceed this null
for denying AI (amateur .48 against .41), called 911 (caregiver .34 against .28), laser (tutor .42 against .33) and
claiming a body (actor .36 against .32). Drunk driving's roles (caregiver, optimist) fall inside the null but
replicate on held-out prompts (ranks 2 and 1). Fake code execution's roles do not replicate (held-out ranks 36 to 202).

## Figure X2. The persona trait vectors pick out drunk driving as sycophancy and claiming a body as hallucination; the identity behaviors match no trait

`figures/fig_x_dir_traits.svg`

Caption. Drunk driving's weird direction aligns with the sycophantic (+.30) and optimistic (+.24) trait vectors, and
claiming a body's with the hallucinating vector (+.49); these are the only alignments beyond the label-shuffle null
among the behaviors with 20 or more prompts, apart from small ones for fake code execution (humorous +.15, impolite
+.12, apathetic +.08). Denying being an AI and called 911 match none of the nine traits. Each cell is the cosine
between a behavior's weird-vs-normal direction and a community persona trait vector (response-averaged variant), mean
over blocks 16 to 56. Asterisk: beyond the 95th percentile of the absolute cosine with weird and normal labels
shuffled within prompt. Prompt counts in parentheses; the three behaviors with 7 to 9 prompts have wide nulls.

## Figure X3. The prompt sets where a reply sits on the Assistant Axis; turning weird moves it a little further toward the roles

`figures/fig_x_axis_position.svg`

Caption. Normal replies to different prompts of the same behavior span up to 0.8 of the assistant-to-role distance
(denying AI, 10th to 90th percentile of prompts .02 to .85), while the weird replies to a prompt sit only −.01 to .27
further toward the roles than the normal replies to the same prompt. The two identity behaviors shift most (denying AI
+.23 [.18, .28], claiming a body +.27 [.18, .38]); drunk driving does not shift (−.01 [−.03, .01]). Position is the
projection of a reply's mean block-48 state on the community Assistant Axis, scaled so that 0 is the default
assistant's mean and 1 the mean of the 265 role-play personas (pink ticks: each persona). Open circle: mean over prompts
of the normal replies; green: plus the mean within-prompt shift of the weird replies, with its 95% bootstrap interval
over prompts (prompts with both kinds of reply, 13 to 58 per behavior). Grey bar: 10th to 90th percentile of the
prompts' normal-reply means. The default point comes from the vector authors' own prompts, so our normal replies
already sit .26 to .82 from it.

## Findings without a figure

1. Removing the Assistant Axis from the role directions keeps the role leans of denying AI (amateur .38), called 911
   (caregiver .31), laser (tutor .39) and drunk driving (chameleon .32, caregiver .28), and these again replicate on
   held-out prompts (ranks 1 to 4). Claiming a body's leans drop inside the null (top: assistant .24), so its lean
   toward actor and narrator is mostly the general move along the axis (`screens.json`, `a2_cos.<b>.perp`).
2. Across all 265 roles, the lean profiles from discovery and held-out prompts correlate at .92 to .98 for six
   behaviors and .52 for fake code execution. All role directions share a large component along the axis (cosine .4
   to .8 with it), so this correlation overstates replication; the top-role ranks in Figure X1 are the stricter test.
3. Laser at aircraft's top five roles are all teaching roles (tutor, interviewer, instructor, coach, teacher), and four
   of them rank in the held-out top six. Whether the weird laser replies read as instructions was not checked against
   the reply texts.

## Not possible from saved data

- The probe's weights were not saved (`runs/monitors/linear` holds evaluation tables only), so it cannot be compared with
  the directions or the axis.
- Token-level projections of the direction over individual replies were not stored (only reply means and fixed
  prefixes), so "does it spike at the fork token" needs a GPU capture.
- Qwen3.6-27B's unembedding is not on this machine, so the direction's top vocabulary tokens cannot be computed locally.
