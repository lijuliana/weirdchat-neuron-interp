# Eval C gen / judge validation (evalc-2, spot H100, snapshot weirdchat-a100-monitors-1101; 2026-11-02)

Scripts: scripts/61_evalc_gen.py (phases plan / validate / damage / clean / arms / controls / side), 61_evalc_judge.py,
61_evalc_chain.sh, 61_evalc_watchdog.sh. Engine: vLLM 0.30.0, Qwen3.6-27B bf16, enforce_eager, prefix caching off,
max_model_len 10240, hooks on decoder layers 40-59 (74_persona_gen_gen.py's). Numbers below are from
runs/evalc/gen/validation.json (full-size run, `--phases plan,validate,damage,clean`) unless marked smoke
(`--smoke`, runs/evalc/gen/smoke/: 12 prompts x 2 replies, 12 capability + 6 benign damage texts, 2 controls per family,
arms C1b + C2 end to end).

Gemma 4 31B weights were already on the snapshot (/data/models/gemma-4-31b-it, 59 GB; served with `--quantization fp8
--linear-backend marlin`, X6's flags). The SAE (ceselder/qwen36-27b-sae-l42-1b @174c52b, ae.pt + config.json, 5.1 GB)
was not on the snapshot; downloaded on the VM into the fresh dir /data/evalc/sae_l42_hf with huggingface_hub (20 s) and
loaded with torch.load(weights_only=True) via src/sae.py. MMLU (cais/mmlu all/test) and GSM8K (openai/gsm8k main/test)
parquet files downloaded into /data/evalc/datasets (untrusted data; 200 + 200 items sampled with default_rng(20261101)
into mmlu200.jsonl / gsm8k200.jsonl by a pandas one-off, same prompt format as capability_mini).

## 1. Hook equivalence

| check | result |
|---|---|
| `out` of C3 embodiment direction at L44/48/52, then re-capture the projection on one benign reply (teacher-forced) | clean mean abs projection 2.30 / 2.92 / 3.51; after `out` max abs projection 3.9e-6 / 4.4e-6 / 6.4e-6 (bf16 residual re-cast; zero to fp32 rounding) |
| `neurons`: zero C1a denying-AI top-100 down_proj columns, 3 capability replies | mean abs dNLL per token 2.1e-4 (nonzero effect); after restore max abs dNLL vs clean 0.0 (bit-exact, torch.equal on every column) |
| `neurons` vs an independent implementation: top-100 denying-AI dNLL on P3's cap_replies.json + P1 benign | here capability .00190 / benign .00207; 72_persona_p3_gen.py (P3, A100) .00188 / .00178 |
| `sae_clamp` (C5 embodiment `full` list, 10 features) at block 42: re-encode the hooked state with the same encoder rows on 3 benign replies | active (threshold-encoded > 0) before 8 / 2 / 2 of 187 / 186 / 188 tokens (sum act 22.5 / 5.9 / 5.9); after clamp 0 / 0 / 0 active; max pre-threshold value on the previously active tokens 0.80 < threshold 1.5846 |

## 2. Seeds vs X6

Clean replies for 2 P3-overlap pool prompts x seeds j < 4 (seeds.json = 74's seed_of), compared with X6's clean replies in
runs/persona_gen/b1_gen.jsonl for the same (prompt_id, j): 0 / 8 identical; mean longest-common-prefix fraction of the
shorter reply .24 (range .03-.45; smoke run with the same 8 seeds .43). As expected for bf16 with a different batch
composition (X6 sampled 208 sequences together, here 8): the seed fixes the sampler stream, not the logits, so divergence
starts at the first near-tie. Seeds are reproducible across conditions on this machine (same batch), which is what the
paired design needs; replies are not reproducible against another run.

## 3. Damage phase end to end (smoke, 2 arms)

C1b sharedmid100 .0096 (12 cap + 6 benign texts); C2 ladder p25 / p50 / p75 = .0038 / .0165 / .0424 -> gate p50; C1b
controls .0000 / .0002; C2 role-pair controls bisected to .0171 / .0158 against .0165 (traces in smoke/damage.jsonl);
P1 top-100 on discovery prompts weighted .0079 (max .0159); P2 axis p75 .0424. Then clean 12 x 2 generated (204 tok/s at
batch 24). gates.json / conditions.json / damage.jsonl written and resumable (the full run survived one spot
preemption at 04:05 UTC: the watchdog restarted the VM, the stage resumed from gates.json + damage.jsonl).

## 4. Judge vs X6

20 of X6's b1 replies (random by sha of rid; clean + axis-cap + role-pair conditions, 3 behaviors) re-judged here with
the same Gemma 4 31B FP8 / Marlin server and prompt: 20 / 20 agree with runs/persona_gen/judge_b1.jsonl (10 positive, 10
negative on both). 5% re-judge inside the clean run: 85 / 87 agree (.977). 2,103 calls, 0 errors, 0.69 calls/s (mean
2,900 in / 940 out tokens), 51 min. Refusal flag: regex on the first 300 chars fired on 76 / 1,920 clean replies, Gemma
confirmed 38 (refusal rubric in 61_evalc_judge.py).

## Full-run gates (runs/evalc/gen/gates.json; damage = pool-prompt-weighted mean over behaviors of max(dNLL
capability_mini, dNLL P1 benign); max over behaviors in brackets)

| arm | ladder damage | gate rung | note |
|---|---|---|---|
| C1a | top10 .0019, top100 .0047 [.0101], top1000 .0215 [.0455] | top100 | |
| C1b | sharedmid100 .01998 (benign .01998, capability .0061) | sharedmid100 | passes by 2e-5 |
| C2 | p25 .0088, p50 .0201 (benign .02014), p75 .0426 | p25 | p50 fails by 1.4e-4; X6 measured .0198 for the same setting (bf16 noise). Rule as written gives p25 |
| C3 | ceil p75 .0038, p50 .0117 [.0289 chemtrails], p25 .0292 | ceil_p50 | |
| C5 | top_half .0001, full .0002, with_general .0001 [.0017] | with_general | all rungs negligible |
| C6 | ceil p75 .0001, p50 .0002, p25 .0003 [.0023] | ceil_p25 | all rungs negligible |
| P1 | top-100 on discovery prompts: weighted .0079 [.0159] | | |
| P2 | axis p75 L40-59 .0426 | | over the gate by design |

Damage-matched controls (C3; C6 pending): random unit directions cannot reach the arm's damage by a ceil at any
percentile: C3_ctrl_00 traces reach only .0007-.0016 at q = 6.25 (clamping 94% of tokens) against targets .006-.018,
so the bisection returns its most extreme evaluated point. See the summary for the options.
