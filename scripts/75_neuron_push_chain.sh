#!/bin/bash
# neuron-push chain on a VM. Idempotent: a finished stage leaves $OUT/<stage>.done.
#   75_neuron_push_chain.sh <stage> py "<75_neuron_push_gen.py args>"     transformers generation / checks
#   75_neuron_push_chain.sh <stage> judge <reqs> <judge out> [shard]       Gemma 4 31B server + 74's judge client
# Launch detached:
#   setsid nohup flock /data/gpu.lock bash ~/evil-neuron-interp/scripts/75_neuron_push_chain.sh gen0 py "--phase gen --shard 0/3" \
#     >> /data/runs/neuron_push/chain.log 2>&1 < /dev/null &
cd ~/evil-neuron-interp
export VLLM_USE_FLASHINFER_SAMPLER=0
OUT=${OUT:-/data/runs/neuron_push}
mkdir -p $OUT
STAGE=$1; KIND=$2; shift 2
T_START=$(date +%s)
[ -f $OUT/$STAGE.done ] && { echo "$STAGE already done"; exit 0; }
echo "[$STAGE] start $(date -u)"
if [ "$KIND" = py ]; then
  ~/venv/bin/python -I scripts/75_neuron_push_gen.py $1 --out $OUT >> $OUT/py_$STAGE.log 2>&1
  tail -3 $OUT/py_$STAGE.log
  tail -2 $OUT/py_$STAGE.log | grep -q -E "CANDS_DONE|SIGNS_DONE|GEN_DONE|AXCAP_DONE" && touch $OUT/$STAGE.done
elif [ "$KIND" = judge ]; then
  REQS=$1; JOUT=$2; SHARD=${3:-0/1}; MACH=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' _)
  ~/vllm-venv/bin/vllm serve /data/models/gemma-4-31b-it --served-model-name gemma --quantization fp8 \
    --language-model-only --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 256 \
    --max-num-batched-tokens 16384 --enable-prefix-caching --port 8012 --seed 0 \
    --compilation-config '{"backend": "eager"}' --linear-backend marlin \
    >> $OUT/gemma_server_$STAGE.log 2>&1 &
  SP=$!
  for i in $(seq 1 120); do curl -sf localhost:8012/health > /dev/null && break; kill -0 $SP 2>/dev/null || { echo "server died $(date -u)"; exit 1; }; sleep 10; done
  echo "[$STAGE] server up $(date -u) after $(( $(date +%s) - T_START ))s"
  ~/venv/bin/python -I scripts/74_persona_gen_judge.py --reqs $REQS --out $JOUT --port 8012 --shard $SHARD --machine $MACH \
    --texts /data/neuron_push/empty.json --concurrency 256 >> $OUT/judge_$STAGE.log 2>&1
  tail -1 $OUT/judge_$STAGE.log
  ~/venv/bin/python -I scripts/74_persona_gen_judge.py --reqs $REQS --out $JOUT --port 8012 --shard $SHARD --machine $MACH \
    --texts /data/neuron_push/empty.json --limit 1 2>&1 | grep -q " 0 to do" && touch $OUT/$STAGE.done
  kill $SP; wait $SP
fi
echo "[$STAGE] exit $(date -u); GPU time $(( $(date +%s) - T_START ))s"
