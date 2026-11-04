#!/bin/bash
# Eval C chain. Local subcommands (laptop) and VM stages (run on evalc-2 under flock /data/gpu.lock).
#   61_evalc_chain.sh sync [VM]                       scp scripts + src + prep + vm prerequisites to the VM
#   61_evalc_chain.sh pull [VM]                       rsync VM:/data/evalc/{gen,judge} -> runs/evalc/{gen,judge}
#   61_evalc_chain.sh <stage> gen "<phases>" [args]   VM: 61_evalc_gen.py (vLLM in-process, hooks); <stage>.done on ALL_DONE
#   61_evalc_chain.sh <stage> judge "<phases>" [args] VM: Gemma 4 31B server (G1/P3/X6 flags) + 61_evalc_judge.py
# Launch a VM stage detached (as the watchdog does):
#   setsid nohup flock /data/gpu.lock bash ~/evil-neuron-interp/scripts/61_evalc_chain.sh damage gen plan,damage \
#     >> /data/evalc/chain.log 2>&1 < /dev/null &
VM=${VM:-evalc-2}; Z=${Z:-us-central1-a}
OUT=${OUT:-/data/evalc}
REPO_LOCAL=/Users/julianali/Desktop/evil-neuron-interp
if [ "$1" = sync ]; then
  VM=${2:-$VM}; cd $REPO_LOCAL
  timeout 300 gcloud compute ssh $VM --zone $Z --command "mkdir -p ~/evil-neuron-interp/scripts ~/evil-neuron-interp/src /data/evalc/prep/vm /data/evalc/gen /data/evalc/judge" || exit 1
  timeout 300 gcloud compute scp --zone $Z scripts/61_evalc_gen.py scripts/61_evalc_judge.py scripts/61_evalc_chain.sh $VM:~/evil-neuron-interp/scripts/ || exit 1
  timeout 300 gcloud compute scp --zone $Z src/judge.py src/sae.py $VM:~/evil-neuron-interp/src/ || exit 1
  timeout 300 gcloud compute scp --zone $Z runs/evalc/prep/pool.json runs/evalc/prep/sets.json runs/evalc/prep/seeds.json $VM:/data/evalc/prep/ || exit 1
  ls runs/evalc/prep/vm/* >/dev/null 2>&1 && { timeout 600 gcloud compute scp --zone $Z runs/evalc/prep/vm/* $VM:/data/evalc/prep/vm/ || exit 1; }
  echo "synced to $VM $(date -u)"; exit 0
fi
if [ "$1" = pull ]; then
  VM=${2:-$VM}; cd $REPO_LOCAL; mkdir -p runs/evalc/gen runs/evalc/judge
  IP=$(gcloud compute instances describe $VM --zone $Z --format='value(networkInterfaces[0].accessConfigs[0].natIP)') || exit 1
  E="ssh -i $HOME/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
  rsync -az -e "$E" julianali@$IP:$OUT/gen/ runs/evalc/gen/ || exit 1
  rsync -az -e "$E" julianali@$IP:$OUT/judge/ runs/evalc/judge/ || exit 1
  rsync -az -e "$E" julianali@$IP:$OUT/chain.log runs/evalc/gen/chain_${VM}.log 2>/dev/null
  echo "pulled from $VM $(date -u)"; exit 0
fi
# ---------------------------------------------------------------- VM stages
cd ~/evil-neuron-interp
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$HOME/vllm-venv/bin:$PATH
mkdir -p $OUT/gen $OUT/judge
STAGE=$1; KIND=$2; shift 2
T_START=$(date +%s)
[ -f $OUT/$STAGE.done ] && { echo "$STAGE already done"; exit 0; }
echo "[$STAGE] start $(date -u)"
if [ "$KIND" = gen ]; then
  PHASES=$1; shift
  ~/vllm-venv/bin/python -I scripts/61_evalc_gen.py --phases "$PHASES" --out $OUT/gen --prep $OUT/prep --vm $OUT/prep/vm "$@" > $OUT/gen_$STAGE.log.tmp 2>&1
  cat $OUT/gen_$STAGE.log.tmp >> $OUT/gen_$STAGE.log; rm -f $OUT/gen_$STAGE.log.tmp
  tail -3 $OUT/gen_$STAGE.log
  grep -q ALL_DONE <(tail -2 $OUT/gen_$STAGE.log) && touch $OUT/$STAGE.done
elif [ "$KIND" = judge ]; then
  PHASES=$1; shift
  MACH=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' _)
  ~/vllm-venv/bin/vllm serve /data/models/gemma-4-31b-it --served-model-name gemma --quantization fp8 \
    --language-model-only --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 96 \
    --max-num-batched-tokens 16384 --enable-prefix-caching --port 8012 --seed 0 \
    --compilation-config '{"backend": "eager"}' --linear-backend marlin \
    >> $OUT/gemma_server_$STAGE.log 2>&1 &
  SP=$!
  for i in $(seq 1 120); do curl -sf localhost:8012/health > /dev/null && break; kill -0 $SP 2>/dev/null || { echo "server died $(date -u)"; exit 1; }; sleep 10; done
  echo "[$STAGE] server up $(date -u) after $(( $(date +%s) - T_START ))s"
  ~/venv/bin/python -I scripts/61_evalc_judge.py --phases "$PHASES" --gen $OUT/gen --out $OUT/judge --port 8012 --machine $MACH "$@" \
    >> $OUT/judge_$STAGE.log 2>&1
  tail -2 $OUT/judge_$STAGE.log
  grep -q ALL_DONE <(tail -2 $OUT/judge_$STAGE.log) && grep -q "errors 0" <(tail -4 $OUT/judge_$STAGE.log) && touch $OUT/$STAGE.done
  kill $SP; wait $SP
fi
echo "[$STAGE] exit $(date -u); GPU time $(( $(date +%s) - T_START ))s" | tee -a $OUT/gpu_stage_seconds.log
