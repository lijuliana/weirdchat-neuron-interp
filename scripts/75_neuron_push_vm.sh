#!/bin/bash
# push code + inputs to an npush VM and launch a chain stage: 75_neuron_push_vm.sh VM ZONE STAGE "<gen args>"
VM=$1; Z=$2; STAGE=$3; GARGS=$4
cd "$(dirname "$0")/.." || exit 1
timeout 120 gcloud compute ssh $VM --zone $Z --command "mkdir -p /data/neuron_push /data/runs/neuron_push" || exit 1
timeout 300 gcloud compute scp --zone $Z scripts/75_neuron_push_gen.py scripts/75_neuron_push_chain.sh scripts/74_persona_gen_judge.py $VM:~/evil-neuron-interp/scripts/ > /dev/null || exit 1
timeout 300 gcloud compute scp --zone $Z src/harness.py src/judge.py $VM:~/evil-neuron-interp/src/ > /dev/null || exit 1
timeout 300 gcloud compute scp --zone $Z data/neuron_push/units.json data/neuron_push/dirs.npy data/neuron_push/prompts.json data/neuron_push/signs.json data/neuron_push/prompt2_cands.json data/neuron_push/empty.json $VM:/data/neuron_push/ > /dev/null || exit 1
timeout 120 gcloud compute ssh $VM --zone $Z --command "pgrep -f '^/bin/bash /home/julianali/gpu_idle_watch.sh' >/dev/null || (setsid nohup /bin/bash /home/julianali/gpu_idle_watch.sh >/dev/null 2>&1 < /dev/null &); setsid nohup flock /data/gpu.lock bash /home/julianali/evil-neuron-interp/scripts/75_neuron_push_chain.sh $STAGE py '$GARGS' >> /data/runs/neuron_push/chain.log 2>&1 < /dev/null & sleep 1; echo launched $VM $STAGE"
