#!/bin/bash
# Local restart watchdog for the spot VM evalc-2 (runs on the laptop).
#   bash scripts/61_evalc_watchdog.sh <stage> <chain args after the stage...>
#   e.g. bash scripts/61_evalc_watchdog.sh damage gen plan,damage
# Every 2 min: if the VM was preempted (TERMINATED/STOPPED) start it again (instance-termination-action STOP keeps the
# disk); if it runs and the stage is neither done nor running, (re)launch 61_evalc_chain.sh <stage> ... detached under
# flock /data/gpu.lock, plus the GPU idle watcher. Exits when /data/evalc/<stage>.done exists, then pulls the results.
VM=${VM:-evalc-2}; Z=${Z:-us-central1-a}
STAGE=$1; shift
ARGS=""
for a in "$@"; do ARGS="$ARGS '$a'"; done
while true; do
  st=$(gcloud compute instances describe $VM --zone $Z --format='value(status)' 2>/dev/null)
  if [ "$st" = TERMINATED ] || [ "$st" = STOPPED ]; then
    echo "$(date -u +%H:%M:%S) $VM $st -> start"
    gcloud compute instances start $VM --zone $Z 2>&1 | tail -1
    sleep 60; continue
  fi
  if [ "$st" != RUNNING ]; then echo "$(date -u +%H:%M:%S) $VM $st"; sleep 60; continue; fi
  r=$(timeout 90 gcloud compute ssh $VM --zone $Z --command "test -f /data/evalc/$STAGE.done && echo DONE || (pgrep -f '[6]1_evalc_chain.sh $STAGE ' >/dev/null && echo RUN || echo IDLE)" 2>/dev/null | tail -1)
  case "$r" in
    DONE) echo "$(date -u +%H:%M:%S) $STAGE done"; bash "$(dirname "$0")/61_evalc_chain.sh" pull $VM; exit 0;;
    RUN) ;;
    IDLE)
      echo "$(date -u +%H:%M:%S) launching $STAGE"
      timeout 90 gcloud compute ssh $VM --zone $Z --command "pgrep -f '^/bin/bash /home/julianali/gpu_idle_watch.sh' >/dev/null || (setsid nohup /bin/bash /home/julianali/gpu_idle_watch.sh >/dev/null 2>&1 < /dev/null &); mkdir -p /data/evalc; setsid nohup flock /data/gpu.lock bash /home/julianali/evil-neuron-interp/scripts/61_evalc_chain.sh $STAGE $ARGS >> /data/evalc/chain.log 2>&1 < /dev/null & sleep 1; echo ok" 2>&1 | tail -1;;
    *) echo "$(date -u +%H:%M:%S) ssh failed ($r)";;
  esac
  sleep 120
done
