#!/bin/bash
# CHIVE Eval B: everything after the fleet is pulled (findings/chive/design.md 4-8). Run from the repo root.
#   bash scripts/98_chive_run_downstream.sh [stage...]   stages: spec labels claims pos predict audit score (default all)
# spec     merge the per-shard top-up specs data/chive/topup/t_s*.json into t1.json (what 94 reads)
# labels/claims/pos   94_chive_claims.py
# predict  95_chive_predict.py on every arm (plain, plain_rep, 3 methods x {real, ctrl, shuf}, pos); ~$390 Opus
# audit    97_chive_blind_audit.py (can a reader tell a real block from its control?)
# score    96_chive_score.py with the explicit arm list
set -e
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
STAGES=${*:-"spec labels claims pos predict audit score"}
EV=data/chive/evidence
ARMS="plain plain_rep neurons neurons_ctrl neurons_shuf olens olens_ctrl olens_shuf axis axis_ctrl axis_shuf pos"
for st in $STAGES; do
  echo "== $st $(date -u +%H:%M:%S)"
  case $st in
    spec)
      $PY -I - <<'PY'
import glob, json, hashlib
ids = set()
for f in sorted(glob.glob("data/chive/topup/t_s*.json")):
    if f.endswith("_summary.json"): continue
    t = json.load(open(f)); assert t["edit_j"] == [32, 64]; ids |= set(t["edit_ids"])
spec = {"tag": "t1", "edit_ids": sorted(ids), "edit_j": [32, 64], "merged_from": sorted(glob.glob("data/chive/topup/t_s?.json"))}
json.dump(spec, open("data/chive/topup/t1.json", "w"), indent=1)
print(f"t1.json: {len(ids)} edits from {len(spec['merged_from'])} shard specs, sha256 {hashlib.sha256(open('data/chive/topup/t1.json','rb').read()).hexdigest()[:16]}")
PY
      ;;
    labels|claims|pos) $PY -I scripts/94_chive_claims.py $st ;;
    predict) $PY -I scripts/95_chive_predict.py --plain-rep --evidence $(for a in neurons neurons_ctrl neurons_shuf olens olens_ctrl olens_shuf axis axis_ctrl axis_shuf pos; do echo -n "$EV/$a.jsonl "; done) ;;
    audit) $PY -I scripts/97_chive_blind_audit.py ;;
    score) $PY -I scripts/96_chive_score.py --arms $ARMS ;;
    *) echo "unknown stage $st"; exit 1 ;;
  esac
done
