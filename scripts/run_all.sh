#!/usr/bin/env bash
# Run every evaluation reported in the paper.
#
# Jobs are written as independent commands; each writes one JSON file per
# subject under runs/ and skips subjects that are already complete, so the
# script can be interrupted and restarted. Set EEGOP_PARALLEL=1 to run the CPU
# (classical decoder) jobs in the background while GPU jobs run sequentially.
#
# Prerequisites: scripts/build_cache.py and scripts/fetch_channel_info.py.
set -euo pipefail
cd "$(dirname "$0")/.."

R="python scripts/run_evaluation.py --out runs"
CORE="full,T1,T2,T3"
EX="full,T1,T2,T3,T2_k5,T2_k10,F128,SM"
ALLC="full,T1,T2,T3,T2_k5,T2_k10,F128,SM,T1_JG,T3_c075,T3_c125"
HGDC="full,T1,T2,T3,T2_k5,T2_k10,F128,SM,T1_JG"
CLS="csp,fbcsp,riemann,tslr"

cpu_jobs=(
  "$R --dataset cho2017 --decoders $CLS --configs $ALLC --tag classical"
  "$R --dataset lee2019_mi --decoders $CLS --configs $ALLC --tag classical"
  "$R --dataset lee2019_mi --mode cross --direction AB --decoders $CLS --configs $ALLC --tag classical"
  "$R --dataset lee2019_mi --mode cross --direction BA --decoders csp,riemann,tslr --configs $CORE --tag classical"
  "$R --dataset bnci2014_001 --decoders $CLS --configs $ALLC --tag classical"
  "$R --dataset bnci2014_001 --mode cross --direction AB --decoders $CLS --configs $ALLC --tag classical"
  "$R --dataset bnci2014_001 --mode cross --direction BA --decoders $CLS --configs $ALLC --tag classical"
  "$R --dataset schirrmeister2017 --decoders csp,riemann,tslr --configs $HGDC --tag classical"
  "$R --dataset schirrmeister2017 --decoders fbcsp --configs $CORE --tag fbcsp"
)
gpu_jobs=(
  "$R --dataset cho2017 --decoders eegnet,atcnet --configs $EX --tag deep"
  "$R --dataset lee2019_mi --decoders eegnet,atcnet --configs $EX --tag deep"
  "$R --dataset bnci2014_001 --decoders eegnet,atcnet --configs $EX --tag deep"
  "$R --dataset bnci2014_001 --mode cross --direction AB --decoders eegnet,atcnet --configs $EX --tag deep"
  "$R --dataset bnci2014_001 --mode cross --direction BA --decoders eegnet,atcnet --configs $CORE --tag deep"
  "$R --dataset lee2019_mi --mode cross --direction AB --decoders eegnet --configs $CORE --tag eegnet"
  "$R --dataset schirrmeister2017 --decoders eegnet --configs full,T1,T2,T3,SM --tag eegnet"
  "$R --dataset schirrmeister2017 --decoders atcnet --configs $CORE --tag atcnet"
  "$R --dataset cho2017 --decoders eegnet_bp --configs $CORE --tag eegnet_bp"
  "$R --dataset lee2019_mi --decoders eegnet_bp --configs $CORE --tag eegnet_bp"
)

run() { echo "[$(date +%T)] $1"; eval "$1"; }

if [ "${EEGOP_PARALLEL:-0}" = "1" ]; then
  for cmd in "${cpu_jobs[@]}"; do run "$cmd" & done
else
  for cmd in "${cpu_jobs[@]}"; do run "$cmd"; done
fi
for cmd in "${gpu_jobs[@]}"; do run "$cmd"; done
wait
python scripts/aggregate.py runs --out tables
