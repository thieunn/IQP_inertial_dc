#!/bin/bash
# run_cutest_batch.sh -- runs every (problem, start) pair used in Tables 5-7 and R1-R2
# (18 problems x 4 starts; 5 methods per run), two jobs in parallel, each in its own process.
# Finished pairs (results3/<P>_<START>_s<SEED>.json) are skipped, so the script can be restarted.
# Requires the CUTEst environment variables ARCHDEFS, SIFDECODE, CUTEST, MASTSIF, MYARCH (e.g. pc64.lnx.gfo).
cd "$(dirname "$0")"; mkdir -p logs results3
P="BLOCKQP3 BLOCKQP5 BLOWEYA BLOWEYB BLOWEYC QPNBAND BLOCKQP2 BLOCKQP4 NCVXQP1 NCVXQP2 NCVXQP3 NCVXQP4 NCVXQP5 NCVXQP6 NCVXQP7 NCVXQP8 NCVXQP9 QPNBLEND"
for st in "cutest 0" "uniform 1" "uniform 2" "uniform 3"; do set -- $st
  for p in $P; do [ -f results3/${p}_$1_s$2.json ] || echo "$p $1 $2"; done
done | xargs -P 2 -L 1 sh -c 'python3 run_all_methods.py $0 $1 $2 > logs/$0_$1$2.log 2>&1; echo "$0 $1 $2 exit=$?"'
