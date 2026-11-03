#!/bin/bash
# Run optimize_r1_variants.py for every surrogate variant, with both data-set
# weightings, for one stage; then summarize the analyses into <stage>_summary.md.
#   STAGE=dp8   : *_dp  surrogates, 8 conditions (no 1880 K)
#   STAGE=dp9   : *_dp  surrogates, 9 conditions (1880 K with the initial H fraction)
#   STAGE=tpm9  : *_tpm surrogates, 9 conditions (T, P, mole fractions active)
PY=${PY:-/home/yk/miniconda3/envs/nn_env/bin/python}
STAGE=${STAGE:?set STAGE=dp8|dp9|tpm9}
case $STAGE in
  dp8)  MODE=dp;  CONDS=1192,1398,1057,968,1015,1079,1132,1218 ;;
  dp9)  MODE=dp;  CONDS=1192,1398,1057,1880,968,1015,1079,1132,1218 ;;
  tpm9) MODE=tpm; CONDS=1192,1398,1057,1880,968,1015,1079,1132,1218 ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p logs
for v in bothlim bothlim_r3 eps eps_r3; do
  for w in dataset indep; do
    ( $PY optimize_r1_variants.py ${v}_$MODE $CONDS $w > logs/opt_${STAGE}_${v}_${w}.out 2>&1
      echo "$STAGE ${v}_$MODE $w exit=$?" ) &
  done
done
wait
$PY summarize_analyses.py $STAGE ${MODE} $(echo $CONDS | tr ',' ' ' | wc -w) > ${STAGE}_summary.md
echo "wrote ${STAGE}_summary.md"
