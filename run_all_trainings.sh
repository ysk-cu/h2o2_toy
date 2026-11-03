#!/bin/bash
# Train every surrogate variant at every condition, one CPU thread per run.
#   bothlim / bothlim_r3 : nn_toy_train_bothlimits.py  [r3]
#   eps     / eps_r3     : nn_toy_train_lowP_eps.py    [r3]
# MODE (optional): dp  -> targets at measured times        (tags ..._dp)
#                  tpm -> + T, P, mole fractions as inputs (tags ..._tpm)
# log: <T>_train_<tag>_log.log   result: result_<T>k_train_<tag>.pt
PY=${PY:-/home/yk/miniconda3/envs/nn_env/bin/python}
CONDS=${CONDS:-"1192 1398 1057 1880 968 1015 1079 1132 1218"}
export MODE=${MODE:-}
SUF=${MODE:+_$MODE}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
jobs=()
for c in $CONDS; do
  jobs+=("nn_toy_train_bothlimits.py $c - bothlim$SUF")
  jobs+=("nn_toy_train_bothlimits.py $c r3 bothlim_r3$SUF")
  jobs+=("nn_toy_train_lowP_eps.py $c - eps$SUF")
  jobs+=("nn_toy_train_lowP_eps.py $c r3 eps_r3$SUF")
done
printf '%s\n' "${jobs[@]}" | xargs -P ${NPAR:-36} -L 1 bash -c \
  '$0 "$1" "$2" $([ "$3" = r3 ] && echo r3) $MODE > "${2}_train_${4}_log.log" 2>&1; echo "$2 $4 exit=$?"' "$PY"
