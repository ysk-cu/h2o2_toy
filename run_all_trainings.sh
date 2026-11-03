#!/bin/bash
# Train every surrogate variant at every condition, one CPU thread per run.
#   bothlim / bothlim_r3 : nn_toy_train_bothlimits.py  [r3]
#   eps     / eps_r3     : nn_toy_train_lowP_eps.py    [r3]
# log: <T>_train_<tag>_log.log   result: result_<T>k_train_<tag>.pt
PY=${PY:-/home/yk/miniconda3/envs/nn_env/bin/python}
CONDS=${CONDS:-"1192 1398 1057 1880 968 1015 1079 1132 1218"}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
jobs=()
for c in $CONDS; do
  jobs+=("nn_toy_train_bothlimits.py $c - bothlim")
  jobs+=("nn_toy_train_bothlimits.py $c r3 bothlim_r3")
  jobs+=("nn_toy_train_lowP_eps.py $c - eps")
  jobs+=("nn_toy_train_lowP_eps.py $c r3 eps_r3")
done
printf '%s\n' "${jobs[@]}" | xargs -P ${NPAR:-36} -L 1 bash -c \
  '$0 "$1" "$2" $([ "$3" = r3 ] && echo r3) > "${2}_train_${4}_log.log" 2>&1; echo "$2 $4 exit=$?"' "$PY"
