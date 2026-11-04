# Stage dp8: dp surrogates, 8 conditions

| variant | weighting | objective | data chi2 / points | max abs F (Cantera) | surrogate shift [sd] | LOO pass | significant | at bound |
|---|---|---|---|---|---|---|---|---|
| bothlim | dataset | 20.48 | 107.8 / 56 | 2.85 | 0.59 | 3/8 | R1, R4 | - |
| bothlim_r3 | dataset | 21.55 | 91.2 / 56 | 2.21 | 1.11 | 3/8 | R1, R2, R5, R4, R3 | - |
| eps | dataset | 17.79 | 92.5 / 56 | 2.40 | 0.71 | 3/8 | R1+eps, R2, R4 | - |
| eps_r3 | dataset | 18.79 | 79.1 / 56 | 2.13 | 1.00 | 4/8 | R1+eps, R5, R4, R3, lnEps_AR_R1 | - |
| bothlim | indep | 103.16 | 97.2 / 56 | 2.61 | 1.26 | 3/8 | R1, R5, R4 | - |
| bothlim_r3 | indep | 90.35 | 84.1 / 56 | 2.12 | 1.15 | 2/8 | R1, R4, R3 | - |
| eps | indep | 91.79 | 89.0 / 56 | 2.24 | 0.69 | 3/8 | R1+eps, R2, R4 | - |
| eps_r3 | indep | 78.06 | 72.3 / 56 | 2.09 | 1.16 | 2/8 | R1+eps, R4, R3 | - |

Rate constants at their pivot temperature (k/k_nom, 1-sd of ln k):

**bothlim, dataset** (`analysis_8cond_bothlim_dp_20261104_184049`): R1 x0.93 ±0.05 at 1201 K; R2 x1.07 ±0.10 at 1212 K; R5 x1.00 ±0.13 at 1429 K; R4 x0.40 ±0.25 at 1300 K; HO2 cross section x1.30

**bothlim_r3, dataset** (`analysis_8cond_bothlim_r3_dp_20261104_184049`): R1 x0.95 ±0.04 at 1182 K; R2 x1.01 ±0.07 at 1197 K; R5 x0.80 ±0.27 at 1455 K; R4 x0.23 ±0.23 at 1270 K; R3 x0.27 ±0.38 at 1322 K; HO2 cross section x0.77

**eps, dataset** (`analysis_8cond_eps_dp_20261104_184049`): R1 x1.02 ±0.31 at 1235 K; R2 x1.20 ±0.12 at 1302 K; R5 x0.95 ±0.25 at 1458 K; R4 x0.28 ±0.27 at 1318 K; HO2 cross section x1.17

**eps_r3, dataset** (`analysis_8cond_eps_r3_dp_20261104_184049`): R1 x0.99 ±0.31 at 1202 K; R2 x0.97 ±0.08 at 1195 K; R5 x0.90 ±0.24 at 1455 K; R4 x0.28 ±0.19 at 1319 K; R3 x0.23 ±0.41 at 1263 K; HO2 cross section x0.78

**bothlim, indep** (`analysis_8cond_bothlim_dp_indep_20261104_184049`): R1 x0.92 ±0.03 at 1157 K; R2 x1.03 ±0.07 at 1204 K; R5 x1.06 ±0.07 at 1411 K; R4 x0.41 ±0.18 at 1316 K; HO2 cross section x1.34

**bothlim_r3, indep** (`analysis_8cond_bothlim_r3_dp_indep_20261104_184049`): R1 x0.94 ±0.02 at 1159 K; R2 x0.94 ±0.04 at 1164 K; R5 x0.73 ±0.15 at 1394 K; R4 x0.25 ±0.13 at 1247 K; R3 x0.21 ±0.23 at 1257 K; HO2 cross section x0.74

**eps, indep** (`analysis_8cond_eps_dp_indep_20261104_184049`): R1 x1.12 ±0.30 at 1046 K; R2 x1.18 ±0.06 at 1237 K; R5 x0.96 ±0.14 at 1409 K; R4 x0.28 ±0.14 at 1264 K; HO2 cross section x1.21

**eps_r3, indep** (`analysis_8cond_eps_r3_dp_indep_20261104_184049`): R1 x1.12 ±0.30 at 1053 K; R2 x0.96 ±0.04 at 1181 K; R5 x0.89 ±0.13 at 1409 K; R4 x0.26 ±0.10 at 1252 K; R3 x0.22 ±0.22 at 1139 K; HO2 cross section x0.75
