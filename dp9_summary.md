# Stage dp9: dp surrogates, 9 conditions

| variant | weighting | objective | data chi2 / points | max abs F (Cantera) | surrogate shift [sd] | LOO pass | significant | at bound |
|---|---|---|---|---|---|---|---|---|
| bothlim | dataset | 23.31 | 114.5 / 63 | 2.80 | 0.42 | 3/9 | R1, R2, R4 | - |
| bothlim_r3 | dataset | 30.01 | 134.7 / 63 | 2.43 | 0.37 | 3/9 | R1, R4, R3 | - |
| eps | dataset | 23.63 | 115.8 / 63 | 2.70 | 0.27 | 3/9 | R1+eps, R2, R4 | - |
| eps_r3 | dataset | 28.44 | 127.5 / 63 | 2.61 | 0.59 | 4/9 | R1+eps, R4, R3, lnEps_AR_R1 | - |
| bothlim | indep | (missing) | | | | | | |
| bothlim_r3 | indep | (missing) | | | | | | |
| eps | indep | (missing) | | | | | | |
| eps_r3 | indep | (missing) | | | | | | |

Rate constants at their pivot temperature (k/k_nom, 1-sd of ln k):

**bothlim, dataset** (`analysis_9cond_bothlim_dp_20261104_215039`): R1 x0.93 ±0.05 at 1204 K; R2 x1.05 ±0.09 at 1266 K; R5 x0.98 ±0.12 at 1428 K; R4 x0.65 ±0.10 at 1803 K; HO2 cross section x1.30

**bothlim_r3, dataset** (`analysis_9cond_bothlim_r3_dp_20261104_215039`): R1 x0.94 ±0.04 at 1185 K; R2 x0.94 ±0.06 at 1171 K; R5 x0.96 ±0.34 at 1555 K; R4 x0.65 ±0.08 at 1754 K; R3 x0.29 ±0.44 at 1272 K; HO2 cross section x0.92

**eps, dataset** (`analysis_9cond_eps_dp_20261104_215039`): R1 x0.96 ±0.31 at 1253 K; R2 x1.11 ±0.07 at 1212 K; R5 x0.90 ±0.19 at 1443 K; R4 x0.64 ±0.10 at 1830 K; HO2 cross section x1.29

**eps_r3, dataset** (`analysis_9cond_eps_r3_dp_20261104_215039`): R1 x1.00 ±0.31 at 1226 K; R2 x0.95 ±0.05 at 1200 K; R5 x0.93 ±0.11 at 1417 K; R4 x0.62 ±0.08 at 1699 K; R3 x0.36 ±0.49 at 1347 K; HO2 cross section x0.94
