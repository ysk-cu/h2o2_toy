# Stage tpm9: tpm surrogates, 9 conditions

| variant | weighting | objective | data chi2 / points | max abs F (Cantera) | surrogate shift [sd] | LOO pass | significant | at bound |
|---|---|---|---|---|---|---|---|---|
| bothlim | dataset | 16.06 | 63.6 / 63 | 1.42 | 0.27 | 2/9 | R2, R4 | - |
| bothlim_r3 | dataset | 19.91 | 76.1 / 63 | 1.91 | 0.42 | 3/9 | R4, R3 | - |
| eps | dataset | 15.20 | 60.5 / 63 | 1.49 | 0.44 | 2/9 | R4 | - |
| eps_r3 | dataset | 20.99 | 80.2 / 63 | 1.91 | 0.24 | 2/9 | R4, R3 | - |
| bothlim | indep | (missing) | | | | | | |
| bothlim_r3 | indep | (missing) | | | | | | |
| eps | indep | (missing) | | | | | | |
| eps_r3 | indep | (missing) | | | | | | |

Rate constants at their pivot temperature (k/k_nom, 1-sd of ln k):

**bothlim, dataset** (`analysis_9cond_bothlim_tpm_20261104_222311`): R1 x0.94 ±0.07 at 1216 K; R2 x1.10 ±0.07 at 1236 K; R5 x0.88 ±0.17 at 1447 K; R4 x0.62 ±0.14 at 1774 K; HO2 cross section x1.18

**bothlim_r3, dataset** (`analysis_9cond_bothlim_r3_tpm_20261104_222311`): R1 x0.93 ±0.06 at 1221 K; R2 x0.98 ±0.06 at 1254 K; R5 x0.90 ±0.15 at 1440 K; R4 x0.60 ±0.12 at 1618 K; R3 x0.47 ±0.50 at 1333 K; HO2 cross section x1.02

**eps, dataset** (`analysis_9cond_eps_tpm_20261104_222311`): R1 x0.95 ±0.32 at 1285 K; R2 x1.01 ±0.09 at 1312 K; R5 x0.94 ±0.12 at 1427 K; R4 x0.69 ±0.14 at 1830 K; HO2 cross section x1.29

**eps_r3, dataset** (`analysis_9cond_eps_r3_tpm_20261104_222311`): R1 x0.97 ±0.32 at 1285 K; R2 x0.99 ±0.06 at 1213 K; R5 x0.89 ±0.17 at 1437 K; R4 x0.61 ±0.12 at 1661 K; R3 x0.48 ±0.59 at 1455 K; HO2 cross section x0.99
