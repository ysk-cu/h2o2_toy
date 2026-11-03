# si_data — experimental targets for the T3–T9 conditions

Source: LaGrotta et al., J. Phys. Chem. A (2023), Supporting Information
(jp2c07297_si_001.pdf), Figs. S3–S9 (the "experimental" symbols), digitized.

Format: no header, two columns: time [ms], value (unsorted; the optimizer sorts
and averages duplicate times).

| file | figure | measurement | original data |
|---|---|---|---|
| 1057_h2o.csv    | S3 | H2O [ppm], 1057 K / 1.83 atm  | Hong et al. 2009 |
| 1880_oh.csv     | S4 | OH [ppm], 1880 K / 1.74 atm   | Hong et al. 2010 (not used yet: needs the initial H fraction as a parameter) |
| 968_abs215.csv  | S5 | absorbance at 215 nm          | Kappel et al. 2002 |
| 1015_abs215.csv | S6 | absorbance at 215 nm          | Kappel et al. 2002 |
| 1079_abs215.csv | S7 | absorbance at 215 nm          | Kappel et al. 2002 |
| 1132_abs215.csv | S8 | absorbance at 215 nm          | Kappel et al. 2002 |
| 1218_abs215.csv | S9 | absorbance at 215 nm          | Kappel et al. 2002 |

Known issues:
- 1218_abs215.csv: the point (0.0862 ms, 0.245) is far above its neighbours
  (~0.13) and above the figure's peak (~0.205); likely a digitization slip.
- 1057_h2o.csv: (0.0116 ms, 1032 ppm) is the outlier also visible in Fig. S3.

The earlier eye-read approximations (<T>K_<OBS>.csv) were replaced by these
files; they remain in the git history (commit 0961cb1).
