# si_data — experimental targets for the T3–T9 conditions

Source: LaGrotta et al., J. Phys. Chem. A (2023), Supporting Information
(jp2c07297_si_001.pdf), Figs. S3–S9 (the "experimental" symbols).

| file | figure | measurement | original data |
|---|---|---|---|
| 1057K_H2O.csv    | S3 | H2O [ppm], 1057 K / 1.83 atm  | Hong et al. 2009 |
| 1880K_OH.csv     | S4 | OH [ppm], 1880 K / 1.74 atm   | Hong et al. 2010 (OH + HO2 via reverse reaction) |
| 968K_ABS215.csv  | S5 | absorbance at 215 nm          | Kappel et al. 2002 |
| 1015K_ABS215.csv | S6 | absorbance at 215 nm          | Kappel et al. 2002 |
| 1079K_ABS215.csv | S7 | absorbance at 215 nm          | Kappel et al. 2002 |
| 1132K_ABS215.csv | S8 | absorbance at 215 nm          | Kappel et al. 2002 |
| 1218K_ABS215.csv | S9 | absorbance at 215 nm          | Kappel et al. 2002 |

**APPROXIMATE.** These values were read by eye from rendered page images of
the figures (no PDF was available to digitize), one representative value per
time along the scatter band.  Expect errors of roughly 3–5 % of the axis range;
early-time OH at 1880 K and the absorbance tails at 1132 / 1218 K are the least
certain in relative terms.  Replace these files with a proper digitization
(same columns: `Time [ms]` and `[H2O] ppm` / `[OH] ppm` / `Absorbance`) before
reporting results; optimize_r1_variants.py reads them unchanged.
