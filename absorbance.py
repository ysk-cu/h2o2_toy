"""
absorbance.py -- observables of the shock-tube targets, shared by the training
scripts and optimize_r1_variants.py so both evaluate exactly the same quantity.

  OH, H2O : mole fraction (one surrogate column per target time)
  ABS215  : 215 nm absorbance (Kappel et al. 2002, LaGrotta et al. 2023 SI Figs.
            S5-S9).  The surrogate predicts the H2O2 and HO2 mole fractions at each
            target time (two columns); the optimizer forms
                A(t) = L * c(t) * (sigma_H2O2(T) X_H2O2 + s_HO2 * sigma_HO2(T) X_HO2)
            with c = P/RT [mol/cm^3], L the path length and s_HO2 a cross-section
            scale that is an active parameter there (as sigma'_1,HO2 is in the MSI
            fit), so the cross sections never have to be fixed at training time.

  Nominal cross sections (sigma'_1, sigma'_2 at 215 nm in the LaGrotta SI table):
      sigma_H2O2(T) = 96000 - 19 T          cm^2/mol   (linear in T)
      sigma_HO2(T)  = 1.155e6 exp(-T/1299)  cm^2/mol
  ASSUMPTION: the SI lists the parameters and units but not the functional forms;
  these are the forms implied by the units (cm^2 mol^-1 K^-1 for sigma'_2,H2O2 and
  K for sigma'_2,HO2).  They only set the target-time selection and the nominal
  point of the optimizer; s_HO2 and the per-condition L absorb a scale error.
"""

import numpy as np

SIG_H2O2_215 = (96000.0, -19.0)      # cm^2/mol, cm^2/(mol K)
SIG_HO2_215  = (1.155e6, 1299.0)     # cm^2/mol, K

# surrogate columns that make up one target of each observable
OBS_COLUMNS = {'OH': ['OH'], 'H2O': ['H2O'], 'ABS215': ['H2O2', 'HO2']}
OBS_UNITS   = {'OH': 'mole fraction', 'H2O': 'mole fraction', 'ABS215': '1/cm',
               'H2O2': 'mole fraction', 'HO2': 'mole fraction'}


def sigma_h2o2_215(T):
    return SIG_H2O2_215[0] + SIG_H2O2_215[1] * T


def sigma_ho2_215(T):
    return SIG_HO2_215[0] * np.exp(-T / SIG_HO2_215[1])


def abs215_per_cm(x_h2o2, x_ho2, T, c_tot, s_ho2=1.0):
    """215 nm absorbance per cm of path; c_tot in mol/cm^3."""
    return c_tot * (sigma_h2o2_215(T) * x_h2o2 + s_ho2 * sigma_ho2_215(T) * x_ho2)


DATA_COLUMN = {'OH': ('[OH] ppm', 1e-6), 'H2O': ('[H2O] ppm', 1e-6), 'ABS215': ('Absorbance', 1.0)}


def read_record(path, obs):
    """Measured record of `obs`: sorted unique times [s] and values in model units
    (mole fraction, or absorbance), duplicate times averaged. Reads both the Hong
    files (header 'Time [ms]', '[OH] ppm') and headerless digitized ones."""
    import pandas as pd
    col, scale = DATA_COLUMN[obs]
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    try:                                   # headerless digitized file: time [ms], value
        float(df.columns[0])
        df = pd.read_csv(path, header=None, names=['Time [ms]', col], skipinitialspace=True)
    except ValueError:
        pass
    a = df.groupby('Time [ms]')[df.columns[1]].mean().reset_index()
    return a['Time [ms]'].values * 1e-3, a[df.columns[1]].values * scale, df


def observe(ph, name):
    """Value of a species mole fraction or of ABS215 [1/cm] for the Cantera phase
    `ph` (e.g. reactor.thermo)."""
    if name == 'ABS215':
        return abs215_per_cm(ph['H2O2'].X[0], ph['HO2'].X[0], ph.T, ph.density_mole * 1e-3)
    return ph[name].X[0]
