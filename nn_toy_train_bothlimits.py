#!/usr/bin/env python
# coding: utf-8
#
# nn_toy_train_bothlimits.py
# Joint surrogate (single NN), active reactions k1/k2/k5 + k4 (HO2+OH, Burke R4)
# (+ k3, HO2 + HO2 <=> H2O2 + O2, with the 'r3' argument)
# TRUE joint D-optimal target selection, FIXED time windows
#
# VARIANT 1 -- R1 (H2O2 (+M) <=> 2 OH (+M), falloff) is perturbed on BOTH limits:
#   A0 and A_inf get the same factor exp(x_A ln f), Ea0 and Ea_inf the same shift
#   x_Ea * SIGMA_E.  P_r and the Troe factor are unchanged, so k1 is scaled by the
#   same factor at every pressure (the falloff shape is not perturbed).
#   Previous scripts (nn_toy_*_train_final.py) scaled the low-pressure limit only.
#
# R3 (optional) -- the duplicate pair 19/20 gets the SAME A factor and the SAME Ea
#   shift on both branches (as R2), inputs 8 and 9.
#
# Observables -- each condition predicts what was measured there (LaGrotta et al.
#   2023 SI, Figs. S1-S9): OH and/or H2O mole fraction, or, for the 215 nm
#   absorbance of Kappel et al., the H2O2 and HO2 mole fractions at each target time
#   (absorbance.py); the optimizer forms the absorbance with its cross sections.
#
# usage:   python nn_toy_train_bothlimits.py 1192            (R1, R2, R5, R4)
#          python nn_toy_train_bothlimits.py 1398 r3         (R1, R2, R5, R4, R3)
#          conditions: see CONDITIONS below


import os, sys, time, copy
import cantera as ct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.stats import qmc, norm
from r1_falloff import R1Falloff
from absorbance import observe, OBS_UNITS, OBS_COLUMNS

R1_SCHEME = 'both_limits'   # recorded in the result file; optimize_r1_variants.py reads it
WITH_R3   = len(sys.argv) > 2 and sys.argv[2].lower() == 'r3'

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Mechanism & nominal rate constants ────────────────────────────────────────

YAML_FILE = 'chem_cti_toy_model_ogog.yaml'
mol_units = ct.UnitSystem({
    "length": "cm", "mass": "g", "time": "s",
    "quantity": "mol", "pressure": "dyn / cm^2", "energy": "erg",
    "temperature": "K", "current": "A", "activation-energy": "cal / mol"})

IDX_R1 = 21   # H2O2(+M) <=> OH + OH (+M)  — falloff   (k1)
# IDX_R2 = 25
IDX_R2_ALL = [25, 26]   # H2O2 + OH <=> HO2 + H2O    — Arrhenius (k2)
IDX_R3_ALL = [19, 20]   # HO2 + HO2 <=> H2O2 + O2    — Arrhenius (k3), duplicate
IDX_R5 = 4    # 2 OH <=> H2O + O           — Arrhenius (k5)
IDX_R4 = 18   # HO2 + OH <=> H2O + O2      — Arrhenius (k4, Burke R4)

_gas_nom = ct.Solution(YAML_FILE)
NOMINAL_A_R1      = _gas_nom.reaction(IDX_R1).rate.low_rate.pre_exponential_factor
NOMINAL_B_R1      = _gas_nom.reaction(IDX_R1).rate.low_rate.temperature_exponent
NOMINAL_EA_R1_si  = _gas_nom.reaction(IDX_R1).rate.low_rate.activation_energy
NOMINAL_EA_R1_cal = mol_units.convert_activation_energy_to(
    f"{NOMINAL_EA_R1_si} J/kmol", "cal / mol")
# NOMINAL_A_R2      = _gas_nom.reaction(IDX_R2).rate.pre_exponential_factor
# NOMINAL_B_R2      = _gas_nom.reaction(IDX_R2).rate.temperature_exponent
# NOMINAL_EA_R2_si  = _gas_nom.reaction(IDX_R2).rate.activation_energy
# NOMINAL_EA_R2_cal = mol_units.convert_activation_energy_to(
#     f"{NOMINAL_EA_R2_si} J/kmol", "cal / mol")
NOMINAL_R2 = [(i,
               _gas_nom.reaction(i).rate.pre_exponential_factor,
               _gas_nom.reaction(i).rate.temperature_exponent,
               mol_units.convert_activation_energy_to(
                   f"{_gas_nom.reaction(i).rate.activation_energy} J/kmol", "cal / mol"))
              for i in IDX_R2_ALL]
NOMINAL_R3 = [(i,
               _gas_nom.reaction(i).rate.pre_exponential_factor,
               _gas_nom.reaction(i).rate.temperature_exponent,
               mol_units.convert_activation_energy_to(
                   f"{_gas_nom.reaction(i).rate.activation_energy} J/kmol", "cal / mol"))
              for i in IDX_R3_ALL]
NOMINAL_A_R5      = _gas_nom.reaction(IDX_R5).rate.pre_exponential_factor
NOMINAL_B_R5      = _gas_nom.reaction(IDX_R5).rate.temperature_exponent
NOMINAL_EA_R5_si  = _gas_nom.reaction(IDX_R5).rate.activation_energy
NOMINAL_EA_R5_cal = mol_units.convert_activation_energy_to(
    f"{NOMINAL_EA_R5_si} J/kmol", "cal / mol")
NOMINAL_A_R4      = _gas_nom.reaction(IDX_R4).rate.pre_exponential_factor
NOMINAL_B_R4      = _gas_nom.reaction(IDX_R4).rate.temperature_exponent   # 0.0
NOMINAL_EA_R4_si  = _gas_nom.reaction(IDX_R4).rate.activation_energy
NOMINAL_EA_R4_cal = mol_units.convert_activation_energy_to(
    f"{NOMINAL_EA_R4_si} J/kmol", "cal / mol")
del _gas_nom

# R1 is rewritten from these stored nominal values (both limits) on every call
R1 = R1Falloff(YAML_FILE, IDX_R1)
_hi = R1.high
print(f'R1 (k1): {R1.equation}   both limits perturbed together')
print(f'         low  A={NOMINAL_A_R1:.3e}  Ea={NOMINAL_EA_R1_cal:.0f} cal/mol')
print(f'         high A={_hi[0]:.3e}  b={_hi[1]:+.2f}  Ea={_hi[2]/4184.0:.0f} cal/mol')
# print(f'R2 (k2): A={NOMINAL_A_R2:.3e}  b={NOMINAL_B_R2:+.2f}  Ea={NOMINAL_EA_R2_cal:.0f} cal/mol')
for (_i, _A0, _b0, _Ea0_cal) in NOMINAL_R2:
    print(f'R2 (k2) rxn {_i}: A={_A0:.3e}  b={_b0:+.2f}  Ea={_Ea0_cal:.0f} cal/mol')
if WITH_R3:
    for (_i, _A0, _b0, _Ea0_cal) in NOMINAL_R3:
        print(f'R3 (k3) rxn {_i}: A={_A0:.3e}  b={_b0:+.2f}  Ea={_Ea0_cal:.0f} cal/mol')
print(f'R5 (k5): A={NOMINAL_A_R5:.3e}  b={NOMINAL_B_R5:+.2f}  Ea={NOMINAL_EA_R5_cal:.0f} cal/mol')
print(f'R4 (k4): A={NOMINAL_A_R4:.3e}  b={NOMINAL_B_R4:+.2f}  Ea={NOMINAL_EA_R4_cal:.0f} cal/mol')

# ── Parameter normalization ───────────────────────────────────────────────────

LN_F        = np.log(10)        # A = A0 * exp(x * LN_F)
SIGMA_E     = 5000.0            # Ea(cal/mol) = Ea0 + x * SIGMA_E
PARAM_NAMES = ['lnA_R1', 'Ea_R1', 'lnA_R2', 'Ea_R2', 'lnA_R5', 'Ea_R5', 'lnA_R4', 'Ea_R4']
if WITH_R3:
    PARAM_NAMES += ['lnA_R3', 'Ea_R3']   # inputs 8, 9
INPUT_DIM   = len(PARAM_NAMES)

# ── Multi-scale Sobol sampling ────────────────────────────────────────────────

TOTAL_SAMPLES = 40000
SIGMA_LIST    = (0.1, 0.3, 0.5)
RATIO_LIST    = (1/4, 1/4, 1/2)

# ── Shock-tube condition (from the command line) ─────────────────────────────
# ppm: initial mole fractions in ppm, the rest is AR (nominal column of the
# LaGrotta et al. 2023 SI table).  obs: {observable: (number of targets, end of
# the target window [s])}; the window ends where the experimental record ends.
# 1192 / 1398: Hong et al. 2010 OH + H2O (hong_*.csv).  1057: Hong et al. 2009
# H2O.  1880: Hong et al. 2010 OH (H2O/O2/H mixture).  968 .. 1218: Kappel et al.
# 2002 absorbance at 215 nm.

CONDITIONS = {
    '1192': dict(T=1192, P_atm=1.95,  ppm=dict(H2O2=2220, H2O=1360, O2=680), obs=dict(OH=(4, 1e-3), H2O=(3, 1e-3))),
    '1398': dict(T=1398, P_atm=1.91,  ppm=dict(H2O2=2540, H2O=1234, O2=617), obs=dict(OH=(4, 1e-4), H2O=(3, 1e-4))),
    '1057': dict(T=1057, P_atm=1.83,  ppm=dict(H2O2=860,  H2O=660,  O2=330), obs=dict(H2O=(7, 6e-3))),     # T3
    '1880': dict(T=1880, P_atm=1.74,  ppm=dict(H2O=13000, O2=9900,  H=0.7),  obs=dict(OH=(7, 1e-3)),       # T4
                 oh_min_ppm=5.0),
    '968':  dict(T=968,  P_atm=3.59,  ppm=dict(H2O2=3140), obs=dict(ABS215=(7, 1.6e-3))),                  # T5
    '1015': dict(T=1015, P_atm=3.298, ppm=dict(H2O2=3260), obs=dict(ABS215=(7, 1.6e-3))),                  # T6
    '1079': dict(T=1079, P_atm=3.444, ppm=dict(H2O2=1950), obs=dict(ABS215=(7, 1.6e-3))),                  # T7
    '1132': dict(T=1132, P_atm=3.55,  ppm=dict(H2O2=1360), obs=dict(ABS215=(7, 0.6e-3))),                  # T8 (near noise level after 0.6 ms)
    '1218': dict(T=1218, P_atm=3.038, ppm=dict(H2O2=1710), obs=dict(ABS215=(7, 0.3e-3))),                  # T9 (noise level after 0.3 ms)
}
COND = sys.argv[1] if len(sys.argv) > 1 else '1192'
if COND not in CONDITIONS:
    sys.exit(f'unknown condition {COND!r}; choose from {list(CONDITIONS)}')
_c = CONDITIONS[COND]
T_INITIAL = _c['T']
P_INITIAL = _c['P_atm'] * ct.one_atm
INITIAL_X = {s: v * 1e-6 for s, v in _c['ppm'].items()}
INITIAL_X['AR'] = 1.0 - sum(INITIAL_X.values())
OBS       = list(_c['obs'])                          # observables, in output-column order
N_OBS     = {o: _c['obs'][o][0] for o in OBS}
T_MAX_OBS = {o: _c['obs'][o][1] for o in OBS}
print(f'\nCondition {COND}: T = {T_INITIAL} K, P = {_c["P_atm"]} atm, X = {INITIAL_X}')
print(f'  observables: ' + ', '.join(f'{o} x{N_OBS[o]} [{OBS_UNITS[o]}]' for o in OBS))

# Nominal grid for target selection: 1 ms (or the longest target window) in
# 10000 steps -- 0.1 us resolution for the 1192 / 1398 K conditions
N_FINE  = 10000
DT_FINE = max(1e-3, max(T_MAX_OBS.values())) / N_FINE
T_FINE  = np.linspace(DT_FINE, DT_FINE * N_FINE, N_FINE)

T_MIN_TARGET = 1e-6       # 1 µs
DT_MIN_OBS   = {o: 0.05 * (T_MAX_OBS[o] - T_MIN_TARGET) for o in OBS}

# ── NN / training hyper-parameters ───────────────────────────────────────────

HIDDEN_DIM     = 32
lr_init        = 0.03
TRAIN_FRAC     = 0.80
VAL_FRAC       = 0.10
EPOCHS         = 5000
BATCH_SIZE     = 1024
val_check      = 10
lr_check       = 200
LR_FACTOR      = 0.5
LR_MIN         = 1e-6
wd_init        = 1e-6
wd_min, wd_max = 1e-8, 1e-4
wd_gap_high    = 1.10
wd_gap_low     = 1.02

# Per-observable loss weight
WEIGHT_OBS = {o: 1.0 for o in OBS}

RUN_TAG         = 'bothlim_r3' if WITH_R3 else 'bothlim'
CHECKPOINT_PATH = f'ckpt_{COND}k_train_{RUN_TAG}.pt'
RESULT_PATH     = f'result_{COND}k_train_{RUN_TAG}.pt'
# smoke-test overrides (environment), e.g.  NN_SAMPLES=300 NN_EPOCHS=20
if os.environ.get('NN_SAMPLES'):
    TOTAL_SAMPLES = int(os.environ['NN_SAMPLES'])
if os.environ.get('NN_EPOCHS'):
    EPOCHS = int(os.environ['NN_EPOCHS'])
if os.environ.get('NN_TAG'):
    CHECKPOINT_PATH = CHECKPOINT_PATH.replace('.pt', f"_{os.environ['NN_TAG']}.pt")
    RESULT_PATH     = RESULT_PATH.replace('.pt', f"_{os.environ['NN_TAG']}.pt")
SIGMA_REQS      = {0.1: (0.01, 0.02), 0.3: (0.02, 0.05), 0.5: (0.03, 0.10)}
LOG_EPS         = 1e-12

UNIT_NORM      = True    # select targets by sensitivity DIRECTION, not magnitude
MAG_FLOOR_FRAC  = 0.1     # drop candidates with ||S|| < frac*max||S|| (0 = no floor)

# ── Nominal profiles & JOINT target-time selection ────────────────────────────

def _run_profiles(gas, t_grid):
    """All observables of this condition on t_grid, one reactor run."""
    gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
    reactor = ct.IdealGasConstPressureReactor(gas, energy='on')
    net     = ct.ReactorNet([reactor])
    out = {o: np.empty(len(t_grid)) for o in OBS}
    for k, t in enumerate(t_grid):
        net.advance(t)
        for o in OBS:
            out[o][k] = observe(reactor.thermo, o)
    return out


print(f'\nRunning nominal profiles ({T_FINE[-1]*1e3:.1f} ms, const-P) ...')
_nom = _run_profiles(ct.Solution(YAML_FILE), T_FINE)
for o in OBS:
    _i_pk = int(np.argmax(_nom[o]))
    print(f'  {o}: start {_nom[o][0]:.4g}  peak {_nom[o][_i_pk]:.4g} at {T_FINE[_i_pk]*1e3:.4f} ms  '
          f'end {_nom[o][-1]:.4g}  [{OBS_UNITS[o]}]')

# Measurability floor of the target candidates: OH >= 30 ppm (oh_min_ppm), an
# absorbance >= 15 % of its peak inside the window (later the record is at the
# noise level, see Figs. S8/S9), no floor for H2O
def _floor(o):
    if o == 'OH':
        return _c.get('oh_min_ppm', 30.0) * 1e-6
    if o.startswith('ABS'):
        w = (T_FINE >= T_MIN_TARGET) & (T_FINE <= T_MAX_OBS[o])
        return 0.15 * float(_nom[o][w].max())
    return -np.inf
FLOOR = {o: _floor(o) for o in OBS}

# ── FIXED TIME WINDOWING (match train3) ─────────────────────────────────────────
# Time windows are fixed to prevent picking targets deep in the decay (for OH)
# or past the experimental data horizon.
for o in OBS:
    print(f'  {o} time window: {T_MIN_TARGET*1e6:.1f} µs to {T_MAX_OBS[o]*1e3:.4f} ms')


def _scaled_gas(mult=None):
    """Mechanism with the A factor of the listed reaction(s) multiplied by f.
    R1 is scaled on both limits, matching how it is perturbed in run_single."""
    g = ct.Solution(YAML_FILE)
    if mult:
        idxs, f = mult
        for i in ([idxs] if isinstance(idxs, int) else idxs):
            if i == IDX_R1:
                R1.apply_rate(g, fA=f)
                continue
            r = g.reaction(i)
            r.rate = ct.Arrhenius(r.rate.pre_exponential_factor*f, r.rate.temperature_exponent, r.rate.activation_energy)
            g.modify_reaction(i, r)
    return g


print('\nComputing sensitivities ...')
d = 0.01
SENS_RXNS = [IDX_R1, IDX_R2_ALL, IDX_R5, IDX_R4] + ([IDX_R3_ALL] if WITH_R3 else [])
SENS_TAG  = 'k1/k2/k5/k4' + ('/k3' if WITH_R3 else '')
_pert = [_run_profiles(_scaled_gas((r, 1+d)), T_FINE) for r in SENS_RXNS]
S = {o: np.vstack([(p[o] - _nom[o]) / _nom[o] / d for p in _pert]).T for o in OBS}   # (N_FINE, 4 or 5)


def select_targets_joint(t_grid, S_mats, n_targets,
                         unit_norm=UNIT_NORM, mag_floor_frac=MAG_FLOOR_FRAC):
    """
    TRUE joint D-optimal selection: every candidate (any observable) is scored by
    det(S_joint^T S_joint) where S_joint stacks ALL currently chosen rows of ALL
    observables plus the candidate.
    """
    cand, S_use, norms = {}, {}, {}
    for o in OBS:
        win  = np.where((t_grid >= T_MIN_TARGET) & (t_grid <= T_MAX_OBS[o]))[0]
        meas = np.where(_nom[o] >= FLOOR[o])[0]
        cand[o] = np.intersect1d(win, meas)
        # Row magnitudes: used for the seed (always), the optional magnitude floor,
        # and to unit-normalize rows when unit_norm=True (select by DIRECTION diversity
        # rather than sensitivity magnitude). With unit_norm=False and mag_floor_frac=0
        # this block is identical to the original magnitude-based D-optimal selection.
        norms[o] = np.linalg.norm(S_mats[o], axis=1)
        if mag_floor_frac > 0 and len(cand[o]):
            cand[o] = cand[o][norms[o][cand[o]] >= mag_floor_frac * norms[o][cand[o]].max()]
        S_use[o] = (S_mats[o] / np.clip(norms[o][:, None], 1e-30, None)) if unit_norm else S_mats[o]

    chosen = {o: [] for o in OBS}
    for o in OBS:
        if len(cand[o]) > 0:
            chosen[o].append(int(cand[o][np.argmax(norms[o][cand[o]])]))   # seed: max RAW magnitude

    def _joint_det(extra_o=None, extra=None):
        S_j = np.vstack([S_use[o][chosen[o] + ([extra] if o == extra_o else [])] for o in OBS])
        return np.linalg.det(S_j.T @ S_j)

    def _too_close(c, o):
        return any(abs(t_grid[c] - t_grid[j]) < DT_MIN_OBS[o] for j in chosen[o])

    while any(len(chosen[o]) < n_targets[o] for o in OBS):
        best_det, best = -np.inf, None
        for o in OBS:
            if len(chosen[o]) >= n_targets[o]:
                continue
            for c in cand[o]:
                if c in chosen[o] or _too_close(c, o):
                    continue
                v = _joint_det(o, c)
                if v > best_det:
                    best_det, best = v, (o, c)
        if best is None:
            break                     # no admissible candidates left
        chosen[best[0]].append(best[1])

    return {o: sorted(chosen[o]) for o in OBS}


_idx = select_targets_joint(T_FINE, S, N_OBS)
TARGET_TIMES = {o: T_FINE[_idx[o]] for o in OBS}
assert all(len(TARGET_TIMES[o]) == N_OBS[o] for o in OBS), (
    f'target selection returned {[len(TARGET_TIMES[o]) for o in OBS]} targets, '
    f'expected {[N_OBS[o] for o in OBS]} -- the net would train on garbage columns.')

LABELS = {o: [f'{o} D-opt @ {t*1e3:.4f} ms' for t in TARGET_TIMES[o]] for o in OBS}
for o in OBS:
    print(f'\n{o} target probes (window {T_MIN_TARGET*1e6:.1f} µs .. {T_MAX_OBS[o]*1e3:.4f} ms, '
          f'TRUE joint D-optimal, {SENS_TAG}):')
    for j, (lbl, t) in enumerate(zip(LABELS[o], TARGET_TIMES[o])):
        print(f'  {j+1}. {lbl:<34}  {o} ~ {np.interp(t, T_FINE, _nom[o]):.4g} {OBS_UNITS[o]}')

# Output columns: observable blocks in OBS order; inside a block, the species of
# OBS_COLUMNS (ABS215 -> H2O2, HO2), each in time order
_cols       = [(o, sp, t) for o in OBS for sp in OBS_COLUMNS[o] for t in TARGET_TIMES[o]]
COL_OBS     = [c[0] for c in _cols]
COL_SPECIES = [c[1] for c in _cols]
COL_TIMES   = np.array([c[2] for c in _cols])
COL_LABELS  = [f'{sp} @ {t*1e3:.4f} ms' + (f' ({o})' if o != sp else '') for o, sp, t in _cols]
N_TARGETS   = len(_cols)
COL_SLICE = {}
_s = 0
for o in OBS:
    _n = N_OBS[o] * len(OBS_COLUMNS[o])
    COL_SLICE[o] = slice(_s, _s + _n); _s += _n

# Nominal temperature and total concentration [mol/cm^3] at each target time;
# the optimizer needs them to form the absorbance from the surrogate's H2O2 / HO2
def _state_at(times):
    g = ct.Solution(YAML_FILE); g.TPX = T_INITIAL, P_INITIAL, INITIAL_X
    rr = ct.IdealGasConstPressureReactor(g, energy='on'); net = ct.ReactorNet([rr])
    Ts, cs = [], []
    for t in times:
        net.advance(t); Ts.append(rr.thermo.T); cs.append(rr.thermo.density_mole * 1e-3)
    return np.array(Ts), np.array(cs)
NOM_T, NOM_CTOT = {}, {}
for o in OBS:
    NOM_T[o], NOM_CTOT[o] = _state_at(TARGET_TIMES[o])

# ── Advance order: merge target times chronologically ─────────────────────────

_sort_idx       = np.argsort(COL_TIMES, kind='stable')
_advance_times  = COL_TIMES[_sort_idx]      # chronological advance times
_advance_obs    = [COL_SPECIES[i] for i in _sort_idx]   # which quantity to record
_advance_out_idx = _sort_idx                # output column for each advance step


# ── Sobol sampling ────────────────────────────────────────────────────────────

def multiscale_sobol(n_total, sigmas, ratios, d=INPUT_DIM, seed=SEED):
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    base    = sampler.random(n_total)
    chunks, labels, idx = [], [], 0
    for i, (s, r) in enumerate(zip(sigmas, ratios)):
        count = n_total - idx if i == len(sigmas) - 1 else int(round(n_total * r))
        u     = base[idx: idx + count]
        p_lo  = norm.cdf(-1.0, loc=0, scale=s)
        p_hi  = norm.cdf( 1.0, loc=0, scale=s)
        chunk = norm.ppf(p_lo + u * (p_hi - p_lo), loc=0, scale=s)
        chunks.append(chunk); labels.append(np.full(count, s)); idx += count
    return np.vstack(chunks), np.concatenate(labels)

X_samples, L_samples = multiscale_sobol(TOTAL_SAMPLES, SIGMA_LIST, RATIO_LIST)


# ── Cantera simulation — all target columns, active reactions perturbed ───────

def run_single(x_vec):
    try:
        gas = ct.Solution(YAML_FILE)
        # R1 (k1) — falloff, BOTH limits: same A factor, same Ea shift
        R1.apply_rate(gas, fA=np.exp(x_vec[0] * LN_F), dEa_cal=x_vec[1] * SIGMA_E)
        # R2 (k2) — simple Arrhenius
        # new_A_R2  = NOMINAL_A_R2 * np.exp(x_vec[2] * LN_F)
        # new_Ea_R2 = (NOMINAL_EA_R2_cal + x_vec[3] * SIGMA_E) * 4184.0
        # rxn2 = gas.reaction(IDX_R2)
        # rxn2.rate = ct.Arrhenius(new_A_R2, NOMINAL_B_R2, new_Ea_R2)
        # gas.modify_reaction(IDX_R2, rxn2)
        # # R2 (k2) — simple Arrhenius [previous version: NOMINAL_R2 was never defined
        # # since its definition was commented out above, so this raised NameError on
        # # every call and silently failed all simulations via the except Exception below]
        for (i, A0, b0, Ea0_cal) in NOMINAL_R2:
            rxn = gas.reaction(i)
            rxn.rate = ct.Arrhenius(A0 * np.exp(x_vec[2] * LN_F), b0, (Ea0_cal + x_vec[3] * SIGMA_E) * 4184.0)
            gas.modify_reaction(i, rxn)
        # R5 (k5) — simple Arrhenius (b=2.42 preserved, not optimized)
        new_A_R5  = NOMINAL_A_R5 * np.exp(x_vec[4] * LN_F)
        new_Ea_R5 = (NOMINAL_EA_R5_cal + x_vec[5] * SIGMA_E) * 4184.0
        rxn5 = gas.reaction(IDX_R5)
        rxn5.rate = ct.Arrhenius(new_A_R5, NOMINAL_B_R5, new_Ea_R5)
        gas.modify_reaction(IDX_R5, rxn5)
        # R4 (k4) — simple Arrhenius (b=0 preserved, not optimized; Burke R4)
        new_A_R4  = NOMINAL_A_R4 * np.exp(x_vec[6] * LN_F)
        new_Ea_R4 = (NOMINAL_EA_R4_cal + x_vec[7] * SIGMA_E) * 4184.0
        rxn4 = gas.reaction(IDX_R4)
        rxn4.rate = ct.Arrhenius(new_A_R4, NOMINAL_B_R4, new_Ea_R4)
        gas.modify_reaction(IDX_R4, rxn4)
        # R3 (k3) — duplicate Arrhenius pair, same A factor and same Ea shift on both
        if WITH_R3:
            for (i, A0, b0, Ea0_cal) in NOMINAL_R3:
                rxn = gas.reaction(i)
                rxn.rate = ct.Arrhenius(A0 * np.exp(x_vec[8] * LN_F), b0, (Ea0_cal + x_vec[9] * SIGMA_E) * 4184.0)
                gas.modify_reaction(i, rxn)

        gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
        reactor  = ct.IdealGasConstPressureReactor(gas, energy='on')
        net      = ct.ReactorNet([reactor])

        y_out = np.full(N_TARGETS, np.nan)   # nan, not np.empty: unfilled slots must fail loudly below

        # Single chronological pass through all target times
        for k in range(len(_advance_times)):
            net.advance(_advance_times[k])
            y_out[_advance_out_idx[k]] = observe(reactor.thermo, _advance_obs[k])

        if not np.isfinite(y_out).all():
            return False, None    # a target slot was never filled
        return True, y_out
    except Exception as ex:
        # report the first failure instead of failing silently (a NameError here once
        # failed every simulation without a trace)
        global _FIRST_ERR
        if _FIRST_ERR is None:
            _FIRST_ERR = repr(ex)
            print(f'  first failed simulation: {_FIRST_ERR}')
        return False, None


_FIRST_ERR = None


# ── Checkpoint / run simulations ──────────────────────────────────────────────

def _ckpt_dict(last_index):
    return {'last_index': last_index, 'y_list': y_list, 'x_list': x_list,
            'l_list': l_list, 'n_failed': n_failed,
            'target_obs': COL_OBS, 'target_species': COL_SPECIES,
            'target_times': COL_TIMES, 'labels': COL_LABELS}

if os.path.exists(CHECKPOINT_PATH):
    ckpt     = torch.load(CHECKPOINT_PATH, weights_only=False)
    raw_y    = ckpt['y_list']; raw_x = ckpt['x_list']
    raw_l    = ckpt.get('l_list', [None] * len(raw_y))
    n_failed = ckpt.get('n_failed', 0)
    valid    = [i for i, y in enumerate(raw_y) if np.asarray(y).shape == (N_TARGETS,)]
    y_list   = [raw_y[i] for i in valid]
    x_list   = [raw_x[i] for i in valid]
    l_list   = [raw_l[i] for i in valid]
    start_idx = ckpt['last_index'] + 1
    print(f'\nResuming idx {start_idx}/{TOTAL_SAMPLES}: '
          f'{len(y_list)} valid, {len(raw_y)-len(valid)} dropped, {n_failed} failed')
else:
    y_list, x_list, l_list, n_failed, start_idx = [], [], [], 0, 0
    print(f'\nStarting fresh: {TOTAL_SAMPLES} simulations')

t0 = time.time()
for i in range(start_idx, TOTAL_SAMPLES):
    ok, y_tgt = run_single(X_samples[i])
    if ok:
        y_list.append(y_tgt); x_list.append(X_samples[i]); l_list.append(L_samples[i])
    else:
        n_failed += 1
    if (i + 1) % 2000 == 0:
        rate = (i + 1 - start_idx) / max(time.time() - t0, 1e-9)
        print(f'  {i+1}/{TOTAL_SAMPLES}  {rate:.1f} sim/s  failed={n_failed}')
        torch.save(_ckpt_dict(i), CHECKPOINT_PATH)

torch.save(_ckpt_dict(TOTAL_SAMPLES - 1), CHECKPOINT_PATH)

X_raw = np.asarray(x_list); Y_raw = np.asarray(y_list); L_raw = np.asarray(l_list)
print(f'\nShapes: X={X_raw.shape}  Y={Y_raw.shape}  (failed={n_failed})')
for o in OBS:
    print(f'  {o} outputs: columns {COL_SLICE[o].start}..{COL_SLICE[o].stop-1}')


# ── Log transform & split ─────────────────────────────────────────────────────

Y_log = np.log(Y_raw + LOG_EPS)
X_t   = torch.tensor(X_raw, dtype=torch.float32)
Y_t   = torch.tensor(Y_log, dtype=torch.float32)
L_t   = torch.tensor(L_raw, dtype=torch.float32)

n_total = len(X_raw)
n_train = int(TRAIN_FRAC * n_total)
n_val   = int(VAL_FRAC   * n_total)
n_test  = n_total - n_train - n_val
full_ds = TensorDataset(X_t, Y_t, L_t)
train_ds, val_ds, test_ds = random_split(
    full_ds, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
print(f'Split: train={n_train}  val={n_val}  test={n_test}')


# ── Architecture — joint INPUT_DIM in -> N_TARGETS out ────────────────────────

class SurrogateNN(nn.Module):
    def __init__(self, hidden=HIDDEN_DIM, n_out=N_TARGETS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
        )
    def forward(self, x):
        return self.net(x)


# ── Training loop ──────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
model     = SurrogateNN()
optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=wd_init)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR,
    patience=lr_check // val_check, min_lr=LR_MIN)

class JointRelativeErrorLoss(nn.Module):
    def forward(self, pred, target):
        pred_raw = torch.exp(pred)
        target_raw = torch.exp(target)
        rel_err = torch.abs(pred_raw - target_raw) / (torch.abs(target_raw) + 1e-30)
        # Compute loss separately for each observable
        return sum(WEIGHT_OBS[o] * torch.mean(rel_err[:, COL_SLICE[o]]) for o in OBS)

criterion = JointRelativeErrorLoss()

train_losses, val_losses, val_epochs = [], [], []
best_val   = float('inf')
best_state = copy.deepcopy(model.state_dict())

def _pass(loader, train_mode):
    total = 0.0
    for xb, yb, _ in loader:
        if train_mode:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
        else:
            with torch.no_grad():
                loss = criterion(model(xb), yb)
        total += loss.item() * len(xb)
    return total / len(loader.dataset)

print(f'\nTraining ({T_INITIAL} K joint {"+".join(OBS)}, k1/k2/k5/k4{"/k3" if WITH_R3 else ""}, R1 {R1_SCHEME}, '
      + ' / '.join(f'{N_OBS[o]}-pt {o} {T_MIN_TARGET*1e6:.0f}us-{T_MAX_OBS[o]*1e3:.4f}ms' for o in OBS)
      + ' [fixed, TRUE JOINT D-OPT]) ...')
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    tl = _pass(train_loader, train_mode=True)
    train_losses.append(tl)

    if (epoch + 1) % val_check == 0 or epoch == 0:
        model.eval()
        vl = _pass(val_loader, train_mode=False)
        val_losses.append(vl); val_epochs.append(epoch + 1)
        scheduler.step(vl)

        gap    = vl / max(tl, 1e-12)
        cur_wd = optimizer.param_groups[0]['weight_decay']
        new_wd = (min(cur_wd * 2.0, wd_max) if gap > wd_gap_high
                  else max(cur_wd * 0.5, wd_min) if gap < wd_gap_low
                  else cur_wd)
        for g in optimizer.param_groups:
            g['weight_decay'] = new_wd

        if vl < best_val:
            best_val = vl; best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 500 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'  ep {epoch+1:5d} | train {tl:.4e} | val {vl:.4e} | '
                  f'gap {gap:.3f} | wd {new_wd:.1e} | lr {lr_now:.1e}')

model.load_state_dict(best_state)
print(f'Done in {(time.time()-t0)/60:.1f} min.  Best val: {best_val:.4e}')


# ── Zhang Table 1 evaluation ──────────────────────────────────────────────────

model.eval()
X_te   = torch.tensor(X_raw[test_ds.indices], dtype=torch.float32)
Y_true = Y_raw[test_ds.indices]
L_test = L_raw[test_ds.indices]
with torch.no_grad():
    Y_pred = np.exp(model(X_te).numpy())
rel_err = np.abs(Y_pred - Y_true) / (np.abs(Y_true) + 1e-30)

print(f'\n{"="*72}')
print(f'  {T_INITIAL} K Joint ({" + ".join(OBS)}), {INPUT_DIM} inputs (k1 both limits,k2,k5,k4{",k3" if WITH_R3 else ""}, TRUE JOINT D-OPT)  |  Zhang Table 1')
print(f'{"="*72}')

all_pass = True
for o in OBS:
    print(f'\n{o} accuracy:')
    for k, sigma in enumerate(SIGMA_LIST, 1):
        sel    = (L_test == sigma)
        for sp in OBS_COLUMNS[o]:
            cols   = [j for j in range(COL_SLICE[o].start, COL_SLICE[o].stop) if COL_SPECIES[j] == sp]
            r      = rel_err[sel][:, cols]
            e_mean = r.mean(); e_95 = np.percentile(r, 95)
            req    = SIGMA_REQS[sigma]
            pm, p9 = e_mean <= req[0], e_95 <= req[1]
            ok     = 'PASS' if (pm and p9) else 'FAIL'
            all_pass = all_pass and pm and p9
            print(f'  Set {k}  sigma={sigma:<4} N={sel.sum():<6} '
                  + (f'{sp:<5}' if sp != o else '')
                  + f'mean={e_mean*100:6.2f}% (<= {req[0]*100:.0f}%)  '
                  f'p95={e_95*100:6.2f}% (<= {req[1]*100:.0f}%)  {ok}')

print(f'{"="*72}')
print('MEETS' if all_pass else 'does NOT meet', 'all Zhang Table 1 requirements.')


# ── Save ──────────────────────────────────────────────────────────────────────

torch.save({
    'model_state':      model.state_dict(),
    'test_indices':     list(test_ds.indices),
    'target_obs':       COL_OBS,          # observable of each output column
    'target_species':   COL_SPECIES,      # quantity of each output column (mole fraction)
    'target_times':     COL_TIMES,        # time of each output column [s]
    'labels':           COL_LABELS,
    'obs_times':        TARGET_TIMES,     # {observable: target times}
    'obs_T_nom':        NOM_T,            # {observable: nominal T at those times}
    'obs_ctot_nom':     NOM_CTOT,         # {observable: nominal c = P/RT [mol/cm^3]}
    # OH / H2O blocks as in the earlier result files (empty when not observed)
    'oh_target_times':  TARGET_TIMES.get('OH',  np.array([])),
    'h2o_target_times': TARGET_TIMES.get('H2O', np.array([])),
    'n_oh_targets':     N_OBS.get('OH', 0),
    'n_h2o_targets':    N_OBS.get('H2O', 0),
    'hidden_dim':       HIDDEN_DIM,
    'input_dim':        INPUT_DIM,
    'param_names':      PARAM_NAMES,
    'idx_r1':            IDX_R1,
    # 'idx_r2':            IDX_R2,
    'idx_r2_all':        IDX_R2_ALL,
    'idx_r5':            IDX_R5,
    'idx_r4':            IDX_R4,
    **({'idx_r3_all': IDX_R3_ALL} if WITH_R3 else {}),   # optimize_r1_variants.py adds R3 when present
    'nominal_b_r5':      NOMINAL_B_R5,   # needed downstream to rebuild R5's rate
    'nominal_b_r4':      NOMINAL_B_R4,   # needed downstream to rebuild R4's rate
    'ln_f':             LN_F,
    'sigma_e':          SIGMA_E,
    'r1_scheme':        R1_SCHEME,        # 'both_limits'; older files lack it = low limit only
    'eps':              None,             # no efficiency knob in this variant
    'condition':        dict(key=COND, T=T_INITIAL, P_atm=P_INITIAL / ct.one_atm, X=INITIAL_X),
    'cantera_version':  ct.__version__,
    'train_losses':     train_losses,
    'val_losses':       val_losses,
    'val_epochs':       val_epochs,
    'best_val':         best_val,
}, RESULT_PATH)
print(f'\nSaved: {RESULT_PATH}')
