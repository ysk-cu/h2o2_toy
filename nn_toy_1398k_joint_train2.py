#!/usr/bin/env python
# coding: utf-8
#
# nn_toy_1398k_joint_train2.py
# Joint OH + H2O surrogate (single NN, trained on both species together) with
# INDEPENDENT per-species D-optimal target selection and time-window constraints
# (1398 K). "Joint" here describes the trained model, not the target-selection
# algorithm -- see select_targets_joint()'s docstring for why competing OH and
# H2O for a shared target quota was dropped in favor of each species picking its
# own best points with the identical selection formula.
# Mixture: 22540 ppm H2O2 / 1234 ppm H2O / 617 ppm O2 / Ar
#
# KEY DIFFERENCES FROM _S:
#   1. Targets are selected per-species independently, using the SAME D-optimal
#      formula for both (see select_targets_joint() docstring for why an earlier
#      cross-species-competition version was replaced)
#   2. SPECIES-SPECIFIC time-window constraints: OH uses a tight early window
#      (peak/early-decay, avoids the ~34% structural-overshoot tail from missing
#      radical-pool chemistry); H2O uses a wider window that keeps its rise-to-
#      plateau anchor, since that's what constrains k1 via H2O2->H2O yield.
#      A single shared window (tried first) starved H2O of that anchor and let
#      k1 be set almost entirely by OH, causing OH peak overshoot + early H2O rise.
#   3. This prevents k2 from absorbing error, which was degrading H2O fit in joint inference
#
# Single NN: x[0..3] -> [N_OH_TARGETS OH probes | N_H2O_TARGETS H2O probes]
#
# Active parameters (identical to separate surrogates):
#   x[0]=lnA_R1  x[1]=Ea_R1  x[2]=lnA_R2  x[3]=Ea_R2

import os, time, copy
import cantera as ct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.stats import qmc, norm

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Mechanism & nominal rate constants ────────────────────────────────────────

YAML_FILE = 'chem_cti_toy_model_og.yaml'
mol_units = ct.UnitSystem({
    "length": "cm", "mass": "g", "time": "s",
    "quantity": "mol", "pressure": "dyn / cm^2", "energy": "erg",
    "temperature": "K", "current": "A", "activation-energy": "cal / mol"})

IDX_R1 = 21   # H2O2(+M) <=> OH + OH (+M)  — falloff   (k1)
IDX_R2 = 25   # H2O2 + OH <=> HO2 + H2O    — Arrhenius (k2)

_gas_nom = ct.Solution(YAML_FILE)
NOMINAL_A_R1      = _gas_nom.reaction(IDX_R1).rate.low_rate.pre_exponential_factor
NOMINAL_B_R1      = _gas_nom.reaction(IDX_R1).rate.low_rate.temperature_exponent
NOMINAL_EA_R1_si  = _gas_nom.reaction(IDX_R1).rate.low_rate.activation_energy
NOMINAL_EA_R1_cal = mol_units.convert_activation_energy_to(
    f"{NOMINAL_EA_R1_si} J/kmol", "cal / mol")
NOMINAL_A_R2      = _gas_nom.reaction(IDX_R2).rate.pre_exponential_factor
NOMINAL_B_R2      = _gas_nom.reaction(IDX_R2).rate.temperature_exponent
NOMINAL_EA_R2_si  = _gas_nom.reaction(IDX_R2).rate.activation_energy
NOMINAL_EA_R2_cal = mol_units.convert_activation_energy_to(
    f"{NOMINAL_EA_R2_si} J/kmol", "cal / mol")
del _gas_nom

print(f'R1 (k1): A={NOMINAL_A_R1:.3e}  Ea={NOMINAL_EA_R1_cal:.0f} cal/mol')
print(f'R2 (k2): A={NOMINAL_A_R2:.3e}  Ea={NOMINAL_EA_R2_cal:.0f} cal/mol')

# ── Parameter normalization ───────────────────────────────────────────────────

LN_F        = np.log(10)        # A = A0 * exp(x * LN_F)
SIGMA_E     = 5000.0            # Ea(cal/mol) = Ea0 + x * SIGMA_E
PARAM_NAMES = ['lnA_R1', 'Ea_R1', 'lnA_R2', 'Ea_R2']
INPUT_DIM   = 4

# ── Multi-scale Sobol sampling ────────────────────────────────────────────────

TOTAL_SAMPLES = 40000
SIGMA_LIST    = (0.1, 0.3, 0.5)
RATIO_LIST    = (1/6, 1/6, 2/3)

# ── Shock-tube condition ──────────────────────────────────────────────────────

T_INITIAL = 1398
P_INITIAL = 1.91 * ct.one_atm
INITIAL_X = {'H2O2': 2540e-6, 'H2O': 1234e-6, 'O2': 617e-6,
             'AR':   1.0 - (2540 + 1234 + 617) * 1e-6}

# Fine nominal grid for OH peak / rise detection (0.1 us resolution)
DT_FINE = 1e-7
N_FINE  = 10000
T_FINE  = np.linspace(DT_FINE, DT_FINE * N_FINE, N_FINE)   # 0.1 us ... 1 ms

# Coarser grid for H2O rise detection (0.1 us resolution)
DT_SIM  = 1e-7
N_STEPS = 10000
T_SIM   = np.linspace(DT_SIM, DT_SIM * N_STEPS, N_STEPS)   # 1 us ... 1 ms

# ── Time-window constraints for trustworthy target selection (species-specific) ──
# CONDITION-SPECIFIC: at 1398 K the Hong experimental time history (both OH and H2O)
# only extends to ~0.1 ms -- the reaction is far faster here (10x higher H2O2 loading,
# ~200 K hotter) than at 1192 K, where data ran out to ~1 ms. Both windows' t_max are
# capped at 0.1 ms because of this: a target time picked past the last real data point
# doesn't error downstream -- np.interp() in the inference notebook silently CLIPS to
# the last available value, fabricating a flat "observation" at whatever time you
# picked past the data range. VERIFY the exact cutoff against hong_1398K_h2o.csv /
# hong_1398K_oh.csv (optimize_1398k_joint_nn.ipynb now prints the true max time from
# each CSV) and tighten these further if 0.1 ms turns out to be optimistic.
#
# OH and H2O still carry their k1/k2 information in different sub-windows within that
# budget, so we keep them separate rather than collapsing to one shared window (see
# the 1192 K file for why a single shared window starves H2O of its rise/plateau
# anchor and lets k1 be set almost entirely by OH).
T_MIN_TARGET_OH = 1e-6    # 1 µs — NOTE: the 10 µs floor used at 1192 K (to dodge
                          # a documented time-zero disagreement between the two Hong
                          # OH traces there) is NOT reused here. At 1398 K the nominal
                          # OH peak lands at ~6-8 µs -- a 10 µs floor would exclude the
                          # peak entirely, forcing all 3 OH targets into decay-only
                          # territory and starving k1/peak-height of any information
                          # (this was diagnosed from the MAP peak undershooting nominal
                          # by ~14% with the old floor). The oh_min_ppm=30 concentration
                          # filter (applied via intersect1d below) still excludes any
                          # candidate before OH is actually measurable, so this floor
                          # only needs to guard against literal t~0 solver artifacts.
T_MAX_TARGET_OH = 1e-4    # 0.1 ms — already at the data ceiling

T_MIN_TARGET_H2O = 1e-6    # 1 µs — was 10 µs. H2O rises from ~1300 to ~2300 ppm
                           # within the FIRST 10 µs at this condition -- that steep-
                           # slope region carries real k1 information and was being
                           # excluded entirely, the same blind spot OH's old 10 µs
                           # floor had for its peak. Opening this up lets the D-optimal
                           # selector consider it instead of always missing it by
                           # construction. (Diagnosed after lowering T_MIN_TARGET_OH:
                           # OH ONLY and H2O ONLY solve()s individually fit well, but
                           # JOINT landed almost exactly on OH's preferred k1/k2 because
                           # OH's log-sensitivity is inherently larger than H2O's
                           # buffered-pool response -- giving H2O more/better real
                           # information is the fix, not reweighting SIG_LOG post-hoc.)

# T_MAX_TARGET_H2O is NOT a fixed constant -- see the H2O-95%-RISE RULE block right
# after the nominal H2O profile is computed below. A fixed absolute cutoff (tried:
# 1e-4, inherited from the OH data ceiling) let independent D-optimality reach for
# the plateau (~100 us) as a 3rd target, which improved raw parameter identifiability
# but visibly cost early-rise fit accuracy -- the plateau point pulls the 2-parameter
# fit toward matching a late-time value at the expense of the early-rise shape you
# actually care about. A hardcoded smaller number (e.g. 5e-5) would fix THIS condition
# but isn't universal: the reaction timescale compresses/expands by ~10x between
# 1192 K and 1398 K, so any fixed millisecond value needs re-tuning by hand at every
# new temperature. Defining the cutoff as a FRACTION of each condition's own rise
# timescale (specifically: the time to reach 95% of the total H2O rise) self-scales
# automatically -- validated against 1192 K, where the working T_MAX=0.4ms almost
# exactly equals that condition's own 95%-rise time (~0.374 ms).
T_MAX_H2O_RISE_FRAC = 0.95   # universal across conditions; only this fraction is tuned,
                             # not an absolute time

# ── NN / training hyper-parameters ───────────────────────────────────────────

HIDDEN_DIM     = 16            # larger for joint training
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

# Per-species loss weight
WEIGHT_OH  = 1.0
WEIGHT_H2O = 1.0

CHECKPOINT_PATH = 'ckpt_1398k_joint_train5.pt'
RESULT_PATH     = 'result_1398k_joint_train5.pt'
SIGMA_REQS      = {0.1: (0.01, 0.02), 0.3: (0.02, 0.05), 0.5: (0.03, 0.10)}
LOG_EPS         = 1e-12

N_OH_TARGETS  = 3
N_H2O_TARGETS = 3
N_TARGETS     = N_OH_TARGETS + N_H2O_TARGETS

# ── Nominal profiles & JOINT target-time selection ────────────────────────────

def _run_nominal(species_idx_list, t_grid):
    gas = ct.Solution(YAML_FILE)
    gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
    reactor = ct.IdealGasConstPressureReactor(gas, energy='on')
    net     = ct.ReactorNet([reactor])
    profiles = [np.empty(len(t_grid)) for _ in species_idx_list]
    for k, t in enumerate(t_grid):
        net.advance(t)
        for j, idx in enumerate(species_idx_list):
            profiles[j][k] = reactor.thermo.X[idx]
    return profiles


print('\nRunning nominal profiles (1 ms, const-P) ...')
_gas_tmp = ct.Solution(YAML_FILE)
_oh_idx  = _gas_tmp.species_index('OH')
_h2o_idx = _gas_tmp.species_index('H2O')
del _gas_tmp

# OH nominal on fine grid
[_nom_oh] = _run_nominal([_oh_idx], T_FINE)
_i_pk     = int(np.argmax(_nom_oh))
print(f'  OH_peak = {_nom_oh[_i_pk]*1e6:.1f} ppm  at t = {T_FINE[_i_pk]*1e3:.4f} ms')

# H2O nominal on coarser grid
[_nom_h2o] = _run_nominal([_h2o_idx], T_SIM)
_h2o_0     = float(_nom_h2o[0])
_h2o_inf   = float(_nom_h2o[-200:].mean())
print(f'  H2O initial = {_h2o_0*1e6:.1f} ppm   H2O plateau = {_h2o_inf*1e6:.1f} ppm')

# ── H2O-95%-RISE RULE: condition-adaptive T_MAX_TARGET_H2O ────────────────────
# Universal rule (see comment above): cap H2O targets at the time this condition's
# OWN nominal profile reaches T_MAX_H2O_RISE_FRAC of its total rise, instead of a
# hand-picked absolute time. Self-scales with each condition's kinetics.
_h2o_rise_frac = (_nom_h2o - _h2o_0) / (_h2o_inf - _h2o_0)
_i_h2o_95      = int(np.argmax(_h2o_rise_frac >= T_MAX_H2O_RISE_FRAC))
T_MAX_TARGET_H2O = float(T_SIM[_i_h2o_95])
print(f'  H2O {T_MAX_H2O_RISE_FRAC*100:.0f}%-rise time = {T_MAX_TARGET_H2O*1e3:.4f} ms '
      f'-> T_MAX_TARGET_H2O (was a fixed 1e-4 s / 0.1 ms)')


def _oh_profile(mult=None):
    g = ct.Solution(YAML_FILE)
    if mult:
        i, f = mult; r = g.reaction(i)
        if i == IDX_R1:
            lr = r.rate.low_rate
            r.rate.low_rate = ct.Arrhenius(lr.pre_exponential_factor*f, lr.temperature_exponent, lr.activation_energy)
        else:
            r.rate = ct.Arrhenius(r.rate.pre_exponential_factor*f, r.rate.temperature_exponent, r.rate.activation_energy)
        g.modify_reaction(i, r)
    g.TPX = T_INITIAL, P_INITIAL, INITIAL_X
    rr = ct.IdealGasConstPressureReactor(g, energy='on'); net = ct.ReactorNet([rr]); oh = g.species_index('OH')
    out = np.empty(N_FINE)
    for k in range(N_FINE): net.advance(T_FINE[k]); out[k] = rr.thermo.X[oh]
    return out


def _h2o_profile(mult=None):
    g = ct.Solution(YAML_FILE)
    if mult:
        i, f = mult; r = g.reaction(i)
        if i == IDX_R1:
            lr = r.rate.low_rate
            r.rate.low_rate = ct.Arrhenius(lr.pre_exponential_factor*f,
                                           lr.temperature_exponent, lr.activation_energy)
        else:
            r.rate = ct.Arrhenius(r.rate.pre_exponential_factor*f,
                                  r.rate.temperature_exponent, r.rate.activation_energy)
        g.modify_reaction(i, r)
    g.TPX = T_INITIAL, P_INITIAL, INITIAL_X
    rr = ct.IdealGasConstPressureReactor(g, energy='on')
    net = ct.ReactorNet([rr]); h2o = g.species_index('H2O')
    out = np.empty(N_STEPS)
    for k in range(N_STEPS):
        net.advance(T_SIM[k]); out[k] = rr.thermo.X[h2o]
    return out


# Compute sensitivities on both grids
print('\nComputing sensitivities (OH on fine grid, H2O on coarse grid) ...')
d = 0.01
S_k1_oh = (_oh_profile((IDX_R1, 1+d)) - _nom_oh) / _nom_oh / d
S_k2_oh = (_oh_profile((IDX_R2, 1+d)) - _nom_oh) / _nom_oh / d
S_oh = np.vstack([S_k1_oh, S_k2_oh]).T   # shape: (N_FINE, 2)

S_k1_h2o = (_h2o_profile((IDX_R1, 1+d)) - _nom_h2o) / _nom_h2o / d
S_k2_h2o = (_h2o_profile((IDX_R2, 1+d)) - _nom_h2o) / _nom_h2o / d
S_h2o = np.vstack([S_k1_h2o, S_k2_h2o]).T   # shape: (N_STEPS, 2)


def _select_independent(S_mat, t_grid, cand, n_targets, dt_min):
    """
    Single-species D-optimal selection: greedily maximize det(S^T S) using ONLY
    this species' own Gram matrix -- no cross-species competition at all. The
    exact same function is called once for OH and once for H2O below, so the
    methodology is fully symmetric/universal between species; only the window,
    candidate filter, and spacing floor passed in differ per species.
    """
    chosen = [cand[np.argmax(np.linalg.norm(S_mat[cand], axis=1))]]
    for _ in range(n_targets - 1):
        best_det, best_idx = -np.inf, None
        for c in cand:
            if c in chosen or any(abs(t_grid[c] - t_grid[j]) < dt_min for j in chosen):
                continue
            S_test = S_mat[chosen + [c]]
            v = np.linalg.det(S_test.T @ S_test)
            if v > best_det:
                best_det, best_idx = v, c
        if best_idx is not None:
            chosen.append(best_idx)
    return sorted(chosen)


def select_targets_joint(t_oh, S_oh_mat, t_h2o, S_h2o_mat, n_oh, n_h2o, oh_min_ppm=30,
                        t_min_oh=T_MIN_TARGET_OH, t_max_oh=T_MAX_TARGET_OH,
                        t_min_h2o=T_MIN_TARGET_H2O, t_max_h2o=T_MAX_TARGET_H2O,
                        dt_min_oh=5e-6, dt_min_h2o=5e-6):
    """
    INDEPENDENT per-species D-optimal selection (name kept as select_targets_joint
    for drop-in compatibility with the rest of this file -- it no longer competes
    OH and H2O against each other for a shared quota).

    Earlier "joint" versions stacked S_oh/S_h2o rows and let whichever species had
    the larger marginal det(S^T S) gain win each greedy step. OH's raw log-
    sensitivity is inherently ~10x larger than H2O's (a direct radical vs. a
    buffered ~1234 ppm product pool -- confirmed by direct measurement: median
    ||S_oh||=0.39 vs median ||S_h2o||=0.04 over the shared candidate window), so
    that competition let OH win almost every step by construction, starving H2O
    of its own best points (RAW: H2O never got past ~18 us, missing its plateau/
    yield anchor entirely). Rescaling each species by its own median sensitivity
    before the competition (tried as an intermediate step) fixed H2O's starvation
    but then let H2O's now-equal-footing candidates outcompete OH's for shared
    slots -- OH lost its rise-region point (1 us) to a second, largely redundant
    near-peak point (7.2 us, right next to the already-chosen 12.2 us pick),
    which measurably hurt OH's own k1/peak identifiability.

    Removing the cross-species competition entirely sidesteps both failure modes:
    each species independently gets its OWN best n-point D-optimal set, using the
    identical selection formula, with no scale asymmetry able to bias picks either
    direction. Verified directly: this recovers OH's full rise+peak+decay spread
    (matching the original un-starved RAW OH picks) while ALSO giving H2O a
    properly wide rise-to-plateau spread (including the yield anchor near the
    window boundary) that neither joint method achieved.
    """
    cand_oh_window = np.where((t_oh >= t_min_oh) & (t_oh <= t_max_oh))[0]
    cand_oh_measurable = np.where(_nom_oh[np.searchsorted(T_FINE, t_oh)] * 1e6 >= oh_min_ppm)[0]
    cand_oh  = np.intersect1d(cand_oh_window, cand_oh_measurable)
    cand_h2o = np.where((t_h2o >= t_min_h2o) & (t_h2o <= t_max_h2o))[0]

    chosen_oh  = _select_independent(S_oh_mat,  t_oh,  cand_oh,  n_oh,  dt_min_oh)
    chosen_h2o = _select_independent(S_h2o_mat, t_h2o, cand_h2o, n_h2o, dt_min_h2o)

    return chosen_oh, chosen_h2o


idx_oh, idx_h2o = select_targets_joint(T_FINE, S_oh, T_SIM, S_h2o, N_OH_TARGETS, N_H2O_TARGETS)
OH_TARGET_TIMES  = T_FINE[idx_oh]
H2O_TARGET_TIMES = T_SIM[idx_h2o]
assert len(OH_TARGET_TIMES) == N_OH_TARGETS and len(H2O_TARGET_TIMES) == N_H2O_TARGETS, (
    f'target selection returned {len(OH_TARGET_TIMES)} OH + {len(H2O_TARGET_TIMES)} H2O, '
    f'expected {N_OH_TARGETS}+{N_H2O_TARGETS} -- the net would train on garbage columns.')

OH_LABELS  = [f'OH D-opt @ {t*1e3:.4f} ms' for t in OH_TARGET_TIMES]
H2O_LABELS = [f'H2O D-opt @ {t*1e3:.4f} ms' for t in H2O_TARGET_TIMES]

print(f'\nOH target probes (window {T_MIN_TARGET_OH*1e6:.1f} µs .. {T_MAX_TARGET_OH*1e3:.2f} ms, independent D-optimal):')
oh_nom_vals = [np.interp(t, T_FINE, _nom_oh) * 1e6 for t in OH_TARGET_TIMES]
for j, (lbl, val, t) in enumerate(zip(OH_LABELS, oh_nom_vals, OH_TARGET_TIMES)):
    print(f'  {j+1}. {lbl:<30}  OH ~ {val:.2f} ppm')

print(f'\nH2O target probes (window {T_MIN_TARGET_H2O*1e6:.1f} µs .. {T_MAX_TARGET_H2O*1e3:.2f} ms, independent D-optimal):')
h2o_nom_vals = [np.interp(t, T_SIM, _nom_h2o) * 1e6 for t in H2O_TARGET_TIMES]
for j, (lbl, val, t) in enumerate(zip(H2O_LABELS, h2o_nom_vals, H2O_TARGET_TIMES)):
    print(f'  {j+1}. {lbl:<30}  H2O ~ {val:.2f} ppm')

# ── Advance order: merge OH and H2O target times chronologically ──────────────

_all_times  = np.concatenate([OH_TARGET_TIMES, H2O_TARGET_TIMES])
_species_id = np.array([0] * N_OH_TARGETS + [1] * N_H2O_TARGETS)  # 0=OH, 1=H2O
_sort_idx   = np.argsort(_all_times)
_advance_times  = _all_times[_sort_idx]      # chronological advance times
_advance_species = _species_id[_sort_idx]    # which species to record at each step

# Output position within the per-species array for each advance step
_oh_counter  = 0
_h2o_counter = 0
_advance_out_idx = np.empty(len(_sort_idx), dtype=int)
for k, sp in enumerate(_advance_species):
    if sp == 0:
        _advance_out_idx[k] = _oh_counter;  _oh_counter  += 1
    else:
        _advance_out_idx[k] = _h2o_counter; _h2o_counter += 1


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


# ── Cantera simulation — joint OH + H2O ───────────────────────────────────────

def run_single(x_vec):
    try:
        gas = ct.Solution(YAML_FILE)
        # R1 (k1) — falloff low-rate
        new_A_R1  = NOMINAL_A_R1 * np.exp(x_vec[0] * LN_F)
        new_Ea_R1 = (NOMINAL_EA_R1_cal + x_vec[1] * SIGMA_E) * 4184.0
        rxn22 = gas.reaction(IDX_R1)
        rxn22.rate.low_rate = ct.Arrhenius(new_A_R1, NOMINAL_B_R1, new_Ea_R1)
        gas.modify_reaction(IDX_R1, rxn22)
        # R2 (k2) — simple Arrhenius
        new_A_R2  = NOMINAL_A_R2 * np.exp(x_vec[2] * LN_F)
        new_Ea_R2 = (NOMINAL_EA_R2_cal + x_vec[3] * SIGMA_E) * 4184.0
        rxn26 = gas.reaction(IDX_R2)
        rxn26.rate = ct.Arrhenius(new_A_R2, NOMINAL_B_R2, new_Ea_R2)
        gas.modify_reaction(IDX_R2, rxn26)

        gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
        reactor  = ct.IdealGasConstPressureReactor(gas, energy='on')
        net      = ct.ReactorNet([reactor])
        oh_idx   = gas.species_index('OH')
        h2o_idx  = gas.species_index('H2O')

        y_oh  = np.full(N_OH_TARGETS,  np.nan)   # nan, not np.empty: unfilled
        y_h2o = np.full(N_H2O_TARGETS, np.nan)   # slots must fail loudly below

        # Single chronological pass through all target times
        for k in range(len(_advance_times)):
            net.advance(_advance_times[k])
            out_pos = _advance_out_idx[k]
            if _advance_species[k] == 0:
                y_oh[out_pos]  = reactor.thermo.X[oh_idx]
            else:
                y_h2o[out_pos] = reactor.thermo.X[h2o_idx]

        y_out = np.concatenate([y_oh, y_h2o])
        if not np.isfinite(y_out).all():
            return False, None    # a target slot was never filled
        return True, y_out
    except Exception:
        return False, None


# ── Checkpoint / run simulations ──────────────────────────────────────────────

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
        torch.save({'last_index': i, 'y_list': y_list, 'x_list': x_list,
                    'l_list': l_list, 'n_failed': n_failed,
                    'oh_target_times':  OH_TARGET_TIMES,
                    'h2o_target_times': H2O_TARGET_TIMES,
                    'oh_labels': OH_LABELS,
                    'h2o_labels': H2O_LABELS}, CHECKPOINT_PATH)

torch.save({'last_index': TOTAL_SAMPLES - 1, 'y_list': y_list, 'x_list': x_list,
            'l_list': l_list, 'n_failed': n_failed,
            'oh_target_times':  OH_TARGET_TIMES,
            'h2o_target_times': H2O_TARGET_TIMES,
            'oh_labels': OH_LABELS,
            'h2o_labels': H2O_LABELS}, CHECKPOINT_PATH)

X_raw = np.asarray(x_list); Y_raw = np.asarray(y_list); L_raw = np.asarray(l_list)
print(f'\nShapes: X={X_raw.shape}  Y={Y_raw.shape}  (failed={n_failed})')
print(f'  OH outputs:  columns 0..{N_OH_TARGETS-1}')
print(f'  H2O outputs: columns {N_OH_TARGETS}..{N_TARGETS-1}')


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


# ── Architecture — joint 4 in -> 6 out ─────────────────────────────────────────

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
        # Compute loss separately for OH and H2O
        loss_oh  = torch.mean(rel_err[:, :N_OH_TARGETS])
        loss_h2o = torch.mean(rel_err[:, N_OH_TARGETS:])
        return WEIGHT_OH * loss_oh + WEIGHT_H2O * loss_h2o

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

print(f'\nTraining (1398 K joint OH+H2O, 3-pt each, OH: {T_MIN_TARGET_OH*1e6:.0f}us-{T_MAX_TARGET_OH*1e3:.3f}ms / '
      f'H2O: {T_MIN_TARGET_H2O*1e6:.0f}us-{T_MAX_TARGET_H2O*1e3:.3f}ms [95%-rise rule]) ...')
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
print('  1398 K Joint (OH + H2O) — 3-pt each (constrained)  |  Zhang Table 1')
print(f'{"="*72}')

print('\nOH accuracy:')
all_pass_oh = True
for k, sigma in enumerate(SIGMA_LIST, 1):
    sel    = (L_test == sigma)
    r      = rel_err[sel, :N_OH_TARGETS]
    e_mean = r.mean(); e_95 = np.percentile(r, 95)
    req    = SIGMA_REQS[sigma]
    pm, p9 = e_mean <= req[0], e_95 <= req[1]
    ok     = 'PASS' if (pm and p9) else 'FAIL'
    all_pass_oh = all_pass_oh and pm and p9
    print(f'  Set {k}  sigma={sigma:<4} N={sel.sum():<6} '
          f'mean={e_mean*100:6.2f}% (<= {req[0]*100:.0f}%)  '
          f'p95={e_95*100:6.2f}% (<= {req[1]*100:.0f}%)  {ok}')

print('\nH2O accuracy:')
all_pass_h2o = True
for k, sigma in enumerate(SIGMA_LIST, 1):
    sel    = (L_test == sigma)
    r      = rel_err[sel, N_OH_TARGETS:]
    e_mean = r.mean(); e_95 = np.percentile(r, 95)
    req    = SIGMA_REQS[sigma]
    pm, p9 = e_mean <= req[0], e_95 <= req[1]
    ok     = 'PASS' if (pm and p9) else 'FAIL'
    all_pass_h2o = all_pass_h2o and pm and p9
    print(f'  Set {k}  sigma={sigma:<4} N={sel.sum():<6} '
          f'mean={e_mean*100:6.2f}% (<= {req[0]*100:.0f}%)  '
          f'p95={e_95*100:6.2f}% (<= {req[1]*100:.0f}%)  {ok}')

all_pass = all_pass_oh and all_pass_h2o
print(f'{"="*72}')
print('MEETS' if all_pass else 'does NOT meet', 'all Zhang Table 1 requirements.')


# ── Save ──────────────────────────────────────────────────────────────────────

torch.save({
    'model_state':      model.state_dict(),
    'test_indices':     list(test_ds.indices),
    'oh_target_times':  OH_TARGET_TIMES,
    'h2o_target_times': H2O_TARGET_TIMES,
    'oh_labels':        OH_LABELS,
    'h2o_labels':       H2O_LABELS,
    'n_oh_targets':     N_OH_TARGETS,
    'n_h2o_targets':    N_H2O_TARGETS,
    'hidden_dim':       HIDDEN_DIM,
    'ln_f':             LN_F,
    'sigma_e':          SIGMA_E,
    'train_losses':     train_losses,
    'val_losses':       val_losses,
    'val_epochs':       val_epochs,
    'best_val':         best_val,
}, RESULT_PATH)
print(f'\nSaved: {RESULT_PATH}')
