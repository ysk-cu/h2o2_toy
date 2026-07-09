#!/usr/bin/env python
# coding: utf-8
#
# nn_toy_1192k_h2o_infer.py  —  User Approach 2
# Train H2O surrogate (1192 K, 1 ms), then recover rate parameters via
# optimization and evaluate resulting OH prediction accuracy.
#
# Condition : 2216 ppm H2O2 / 1364 ppm H2O / 682 ppm O2 / Ar
#             T = 1192 K, P = 1.95 atm
#
# Workflow:
#   1. Build NN surrogate: (lnA_R1, Ea_R1, lnA_R2, Ea_R2) → H2O(t) at 5 times
#   2. Evaluate H2O surrogate accuracy (Zhang Table 1)
#   3. For test samples: optimize x to fit observed H2O → run Cantera for OH
#   4. Report OH prediction accuracy from the recovered parameters
#
# H2O is monotonically increasing (no sharp peak timing problem) → easy to surrogate.
# Inputs:  x[0]=lnA_R1  x[1]=Ea_R1  x[2]=lnA_R2  x[3]=Ea_R2

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

# ── Mechanism & nominal rate constants ─────────────────────────────────────────

YAML_FILE = 'chem_cti_toy_model_og.yaml'
mol_units = ct.UnitSystem({
    "length": "cm", "mass": "g", "time": "s",
    "quantity": "mol", "pressure": "dyn / cm^2", "energy": "erg",
    "temperature": "K", "current": "A", "activation-energy": "cal / mol"})

IDX_R1 = 21
IDX_R2 = 25

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

print(f'R1: A={NOMINAL_A_R1:.3e}  Ea={NOMINAL_EA_R1_cal:.0f} cal/mol')
print(f'R2: A={NOMINAL_A_R2:.3e}  Ea={NOMINAL_EA_R2_cal:.0f} cal/mol')

LN_F        = np.log(10)
SIGMA_E     = 5000.0
PARAM_NAMES = ['lnA_R1', 'Ea_R1', 'lnA_R2', 'Ea_R2']
INPUT_DIM   = 4

TOTAL_SAMPLES = 40000
SIGMA_LIST    = (0.1, 0.3, 0.5)
RATIO_LIST    = (1/6, 1/6, 2/3)

# ── Shock tube condition (1192 K) ──────────────────────────────────────────────

T_INITIAL = 1192
P_INITIAL = 1.95 * ct.one_atm
INITIAL_X = {
    'H2O2': 2216e-6,
    'H2O':  1364e-6,
    'O2':   682e-6,
    'AR':   1.0 - (2216 + 1364 + 682) * 1e-6,
}

# Fixed-step loop for H2O profile: 1 µs × 1000 steps = 1 ms total
DT_SIM     = 1e-6
N_STEPS    = 1000
T_SIM      = np.linspace(DT_SIM, DT_SIM * N_STEPS, N_STEPS)

# ── NN / training hyper-parameters ────────────────────────────────────────────

HIDDEN_DIM     = 16
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

CHECKPOINT_PATH = 'ckpt_1192k_h2o_train_S.pt'
RESULT_PATH     = 'result_1192k_h2o_train_S.pt'
SIGMA_REQS      = {0.1: (0.01, 0.02), 0.3: (0.02, 0.05), 0.5: (0.03, 0.10)}
LOG_EPS         = 1e-12

N_TARGETS = 3  # rise, plateau points chosen by sensitivity analysis


# ── Auto-select H2O target times from nominal profile ─────────────────────────

def run_nominal_h2o():
    local_gas = ct.Solution(YAML_FILE)
    local_gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
    reactor = ct.IdealGasConstPressureReactor(local_gas, energy='on')
    net     = ct.ReactorNet([reactor])
    h2o_idx = local_gas.species_index('H2O')
    profile = np.empty(N_STEPS)
    for k in range(N_STEPS):
        net.advance(T_SIM[k])
        profile[k] = reactor.thermo.X[h2o_idx]
    return profile


print('\nRunning nominal H2O profile (1 ms) ...')
_nom_h2o = run_nominal_h2o()
_h2o_0   = float(_nom_h2o[0])
_h2o_inf = float(_nom_h2o[-200:].mean())
print(f'  H2O initial = {_h2o_0*1e6:.1f} ppm   H2O plateau = {_h2o_inf*1e6:.1f} ppm')


# ── H2O sensitivity-based target selection (pure-k1 probe) ────────────────────
# H2O sees essentially only k1 (S_k2 ~ 0), so the OH det(S^T W S) objective
# degenerates. We place a few points across the high-S_k1 rise with a minimum
# spacing, and keep one plateau point as a stoichiometry/yield consistency anchor.

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


# ── OLD: k1-only target selection (commented out) ──────────────────────────
# def select_h2o_targets_k1_only(t_grid, H2O_nom, S_k1, n_rise, min_dt=0.03e-3, yield_anchor=True):
#     """
#     n_rise       : number of k1-informative points placed on the rise
#     min_dt       : minimum spacing (s) so points don't collapse onto the S_k1 peak
#     yield_anchor : if True, add one point near the plateau (~95% of rise) as a
#                    stoichiometry check (low k1 info, but harmless and useful)
#     Returns sorted indices into t_grid.
#     """
#     chosen = []
#     for c in np.argsort(-np.abs(S_k1)):                     # most k1-informative first
#         if all(abs(t_grid[c] - t_grid[j]) >= min_dt for j in chosen):
#             chosen.append(int(c))
#         if len(chosen) == n_rise:
#             break
#     if yield_anchor:
#         frac = (H2O_nom - H2O_nom[0]) / (H2O_nom[-1] - H2O_nom[0])
#         anchor = int(np.argmin(np.abs(frac - 0.95)))
#         if anchor not in chosen:
#             chosen.append(anchor)
#     return sorted(chosen)

# ── NEW: k1+k2 combined target selection (for joint compatibility) ──────────
def select_h2o_targets(t_grid, H2O_nom, S_k1, S_k2, n_targets, min_dt=0.03e-3):
    """
    NEW: Uses both k1 and k2 sensitivities for target selection.
    This makes H2O targets compatible with joint OH+H2O inference where k1/k2 are coupled.

    Args:
        t_grid     : time grid
        H2O_nom    : nominal H2O profile
        S_k1       : d(log H2O)/d(lnA_R1)
        S_k2       : d(log H2O)/d(lnA_R2)
        n_targets  : total number of targets to select
        min_dt     : minimum spacing between targets

    Returns sorted indices into t_grid.
    """
    S = np.vstack([S_k1, S_k2]).T  # shape: (N, 2) — combined sensitivity matrix

    # Candidates: use H2O threshold like before
    cand = np.arange(len(t_grid))

    # Seed: strongest combined sensitivity
    norm_S = np.linalg.norm(S[cand], axis=1)
    chosen = [cand[np.argmax(norm_S)]]

    # Greedily add targets maximizing det(S^T S) with spacing constraint
    for _ in range(n_targets - 1):
        best_det = -np.inf
        best_idx = None

        for c in cand:
            if c not in chosen and all(abs(t_grid[c] - t_grid[j]) >= min_dt for j in chosen):
                S_test = S[chosen + [c]]
                if S_test.shape[0] >= 2:
                    det_val = np.linalg.det(S_test.T @ S_test)
                    if det_val > best_det:
                        best_det = det_val
                        best_idx = c

        if best_idx is not None:
            chosen.append(best_idx)

    return sorted(chosen)
# 5 target times at 20 / 40 / 60 / 80 / 95% of the H2O rise
# _fracs   = np.array([0.20, 0.40, 0.60, 0.80, 0.95])
# 3 target times at 20 / 60 / 95% of the H2O rise
# _fracs   = np.array([0.20, 0.60, 0.95])
# _targets = _h2o_0 + _fracs * (_h2o_inf - _h2o_0)
# TARGET_TIMES = np.array([
#     T_SIM[np.argmax(_nom_h2o >= xt)] if np.any(_nom_h2o >= xt)
#     else T_SIM[-1]
#     for xt in _targets
# ])
# N_TARGETS = len(TARGET_TIMES)

# print('\nH2O target times:')
# for j, (f, t) in enumerate(zip(_fracs, TARGET_TIMES)):
#     h2o_ppm = np.interp(t, T_SIM, _nom_h2o) * 1e6
#     print(f'  {j+1}. {int(f*100):2d}% rise  t = {t*1e3:.3f} ms   H2O ≈ {h2o_ppm:.1f} ppm')

_h2o_nom = _h2o_profile()
d = 0.01
S_k1 = (_h2o_profile((IDX_R1, 1 + d)) - _h2o_nom) / _h2o_nom / d
S_k2 = (_h2o_profile((IDX_R2, 1 + d)) - _h2o_nom) / _h2o_nom / d
# NEW: Include S_k2 even though H2O is mostly k1-sensitive, for joint compatibility

# ── OLD: k1-only selection (commented) ──────────────────────────────────────
# idx = select_h2o_targets_k1_only(T_SIM, _h2o_nom, S_k1,
#                          n_rise=N_TARGETS - 1,   # leave one slot for the anchor
#                          min_dt=0.03e-3, yield_anchor=True)

# ── NEW: k1+k2 joint selection ─────────────────────────────────────────────
idx = select_h2o_targets(T_SIM, _h2o_nom, S_k1, S_k2,
                         n_targets=N_TARGETS, min_dt=0.03e-3)
TARGET_TIMES = T_SIM[idx]

print('\nSelected H2O target times (k1+k2 combined sensitivity):')
for j in idx:
    print(f"  t={T_SIM[j]*1e3:6.3f} ms   H2O={_h2o_nom[j]*1e6:5.0f} ppm   S_k1={S_k1[j]:+.3f}  S_k2={S_k2[j]:+.3f}")

# ── Sobol sampling ─────────────────────────────────────────────────────────────

def multiscale_sobol(n_total, sigmas, ratios, d=INPUT_DIM, seed=SEED):
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    base    = sampler.random(n_total)
    chunks, labels = [], []
    idx = 0
    for i, (s, r) in enumerate(zip(sigmas, ratios)):
        count  = n_total - idx if i == len(sigmas) - 1 else int(round(n_total * r))
        u      = base[idx: idx + count]
        p_lo   = norm.cdf(-1.0, loc=0, scale=s)
        p_hi   = norm.cdf( 1.0, loc=0, scale=s)
        chunk  = norm.ppf(p_lo + u * (p_hi - p_lo), loc=0, scale=s)
        chunks.append(chunk)
        labels.append(np.full(count, s))
        idx += count
    return np.vstack(chunks), np.concatenate(labels)

X_samples, L_samples = multiscale_sobol(TOTAL_SAMPLES, SIGMA_LIST, RATIO_LIST)


# ── Cantera simulation — H2O at target times ───────────────────────────────────

def _perturb_gas(x_vec):
    local_gas = ct.Solution(YAML_FILE)
    new_A_R1  = NOMINAL_A_R1 * np.exp(x_vec[0] * LN_F)
    new_Ea_R1 = (NOMINAL_EA_R1_cal + x_vec[1] * SIGMA_E) * 4184.0
    rxn22 = local_gas.reaction(IDX_R1)
    rxn22.rate.low_rate = ct.Arrhenius(new_A_R1, NOMINAL_B_R1, new_Ea_R1)
    local_gas.modify_reaction(IDX_R1, rxn22)
    new_A_R2  = NOMINAL_A_R2 * np.exp(x_vec[2] * LN_F)
    new_Ea_R2 = (NOMINAL_EA_R2_cal + x_vec[3] * SIGMA_E) * 4184.0
    rxn26 = local_gas.reaction(IDX_R2)
    rxn26.rate = ct.Arrhenius(new_A_R2, NOMINAL_B_R2, new_Ea_R2)
    local_gas.modify_reaction(IDX_R2, rxn26)
    return local_gas


def run_single(x_vec):
    """Run fixed-step H2O simulation, interpolate to TARGET_TIMES."""
    try:
        local_gas = _perturb_gas(x_vec)
        local_gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
        reactor = ct.IdealGasConstPressureReactor(local_gas, energy='on')
        net     = ct.ReactorNet([reactor])
        h2o_idx = local_gas.species_index('H2O')
        profile = np.empty(N_STEPS)
        for k in range(N_STEPS):
            net.advance(T_SIM[k])
            profile[k] = reactor.thermo.X[h2o_idx]
        return True, np.interp(TARGET_TIMES, T_SIM, profile)
    except Exception:
        return False, None



# ── Checkpoint / run simulations ───────────────────────────────────────────────

if os.path.exists(CHECKPOINT_PATH):
    ckpt     = torch.load(CHECKPOINT_PATH, weights_only=False)
    raw_y    = ckpt['y_list']
    raw_x    = ckpt['x_list']
    raw_l    = ckpt.get('l_list', [None] * len(raw_y))
    n_failed = ckpt.get('n_failed', 0)
    valid    = [i for i, y in enumerate(raw_y) if np.asarray(y).shape == (N_TARGETS,)]
    y_list   = [raw_y[i] for i in valid]
    x_list   = [raw_x[i] for i in valid]
    l_list   = [raw_l[i] for i in valid]
    dropped  = len(raw_y) - len(valid)
    start_idx = ckpt['last_index'] + 1
    print(f'\nResuming from idx {start_idx}/{TOTAL_SAMPLES}: '
          f'{len(y_list)} valid, {dropped} dropped, {n_failed} failed')
else:
    y_list, x_list, l_list, n_failed, start_idx = [], [], [], 0, 0
    print(f'\nStarting fresh: {TOTAL_SAMPLES} simulations')

t0 = time.time()
for i in range(start_idx, TOTAL_SAMPLES):
    ok, y_tgt = run_single(X_samples[i])
    if ok:
        y_list.append(y_tgt)
        x_list.append(X_samples[i])
        l_list.append(L_samples[i])
    else:
        n_failed += 1
    if (i + 1) % 2000 == 0:
        rate = (i + 1 - start_idx) / max(time.time() - t0, 1e-9)
        print(f'  {i+1}/{TOTAL_SAMPLES}  {rate:.1f} sim/s  failed={n_failed}')
        torch.save({'last_index': i, 'y_list': y_list, 'x_list': x_list,
                    'l_list': l_list, 'n_failed': n_failed,
                    'target_times': TARGET_TIMES}, CHECKPOINT_PATH)

torch.save({'last_index': TOTAL_SAMPLES - 1, 'y_list': y_list, 'x_list': x_list,
            'l_list': l_list, 'n_failed': n_failed,
            'target_times': TARGET_TIMES}, CHECKPOINT_PATH)

sim_elapsed = time.time() - t0
print(f'\nSample generation: {sim_elapsed:.1f} s  ({sim_elapsed/60:.2f} min)  '
      f'{(TOTAL_SAMPLES - start_idx) / max(sim_elapsed, 1e-9):.1f} sim/s  '
      f'failed={n_failed}')

X_raw = np.asarray(x_list)
Y_raw = np.asarray(y_list)
L_raw = np.asarray(l_list)
print(f'\nShapes: X={X_raw.shape}  Y={Y_raw.shape}  (failed={n_failed})')


# ── Log transform & split ──────────────────────────────────────────────────────

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


# ── Architecture ───────────────────────────────────────────────────────────────

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
class RelativeErrorLoss(nn.Module):
    def forward(self, pred, target):
        pred_raw = torch.exp(pred)
        target_raw = torch.exp(target)
        return torch.mean(torch.abs(pred_raw - target_raw) / (torch.abs(target_raw) + 1e-30))

criterion = RelativeErrorLoss()

train_losses, val_losses, val_epochs = [], [], []
best_val   = float('inf')
best_state = copy.deepcopy(model.state_dict())


def _pass(loader, train_mode):
    total = 0.0
    for xb, yb, _ in loader:
        if train_mode:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = criterion(model(xb), yb)
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


print('\nTraining (1192K H2O surrogate, 1 ms, ReLU, relative error loss) ...')
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    tl = _pass(train_loader, train_mode=True)
    train_losses.append(tl)

    if (epoch + 1) % val_check == 0 or epoch == 0:
        model.eval()
        vl = _pass(val_loader, train_mode=False)
        val_losses.append(vl)
        val_epochs.append(epoch + 1)
        scheduler.step(vl)

        gap    = vl / max(tl, 1e-12)
        cur_wd = optimizer.param_groups[0]['weight_decay']
        new_wd = (min(cur_wd * 2.0, wd_max) if gap > wd_gap_high
                  else max(cur_wd * 0.5, wd_min) if gap < wd_gap_low
                  else cur_wd)
        for g in optimizer.param_groups:
            g['weight_decay'] = new_wd

        if vl < best_val:
            best_val   = vl
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 500 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'  ep {epoch+1:5d} | train {tl:.4e} | val {vl:.4e} | '
                  f'gap {gap:.3f} | wd {new_wd:.1e} | lr {lr_now:.1e}')

model.load_state_dict(best_state)
print(f'Done in {(time.time()-t0)/60:.1f} min.  Best val: {best_val:.4e}')


# ── H2O surrogate accuracy (Zhang Table 1) ────────────────────────────────────

model.eval()
X_te   = torch.tensor(X_raw[test_ds.indices], dtype=torch.float32)
Y_true = Y_raw[test_ds.indices]
L_test = L_raw[test_ds.indices]

with torch.no_grad():
    Y_pred = np.exp(model(X_te).numpy())
rel_err_h2o = np.abs(Y_pred - Y_true) / (np.abs(Y_true) + 1e-30)

print(f'\n{"="*70}')
print('  1192K H2O surrogate  |  Zhang Table 1')
print(f'{"="*70}')
all_pass = True
for k, sigma in enumerate(SIGMA_LIST, 1):
    sel    = (L_test == sigma)
    r      = rel_err_h2o[sel]
    e_mean = r.mean()
    e_95   = np.percentile(r, 95)
    req    = SIGMA_REQS[sigma]
    pm, p9 = e_mean <= req[0], e_95 <= req[1]
    ok     = 'PASS' if (pm and p9) else 'FAIL'
    all_pass = all_pass and pm and p9
    print(f'  Set {k}  σ={sigma:<4}  N={sel.sum():<5}  '
          f'mean={e_mean*100:6.2f}%  (≤{req[0]*100:.0f}%)   '
          f'p95={e_95*100:6.2f}%  (≤{req[1]*100:.0f}%)   {ok}')
print(f'{"="*70}')
print('MEETS' if all_pass else 'does NOT meet', 'all Zhang Table 1 requirements.')

print('\nPer-H2O-target mean relative error (%):')
_h2o_rise = _h2o_nom[-1] - _h2o_nom[0]
for j, t in enumerate(TARGET_TIMES):
    frac = (np.interp(t, T_SIM, _h2o_nom) - _h2o_nom[0]) / _h2o_rise
    print(f'  {j+1}. {frac*100:5.1f}% rise  t={t*1e3:.3f} ms   '
          f'err={rel_err_h2o[:, j].mean()*100:.2f}%')


# ── Save ───────────────────────────────────────────────────────────────────────

torch.save({
    'model_state':  model.state_dict(),
    'test_indices': list(test_ds.indices),
    'target_times': TARGET_TIMES,
    'hidden_dim':   HIDDEN_DIM,
    'n_targets':    N_TARGETS,
    'ln_f':         LN_F,
    'sigma_e':      SIGMA_E,
    'train_losses': train_losses,
    'val_losses':   val_losses,
    'val_epochs':   val_epochs,
    'best_val':     best_val,
}, RESULT_PATH)
print(f'\nSaved: {RESULT_PATH}')
