#!/usr/bin/env python
# coding: utf-8
#
# Discrete-target NN surrogate — ReLU, R22 only (2 params)
# Identical to h2o2_NN_toy_R22_discrete_targets_trainv2.py except:
#   * activation: ReLU instead of GELU
#   * saves to relu_r22_discrete_targets.pt
#   * reuses simulation_checkpoint_r22_discrete.pt (no new Cantera runs needed)

import os, time, copy
import cantera as ct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.stats import qmc, norm
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
torch.set_num_threads(16)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────

YAML_FILE = 'chem_cti_toy_model_og.yaml'
mol_units = ct.UnitSystem({
    "length": "cm", "mass": "g", "time": "s",
    "quantity": "mol", "pressure": "dyn / cm^2", "energy": "erg",
    "temperature": "K", "current": "A", "activation-energy": "cal / mol"})

IDX_R22 = 21
_gas_nom = ct.Solution(YAML_FILE)
NOMINAL_A_R22      = _gas_nom.reaction(IDX_R22).rate.low_rate.pre_exponential_factor
NOMINAL_B_R22      = _gas_nom.reaction(IDX_R22).rate.low_rate.temperature_exponent
NOMINAL_EA_R22_si  = _gas_nom.reaction(IDX_R22).rate.low_rate.activation_energy
NOMINAL_EA_R22_cal = mol_units.convert_activation_energy_to(
    f"{NOMINAL_EA_R22_si} J/kmol", "cal / mol")
del _gas_nom

LN_F        = 10
SIGMA_E     = 2000.0      # cal/mol half-width for Ea
PARAM_NAMES = ['lnA_R22', 'Ea_R22']
INPUT_DIM   = 2

TOTAL_SAMPLES = 40000
SIGMA_LIST    = (0.1, 0.3, 0.5)
RATIO_LIST    = (1/6, 1/6, 2/3)

T_INITIAL  = 1057
P_INITIAL  = 1.83 * ct.one_atm
INITIAL_X  = {'H2O2': 860e-6, 'H2O': 663e-6, 'O2': 332e-6,
               'AR':   1.0 - (860+663+332)*1e-6}
DT_MAX     = 1e-6
TIME_STEPS = 6000
T_SIM      = np.linspace(DT_MAX, DT_MAX * TIME_STEPS, TIME_STEPS)

HIDDEN_DIM  = 16
lr_init     = 0.03
TRAIN_FRAC  = 0.80
VAL_FRAC    = 0.10
EPOCHS      = 5000
BATCH_SIZE  = 1024
val_check   = 10
lr_check    = 200
LR_FACTOR   = 0.5
LR_MIN      = 1e-6
wd_init     = 1e-6
wd_min, wd_max = 1e-8, 1e-4
wd_gap_high    = 1.10
wd_gap_low     = 1.02

CHECKPOINT_PATH = 'simulation_checkpoint_r22_discrete.pt'   # reuse GELU sims
RESULT_PATH     = 'relu_r22_discrete_targets.pt'
SIGMA_REQS      = {0.1: (0.01, 0.02), 0.3: (0.02, 0.05), 0.5: (0.03, 0.10)}
LOG_EPS         = 1e-10


# ── Auto-select TARGET_TIMES from nominal profile ─────────────────────────────

def run_full_profile(x_vec):
    local_gas = ct.Solution(YAML_FILE)
    new_A     = NOMINAL_A_R22 * np.exp(x_vec[0] * LN_F)
    new_Ea    = (NOMINAL_EA_R22_cal + x_vec[1] * SIGMA_E) * 4184.0
    rxn22 = local_gas.reaction(IDX_R22)
    rxn22.rate.low_rate = ct.Arrhenius(new_A, NOMINAL_B_R22, new_Ea)
    local_gas.modify_reaction(IDX_R22, rxn22)
    local_gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
    reactor = ct.IdealGasReactor(local_gas, energy='on')
    net     = ct.ReactorNet([reactor])
    h2o_idx = local_gas.species_index('H2O')
    profile = np.empty(TIME_STEPS)
    for step in range(TIME_STEPS):
        net.advance(net.time + DT_MAX)
        profile[step] = reactor.thermo.X[h2o_idx]
    return profile

_nom   = run_full_profile(np.zeros(INPUT_DIM))
_X0    = _nom[0]
_Xinf  = _nom[-200:].mean()
_Xtgt  = _X0 + np.array([0.25, 0.50, 0.75, 0.90]) * (_Xinf - _X0)
TARGET_TIMES = np.array([T_SIM[np.argmax(_nom >= xt)] for xt in _Xtgt])
TARGET_TIMES = np.append(TARGET_TIMES, 5.0e-3)
N_TARGETS    = len(TARGET_TIMES)
print(f'Auto-selected TARGET_TIMES (s): {TARGET_TIMES}')


# ── Sobol sampling ────────────────────────────────────────────────────────────

def multiscale_sobol(n_total, sigmas, ratios, d=INPUT_DIM, seed=SEED):
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    base    = sampler.random(n_total)
    chunks, labels = [], []
    idx = 0
    for i, (s, r) in enumerate(zip(sigmas, ratios)):
        count = n_total - idx if i == len(sigmas) - 1 else int(round(n_total * r))
        u = base[idx: idx + count]
        p_low  = norm.cdf(-1.0, loc=0, scale=s)
        p_high = norm.cdf( 1.0, loc=0, scale=s)
        chunk  = norm.ppf(p_low + u * (p_high - p_low), loc=0, scale=s)
        chunks.append(chunk)
        labels.append(np.full(count, s))
        idx += count
    return np.vstack(chunks), np.concatenate(labels)

X_samples, L_samples = multiscale_sobol(TOTAL_SAMPLES, SIGMA_LIST, RATIO_LIST)


# ── Cantera simulation ────────────────────────────────────────────────────────

def run_single_simulation(x_vec):
    try:
        local_gas = ct.Solution(YAML_FILE)
        new_A     = NOMINAL_A_R22 * np.exp(x_vec[0] * LN_F)
        new_Ea    = (NOMINAL_EA_R22_cal + x_vec[1] * SIGMA_E) * 4184.0
        rxn22 = local_gas.reaction(IDX_R22)
        rxn22.rate.low_rate = ct.Arrhenius(new_A, NOMINAL_B_R22, new_Ea)
        local_gas.modify_reaction(IDX_R22, rxn22)
        local_gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
        reactor = ct.IdealGasReactor(local_gas, energy='on')
        net     = ct.ReactorNet([reactor])
        h2o_idx = local_gas.species_index('H2O')
        profile = np.empty(TIME_STEPS)
        for step in range(TIME_STEPS):
            net.advance(net.time + DT_MAX)
            profile[step] = reactor.thermo.X[h2o_idx]
        return True, np.interp(TARGET_TIMES, T_SIM, profile)
    except Exception:
        return False, None


if os.path.exists(CHECKPOINT_PATH):
    ckpt     = torch.load(CHECKPOINT_PATH, weights_only=False)
    raw_y    = ckpt['y_list']
    raw_x    = ckpt['x_list']
    raw_l    = ckpt.get('l_list', [None] * len(raw_y))
    n_failed = ckpt.get('n_failed', 0)
    # Keep only entries whose length matches the current N_TARGETS.
    # Mixed-length entries appear when a checkpoint was written by a previous
    # run that used a different TARGET_TIMES selection.
    valid  = [i for i, y in enumerate(raw_y)
              if np.asarray(y).shape == (N_TARGETS,)]
    y_list = [raw_y[i] for i in valid]
    x_list = [raw_x[i] for i in valid]
    l_list = [raw_l[i] for i in valid]
    dropped   = len(raw_y) - len(valid)
    start_idx = ckpt['last_index'] + 1
    print(f'Resuming from idx {start_idx}/{TOTAL_SAMPLES}: '
          f'{len(y_list)} valid entries kept, {dropped} dropped (shape mismatch), '
          f'{n_failed} failed')
else:
    y_list, x_list, l_list, n_failed, start_idx = [], [], [], 0, 0
    print(f'Starting fresh: {TOTAL_SAMPLES} simulations')

t0 = time.time()
for i in range(start_idx, TOTAL_SAMPLES):
    success, y_targets = run_single_simulation(X_samples[i])
    if success:
        y_list.append(y_targets); x_list.append(X_samples[i]); l_list.append(L_samples[i])
    else:
        n_failed += 1
    if (i + 1) % 2000 == 0:
        torch.save({'last_index': i, 'y_list': y_list, 'x_list': x_list,
                    'l_list': l_list, 'n_failed': n_failed,
                    'target_times': TARGET_TIMES}, CHECKPOINT_PATH)

torch.save({'last_index': TOTAL_SAMPLES - 1, 'y_list': y_list, 'x_list': x_list,
            'l_list': l_list, 'n_failed': n_failed,
            'target_times': TARGET_TIMES}, CHECKPOINT_PATH)

X_raw = np.asarray(x_list)
Y_raw = np.asarray(y_list)
L_raw = np.asarray(l_list)
print(f'Shapes: X={X_raw.shape}, Y={Y_raw.shape}  (failed: {n_failed})')


# ── Log transform ─────────────────────────────────────────────────────────────

Y_log = np.log(Y_raw + LOG_EPS)

X_t = torch.tensor(X_raw, dtype=torch.float32)
Y_t = torch.tensor(Y_log, dtype=torch.float32)
L_t = torch.tensor(L_raw, dtype=torch.float32)


# ── Train / val / test split ──────────────────────────────────────────────────

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
print(f'Split: train={n_train}, val={n_val}, test={n_test}')


# ── Architecture — ReLU activation ───────────────────────────────────────────

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


# ── Training ──────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
model     = SurrogateNN()
optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=wd_init)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR,
    patience=lr_check // val_check, min_lr=LR_MIN)
criterion = nn.MSELoss()

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

print('\nTraining: discrete-target ReLU surrogate (R22 only)')
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
print(f'Done in {(time.time()-t0)/60:.1f} min. Best val: {best_val:.4e}')


# ── Zhang Table 1 evaluation ──────────────────────────────────────────────────

model.eval()
X_te   = torch.tensor(X_raw[test_ds.indices], dtype=torch.float32)
Y_true = Y_raw[test_ds.indices]
L_test = L_raw[test_ds.indices]

with torch.no_grad():
    Y_pred = np.exp(model(X_te).numpy())
rel_err = np.abs(Y_pred - Y_true) / (np.abs(Y_true) + 1e-30)

print(f'\n{"="*70}')
print('  ReLU discrete-target (R22) — Zhang Table 1')
print(f'{"="*70}')
all_pass = True
for k, sigma in enumerate(SIGMA_LIST, 1):
    sel     = (L_test == sigma)
    r       = rel_err[sel]
    e_mean  = r.mean()
    e_95    = np.percentile(r, 95)
    req     = SIGMA_REQS[sigma]
    pm, p9  = e_mean <= req[0], e_95 <= req[1]
    ok      = 'PASS' if (pm and p9) else 'FAIL'
    all_pass = all_pass and pm and p9
    print(f'  Set {k}  sigma={sigma:<4} N={sel.sum():<5} '
          f'mean={e_mean*100:6.2f}% (<= {req[0]*100:.0f}%)  '
          f'p95={e_95*100:6.2f}% (<= {req[1]*100:.0f}%)  {ok}')
print(f'{"="*70}')
print('MEETS' if all_pass else 'does NOT meet', 'all Zhang Table 1 requirements.')

print('\nPer-target mean relative error (%):')
for j, t in enumerate(TARGET_TIMES):
    print(f'  t={t:.1e}s : {rel_err[:, j].mean()*100:6.2f}%')


# ── Save ──────────────────────────────────────────────────────────────────────

torch.save({
    'model_state':  model.state_dict(),
    'test_indices': list(test_ds.indices),
    'target_times': TARGET_TIMES,
    'train_losses': train_losses, 'val_losses': val_losses,
    'val_epochs':   val_epochs,   'best_val':   best_val,
}, RESULT_PATH)
print(f'Saved: {RESULT_PATH}')
