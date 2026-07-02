#!/usr/bin/env python
# coding: utf-8
#
# nn_toy_1192k_oh.py  —  OH time-history NN surrogate (5 target points, gap-relative)
# Condition : 2216 ppm H2O2 / 1364 ppm H2O / 682 ppm O2 / Ar
#             T = 1192 K, P = 1.95 atm
#
# 5 target points auto-selected from the nominal OH profile (gap-relative fractions):
#   1. Pre-peak    — (initial + peak) / 2                           (rising edge)
#   2. Peak        — t of argmax(OH)                                (peak)
#   3. Inflection  — first zero-crossing of d²OH/dt²  after peak   (inflection)
#   4. Mid-decay   — peak - 0.50 × (peak - initial)               (50% below peak in gap)
#   5. Late-decay  — peak - 0.75 × (peak - initial)               (75% below peak in gap)
#
# Training: Cantera advances directly to each target time (no fixed-step loop).
# Inputs:  x[0]=lnA_R22  x[1]=Ea_R22  x[2]=lnA_R26  x[3]=Ea_R26

import os, time, copy
import cantera as ct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.stats import qmc, norm
from scipy.ndimage import uniform_filter1d

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Mechanism & nominal rate constants ─────────────────────────────────────────

YAML_FILE = 'chem_cti_toy_model_og.yaml'
mol_units = ct.UnitSystem({
    "length": "cm", "mass": "g", "time": "s",
    "quantity": "mol", "pressure": "dyn / cm^2", "energy": "erg",
    "temperature": "K", "current": "A", "activation-energy": "cal / mol"})

IDX_R22 = 21   # H2O2(+M) <=> OH + OH (+M)  — falloff
IDX_R26 = 25   # H2O2 + OH <=> HO2 + H2O    — simple Arrhenius

_gas_nom = ct.Solution(YAML_FILE)
NOMINAL_A_R22      = _gas_nom.reaction(IDX_R22).rate.low_rate.pre_exponential_factor
NOMINAL_B_R22      = _gas_nom.reaction(IDX_R22).rate.low_rate.temperature_exponent
NOMINAL_EA_R22_si  = _gas_nom.reaction(IDX_R22).rate.low_rate.activation_energy
NOMINAL_EA_R22_cal = mol_units.convert_activation_energy_to(
    f"{NOMINAL_EA_R22_si} J/kmol", "cal / mol")
NOMINAL_A_R26      = _gas_nom.reaction(IDX_R26).rate.pre_exponential_factor
NOMINAL_B_R26      = _gas_nom.reaction(IDX_R26).rate.temperature_exponent
NOMINAL_EA_R26_si  = _gas_nom.reaction(IDX_R26).rate.activation_energy
NOMINAL_EA_R26_cal = mol_units.convert_activation_energy_to(
    f"{NOMINAL_EA_R26_si} J/kmol", "cal / mol")
del _gas_nom

print(f'R22: A={NOMINAL_A_R22:.3e}  Ea={NOMINAL_EA_R22_cal:.0f} cal/mol')
print(f'R26: A={NOMINAL_A_R26:.3e}  Ea={NOMINAL_EA_R26_cal:.0f} cal/mol')

LN_F        = 10
SIGMA_E     = 5000.0
PARAM_NAMES = ['lnA_R22', 'Ea_R22', 'lnA_R26', 'Ea_R26']
INPUT_DIM   = 4

TOTAL_SAMPLES = 40000
SIGMA_LIST    = (0.1, 0.3, 0.5)
RATIO_LIST    = (1/6, 1/6, 2/3)

# ── Shock tube condition ───────────────────────────────────────────────────────

T_INITIAL = 1192
P_INITIAL = 1.95 * ct.one_atm
INITIAL_X = {
    'H2O2': 2216e-6,
    'H2O':  1364e-6,
    'O2':   682e-6,
    'AR':   1.0 - (2216 + 1364 + 682) * 1e-6,
}

# Fine grid for nominal profile (used only for target detection)
# Peak is at ~0.017 ms → 0.1 µs step gives ~170 points before peak
DT_FINE  = 1e-8       # 0.01 µs
N_FINE   = 100000      # 1 ms total
T_FINE   = np.linspace(DT_FINE, DT_FINE * N_FINE, N_FINE)

# ── Target selection knobs — adjust here if needed ────────────────────────────

# NEW: Gap-relative target selection (% of rise from initial to peak)
USE_GAP_RELATIVE = True  # Use gap-relative fractions instead of absolute peak fractions
FRAC_MID_DECAY   = 0.50  # Point 4: peak - 0.50 × (peak - initial)
FRAC_LATE_DECAY  = 0.75  # Point 5: peak - 0.75 × (peak - initial)
SMOOTH_WIN       = 51    # smoothing window (samples) for inflection detection

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

CHECKPOINT_PATH = 'ckpt_1192k_oh_5pt_gap.pt'
RESULT_PATH     = 'result_1192k_oh_5pt_gap.pt'
SIGMA_REQS      = {0.1: (0.01, 0.02), 0.3: (0.02, 0.05), 0.5: (0.03, 0.10)}
LOG_EPS         = 1e-12

TARGET_LABELS   = ['pre-peak', 'peak', 'inflection', 'mid-decay (50% gap)', 'late-decay (75% gap)']


# ── Nominal OH profile (fine grid, run once) ──────────────────────────────────

def run_nominal_fine():
    local_gas = ct.Solution(YAML_FILE)
    local_gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
    reactor = ct.IdealGasReactor(local_gas, energy='on')
    net     = ct.ReactorNet([reactor])
    oh_idx  = local_gas.species_index('OH')
    profile = np.empty(N_FINE)
    for k in range(N_FINE):
        net.advance(T_FINE[k])
        profile[k] = reactor.thermo.X[oh_idx]
    return profile


def find_targets_5pt(profile, t_sim):
    """
    Auto-detect 5 target times from the nominal OH profile using gap-relative fractions.

    Returns np.array of 5 times (strictly increasing):
      1. Pre-peak at (initial + peak) / 2         (rising edge)
      2. t of maximum OH                          (peak)
      3. First zero-crossing of d²OH/dt² after peak (inflection)
      4. peak - FRAC_MID_DECAY × (peak - initial) (mid-decay, 50% below peak in gap)
      5. peak - FRAC_LATE_DECAY × (peak - initial) (late-decay, 75% below peak in gap)
    """
    i_peak  = int(np.argmax(profile))
    oh_peak = float(profile[i_peak])
    oh_init = float(profile[0])
    gap     = oh_peak - oh_init

    # Point 1: pre-peak at average of initial and peak
    oh_target_1 = (oh_init + oh_peak) / 2.0
    mask1 = np.abs(profile[:i_peak] - oh_target_1) < gap * 0.02  # tolerance: 2% of gap
    if mask1.any():
        i_1 = int(np.argmax(mask1))
    else:
        i_1 = int(np.argmin(np.abs(profile[:i_peak] - oh_target_1)))
    t_1 = t_sim[i_1]

    # Point 2: peak
    t_2 = t_sim[i_peak]

    # Point 3: inflection — first d²OH/dt²=0 after peak
    prof_d = profile[i_peak:].astype(float)
    t_d    = t_sim[i_peak:]
    w      = min(SMOOTH_WIN, max(5, 2 * (len(prof_d) // 100) + 1))
    if w % 2 == 0:
        w += 1
    smooth = uniform_filter1d(prof_d, size=w)
    d2     = np.gradient(np.gradient(smooth, t_d), t_d)
    skip   = max(3, len(d2) // 50)
    d2_sub = d2[skip:]
    crossings = np.where(np.diff(np.sign(d2_sub)))[0]
    if crossings.size > 0:
        i_infl_rel = skip + crossings[0] + 1
    else:
        # No zero-crossing (pure decay); use max |d²| in first third as proxy
        third = max(1, len(d2_sub) // 3)
        i_infl_rel = skip + int(np.argmax(np.abs(d2_sub[:third])))
        print('  [INFO] No inflection zero-crossing; using max |d²OH/dt²| as proxy.')
    i_infl = i_peak + min(i_infl_rel, len(prof_d) - 1)
    t_3    = t_sim[min(i_infl, len(t_sim) - 1)]

    # Point 4: mid-decay at peak - FRAC_MID_DECAY × gap
    oh_target_4 = oh_peak - FRAC_MID_DECAY * gap
    mask4 = np.abs(profile[i_peak:] - oh_target_4) < gap * 0.02  # tolerance: 2% of gap
    if mask4.any():
        i_4 = i_peak + int(np.argmax(mask4))
        t_4 = t_sim[i_4]
    else:
        i_4 = i_peak + int(np.argmin(np.abs(profile[i_peak:] - oh_target_4)))
        t_4 = t_sim[i_4]
        print(f'  [WARN] Mid-decay target {oh_target_4:.1f} ppm not found; using closest at {profile[i_4]*1e6:.1f} ppm')

    # Point 5: late-decay at peak - FRAC_LATE_DECAY × gap
    oh_target_5 = oh_peak - FRAC_LATE_DECAY * gap
    mask5 = np.abs(profile[i_peak:] - oh_target_5) < gap * 0.02  # tolerance: 2% of gap
    if mask5.any():
        i_5 = i_peak + int(np.argmax(mask5))
        t_5 = t_sim[i_5]
    else:
        i_5 = i_peak + int(np.argmin(np.abs(profile[i_peak:] - oh_target_5)))
        t_5 = t_sim[i_5]
        if abs(profile[i_5] - oh_target_5) > gap * 0.10:  # warn if > 10% away
            print(f'  [WARN] Late-decay target {oh_target_5:.1f} ppm not well-reached; using closest at {profile[i_5]*1e6:.1f} ppm')

    # Enforce strict ordering (guard against detection collisions)
    times = np.array([t_1, t_2, t_3, t_4, t_5])
    dt_guard = DT_FINE * 2
    for k in range(1, len(times)):
        if times[k] <= times[k - 1]:
            times[k] = times[k - 1] + dt_guard
    return times


print('\nRunning nominal OH profile (fine grid) ...')
_nom = run_nominal_fine()
_i_pk = int(np.argmax(_nom))
print(f'  OH_peak = {_nom[_i_pk]:.3e} mol/mol  ({_nom[_i_pk]*1e6:.1f} ppm)'
      f'  at t = {T_FINE[_i_pk]*1e3:.4f} ms')

TARGET_TIMES = find_targets_5pt(_nom, T_FINE)
N_TARGETS    = len(TARGET_TIMES)

print('\nSelected target times:')
for j, (lbl, t_tgt) in enumerate(zip(TARGET_LABELS, TARGET_TIMES)):
    i_near = int(np.argmin(np.abs(T_FINE - t_tgt)))
    oh_ppm = _nom[i_near] * 1e6
    print(f'  {j+1}. {lbl:<12}  t = {t_tgt*1e3:.4f} ms   '
          f'OH ≈ {oh_ppm:.1f} ppm  ({oh_ppm/_nom[_i_pk]/1e6*100:.1f}% of peak)')


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


# ── Cantera simulation — advance directly to target times ──────────────────────

def run_single(x_vec):
    try:
        local_gas = ct.Solution(YAML_FILE)

        # R22 — falloff: perturb low-rate A and Ea
        new_A_R22  = NOMINAL_A_R22 * np.exp(x_vec[0] * LN_F)
        new_Ea_R22 = (NOMINAL_EA_R22_cal + x_vec[1] * SIGMA_E) * 4184.0
        rxn22 = local_gas.reaction(IDX_R22)
        rxn22.rate.low_rate = ct.Arrhenius(new_A_R22, NOMINAL_B_R22, new_Ea_R22)
        local_gas.modify_reaction(IDX_R22, rxn22)

        # R26 — simple Arrhenius: perturb A and Ea
        new_A_R26  = NOMINAL_A_R26 * np.exp(x_vec[2] * LN_F)
        new_Ea_R26 = (NOMINAL_EA_R26_cal + x_vec[3] * SIGMA_E) * 4184.0
        rxn26 = local_gas.reaction(IDX_R26)
        rxn26.rate = ct.Arrhenius(new_A_R26, NOMINAL_B_R26, new_Ea_R26)
        local_gas.modify_reaction(IDX_R26, rxn26)

        local_gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
        reactor = ct.IdealGasReactor(local_gas, energy='on')
        net     = ct.ReactorNet([reactor])
        oh_idx  = local_gas.species_index('OH')

        y_out = np.empty(N_TARGETS)
        for j, t_tgt in enumerate(TARGET_TIMES):
            net.advance(t_tgt)
            y_out[j] = reactor.thermo.X[oh_idx]
        return True, y_out
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


# ── Log transform & train/val/test split ──────────────────────────────────────

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


# ── Architecture — one hidden layer, ReLU ─────────────────────────────────────

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
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = criterion(model(xb), yb)
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


print('\nTraining (1192K OH, 5-pt gap-relative, ReLU) ...')
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


# ── Evaluation — Zhang Table 1 ────────────────────────────────────────────────

model.eval()
X_te   = torch.tensor(X_raw[test_ds.indices], dtype=torch.float32)
Y_true = Y_raw[test_ds.indices]
L_test = L_raw[test_ds.indices]

with torch.no_grad():
    Y_pred = np.exp(model(X_te).numpy())
rel_err = np.abs(Y_pred - Y_true) / (np.abs(Y_true) + 1e-30)

print(f'\n{"="*70}')
print('  1192K OH — 5-pt gap-relative ReLU  |  Zhang Table 1')
print(f'{"="*70}')
all_pass = True
for k, sigma in enumerate(SIGMA_LIST, 1):
    sel    = (L_test == sigma)
    r      = rel_err[sel]
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

print('\nPer-target mean relative error (%):')
for j, (lbl, t_tgt) in enumerate(zip(TARGET_LABELS, TARGET_TIMES)):
    print(f'  {j+1}. {lbl:<12}  t={t_tgt*1e3:.4f} ms   '
          f'err={rel_err[:, j].mean()*100:.2f}%')


# ── Save ───────────────────────────────────────────────────────────────────────

torch.save({
    'model_state':    model.state_dict(),
    'test_indices':   list(test_ds.indices),
    'target_times':   TARGET_TIMES,
    'target_labels':  TARGET_LABELS,
    'train_losses':   train_losses,
    'val_losses':     val_losses,
    'val_epochs':     val_epochs,
    'best_val':       best_val,
}, RESULT_PATH)
print(f'\nSaved: {RESULT_PATH}')
