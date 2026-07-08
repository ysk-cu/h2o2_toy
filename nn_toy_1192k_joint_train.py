#!/usr/bin/env python
# coding: utf-8
#
# nn_toy_1192k_joint_train.py
# Joint OH + H2O surrogate, 1192 K / 1.95 atm  (Hong cond. 1)
# Mixture: 2216 ppm H2O2 / 1364 ppm H2O / 682 ppm O2 / Ar
#
# Single NN: x[0..3] -> [N_OH_TARGETS OH probes | N_H2O_TARGETS H2O probes]
#
# OH probes  (5): rise(50%), [OH]_peak, OH @ 2*t_peak, OH @ 5*t_peak, OH @ 0.5ms
# H2O probes (3): H2O at 20 / 60 / 95% of the monotone H2O rise
#
# Each Cantera run advances once through all 7 target times in chronological
# order, recording OH and H2O simultaneously -> single dataset for both species.
#
# Active parameters (identical to the separate surrogates for joint MAP):
#   x[0]=lnA_R22  x[1]=Ea_R22  x[2]=lnA_R26  x[3]=Ea_R26
#
# NOTE: LN_F and SIGMA_E MUST remain identical to any separately-trained
#   surrogate if you intend to use this in a joint MAP step with that surrogate.
#   Changing them here while keeping them at 10 / 5000 in the other surrogate
#   will corrupt the shared x vector during optimization.

import os, time, copy
import cantera as ct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.stats import qmc, norm

torch.set_num_threads(16)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Mechanism & nominal rate constants ────────────────────────────────────────

YAML_FILE = 'chem_cti_toy_model_og.yaml'
mol_units = ct.UnitSystem({
    "length": "cm", "mass": "g", "time": "s",
    "quantity": "mol", "pressure": "dyn / cm^2", "energy": "erg",
    "temperature": "K", "current": "A", "activation-energy": "cal / mol"})

IDX_R22 = 21   # H2O2(+M) <=> OH + OH (+M)  -- falloff  (k1)
IDX_R26 = 25   # H2O2 + OH <=> HO2 + H2O    -- Arrhenius (k2)

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

print(f'R22 (k1): A={NOMINAL_A_R22:.3e}  Ea={NOMINAL_EA_R22_cal:.0f} cal/mol')
print(f'R26 (k2): A={NOMINAL_A_R26:.3e}  Ea={NOMINAL_EA_R26_cal:.0f} cal/mol')

# ── Parameter normalization ───────────────────────────────────────────────────

LN_F        = 3       # A = A0 * exp(x * LN_F)
SIGMA_E     = 5000.0   # Ea(cal/mol) = Ea0 + x * SIGMA_E
PARAM_NAMES = ['lnA_R22', 'Ea_R22', 'lnA_R26', 'Ea_R26']
INPUT_DIM   = 4

# ── Multi-scale Sobol sampling ────────────────────────────────────────────────

TOTAL_SAMPLES = 40000
SIGMA_LIST    = (0.1, 0.3, 0.5)
RATIO_LIST    = (1/6, 1/6, 2/3)

# ── Shock-tube condition ──────────────────────────────────────────────────────

T_INITIAL = 1192
P_INITIAL = 1.95 * ct.one_atm
INITIAL_X = {'H2O2': 2216e-6, 'H2O': 1364e-6, 'O2': 682e-6,
             'AR':   1.0 - (2216 + 1364 + 682) * 1e-6}

# Fine nominal grid for OH peak / rise detection (0.1 us resolution)
DT_FINE = 1e-7
N_FINE  = 10000
T_FINE  = np.linspace(DT_FINE, DT_FINE * N_FINE, N_FINE)   # 0.1 us ... 1 ms

# Coarser grid for H2O rise detection (1 us resolution)
DT_SIM  = 1e-6
N_STEPS = 1000
T_SIM   = np.linspace(DT_SIM, DT_SIM * N_STEPS, N_STEPS)   # 1 us ... 1 ms

# ── NN / training hyper-parameters ───────────────────────────────────────────

HIDDEN_DIM     = 16    # larger than either separate surrogate (was 16/32)
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

# Per-species loss weight: increase WEIGHT_OH if OH errors dominate after training
WEIGHT_OH  = 1.0
WEIGHT_H2O = 1.0

CHECKPOINT_PATH = 'ckpt_1192k_joint_train.pt'
RESULT_PATH     = 'result_1192k_joint_train.pt'
SIGMA_REQS      = {0.1: (0.01, 0.02), 0.3: (0.02, 0.05), 0.5: (0.03, 0.10)}
LOG_EPS         = 1e-12

# ── Nominal profiles & target-time selection ──────────────────────────────────

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


def find_oh_targets(profile, t_sim):
    i_peak   = int(np.argmax(profile))
    oh_peak  = float(profile[i_peak])
    oh_init  = float(profile[0])
    t_peak   = float(t_sim[i_peak])
    rise_gap = oh_peak - oh_init
    i1 = int(np.argmin(np.abs(profile[:i_peak] - (oh_init + 0.50 * rise_gap))))
    times = np.array([
        t_sim[i1],    # P1: rise 50%
        t_peak,       # P2: [OH]_peak
        2.0 * t_peak, # P3: OH @ 2*t_peak
        5.0 * t_peak, # P4: OH @ 5*t_peak
        0.5e-3,       # P5: OH @ 0.5ms (late decay)
    ])
    dt_guard = (t_sim[1] - t_sim[0]) * 2
    for k in range(1, len(times)):
        if times[k] <= times[k - 1]:
            times[k] = times[k - 1] + dt_guard
    return times


def find_h2o_targets(profile, t_sim, fracs=(0.20, 0.60, 0.95)):
    h2o_0   = float(profile[0])
    h2o_inf = float(profile[-200:].mean())
    targets = h2o_0 + np.array(fracs) * (h2o_inf - h2o_0)
    times   = np.array([
        t_sim[np.argmax(profile >= xt)] if np.any(profile >= xt) else t_sim[-1]
        for xt in targets
    ])
    return times, np.array(fracs)


OH_TARGET_TIMES              = find_oh_targets(_nom_oh,  T_FINE)
H2O_TARGET_TIMES, H2O_FRACS = find_h2o_targets(_nom_h2o, T_SIM)
N_OH_TARGETS  = len(OH_TARGET_TIMES)
N_H2O_TARGETS = len(H2O_TARGET_TIMES)
N_TARGETS     = N_OH_TARGETS + N_H2O_TARGETS   # total NN outputs

OH_LABELS  = ['rise (50% up)', '[OH]_peak', 'OH @ 2*t_peak', 'OH @ 5*t_peak', 'OH @ 0.5ms']
H2O_LABELS = [f'H2O {int(f*100)}% rise' for f in H2O_FRACS]

print('\nOH target probes (nominal):')
oh_nom_vals = [np.interp(t, T_FINE, _nom_oh) * 1e6 for t in OH_TARGET_TIMES]
for j, (lbl, val, t) in enumerate(zip(OH_LABELS, oh_nom_vals, OH_TARGET_TIMES)):
    print(f'  {j+1}. {lbl:<22}  t={t*1e3:.4f} ms   OH ~ {val:.2f} ppm')

print('\nH2O target probes (nominal):')
h2o_nom_vals = [np.interp(t, T_SIM, _nom_h2o) * 1e6 for t in H2O_TARGET_TIMES]
for j, (lbl, val, t) in enumerate(zip(H2O_LABELS, h2o_nom_vals, H2O_TARGET_TIMES)):
    print(f'  {j+1}. {lbl:<22}  t={t*1e3:.4f} ms   H2O ~ {val:.2f} ppm')

# ── Advance order: merge OH and H2O target times chronologically ──────────────
# Each Cantera run advances once through this sorted list, recording both species.

_all_times  = np.concatenate([OH_TARGET_TIMES, H2O_TARGET_TIMES])
_species_id = np.array([0] * N_OH_TARGETS + [1] * N_H2O_TARGETS)  # 0=OH, 1=H2O
_sort_idx   = np.argsort(_all_times)
_sorted_times   = _all_times[_species_id]    # will be rebuilt below
_advance_times  = _all_times[_sort_idx]      # chronological advance times
_advance_species = _species_id[_sort_idx]    # which species to record at each step
_advance_pos    = np.zeros(len(_sort_idx), dtype=int)

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
        # R22 (k1) -- falloff low-rate
        new_A_R22  = NOMINAL_A_R22 * np.exp(x_vec[0] * LN_F)
        new_Ea_R22 = (NOMINAL_EA_R22_cal + x_vec[1] * SIGMA_E) * 4184.0
        rxn22 = gas.reaction(IDX_R22)
        rxn22.rate.low_rate = ct.Arrhenius(new_A_R22, NOMINAL_B_R22, new_Ea_R22)
        gas.modify_reaction(IDX_R22, rxn22)
        # R26 (k2) -- simple Arrhenius
        new_A_R26  = NOMINAL_A_R26 * np.exp(x_vec[2] * LN_F)
        new_Ea_R26 = (NOMINAL_EA_R26_cal + x_vec[3] * SIGMA_E) * 4184.0
        rxn26 = gas.reaction(IDX_R26)
        rxn26.rate = ct.Arrhenius(new_A_R26, NOMINAL_B_R26, new_Ea_R26)
        gas.modify_reaction(IDX_R26, rxn26)

        gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
        reactor  = ct.IdealGasConstPressureReactor(gas, energy='on')
        net      = ct.ReactorNet([reactor])
        oh_idx   = gas.species_index('OH')
        h2o_idx  = gas.species_index('H2O')

        y_oh  = np.empty(N_OH_TARGETS)
        y_h2o = np.empty(N_H2O_TARGETS)

        # Single chronological pass through all target times
        for k in range(len(_advance_times)):
            net.advance(_advance_times[k])
            out_pos = _advance_out_idx[k]
            if _advance_species[k] == 0:
                y_oh[out_pos]  = reactor.thermo.X[oh_idx]
            else:
                y_h2o[out_pos] = reactor.thermo.X[h2o_idx]

        y_out = np.concatenate([y_oh, y_h2o])
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
                    'h2o_target_times': H2O_TARGET_TIMES}, CHECKPOINT_PATH)

torch.save({'last_index': TOTAL_SAMPLES - 1, 'y_list': y_list, 'x_list': x_list,
            'l_list': l_list, 'n_failed': n_failed,
            'oh_target_times':  OH_TARGET_TIMES,
            'h2o_target_times': H2O_TARGET_TIMES}, CHECKPOINT_PATH)

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


# ── Architecture: 4 -> HIDDEN_DIM -> N_TARGETS ───────────────────────────────

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


# ── Loss: weighted per-species relative error ─────────────────────────────────

class JointRelativeErrorLoss(nn.Module):
    def __init__(self, n_oh, n_h2o, w_oh=1.0, w_h2o=1.0):
        super().__init__()
        self.n_oh  = n_oh
        self.n_h2o = n_h2o
        self.w_oh  = w_oh
        self.w_h2o = w_h2o

    def forward(self, pred, target):
        pred_raw   = torch.exp(pred)
        target_raw = torch.exp(target)
        rel_err    = torch.abs(pred_raw - target_raw) / (target_raw.abs() + 1e-30)
        loss_oh    = rel_err[:, :self.n_oh].mean()
        loss_h2o   = rel_err[:, self.n_oh:].mean()
        return self.w_oh * loss_oh + self.w_h2o * loss_h2o


criterion = JointRelativeErrorLoss(N_OH_TARGETS, N_H2O_TARGETS,
                                   w_oh=WEIGHT_OH, w_h2o=WEIGHT_H2O)

# ── Training loop ──────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
model     = SurrogateNN()
optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=wd_init)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR,
    patience=lr_check // val_check, min_lr=LR_MIN)

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


print(f'\nTraining (joint OH+H2O, hidden={HIDDEN_DIM}, '
      f'w_OH={WEIGHT_OH}, w_H2O={WEIGHT_H2O}, ReLU) ...')
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


# ── Zhang Table 1 evaluation — per species ────────────────────────────────────

model.eval()
X_te   = torch.tensor(X_raw[test_ds.indices], dtype=torch.float32)
Y_true = Y_raw[test_ds.indices]
L_test = L_raw[test_ds.indices]

with torch.no_grad():
    Y_pred = np.exp(model(X_te).numpy())

rel_err     = np.abs(Y_pred - Y_true) / (np.abs(Y_true) + 1e-30)
rel_err_oh  = rel_err[:, :N_OH_TARGETS]
rel_err_h2o = rel_err[:, N_OH_TARGETS:]


def zhang_table(label, rel_err_sp):
    print(f'\n{"="*72}')
    print(f'  {label}  |  Zhang Table 1')
    print(f'{"="*72}')
    all_pass = True
    for k, sigma in enumerate(SIGMA_LIST, 1):
        sel    = (L_test == sigma)
        r      = rel_err_sp[sel]
        e_mean = r.mean(); e_95 = np.percentile(r, 95)
        req    = SIGMA_REQS[sigma]
        pm, p9 = e_mean <= req[0], e_95 <= req[1]
        ok     = 'PASS' if (pm and p9) else 'FAIL'
        all_pass = all_pass and pm and p9
        print(f'  Set {k}  sigma={sigma:<4} N={sel.sum():<6} '
              f'mean={e_mean*100:6.2f}% (<= {req[0]*100:.0f}%)  '
              f'p95={e_95*100:6.2f}% (<= {req[1]*100:.0f}%)  {ok}')
    print(f'{"="*72}')
    print('MEETS' if all_pass else 'does NOT meet', 'all Zhang Table 1 requirements.')
    return all_pass


pass_oh  = zhang_table('1192 K OH  (joint NN)', rel_err_oh)
pass_h2o = zhang_table('1192 K H2O (joint NN)', rel_err_h2o)

print('\nPer-OH-target mean relative error (%):')
for j, (lbl, t_tgt) in enumerate(zip(OH_LABELS, OH_TARGET_TIMES)):
    print(f'  {j+1}. {lbl:<22} t={t_tgt*1e3:.4f} ms   err={rel_err_oh[:, j].mean()*100:.2f}%')

print('\nPer-H2O-target mean relative error (%):')
for j, (lbl, t_tgt) in enumerate(zip(H2O_LABELS, H2O_TARGET_TIMES)):
    print(f'  {j+1}. {lbl:<22} t={t_tgt*1e3:.4f} ms   err={rel_err_h2o[:, j].mean()*100:.2f}%')

print(f'\nOverall: OH {"PASS" if pass_oh else "FAIL"}  |  H2O {"PASS" if pass_h2o else "FAIL"}')


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
    'n_targets':        N_TARGETS,
    'ln_f':             LN_F,
    'sigma_e':          SIGMA_E,
    'weight_oh':        WEIGHT_OH,
    'weight_h2o':       WEIGHT_H2O,
    'train_losses':     train_losses,
    'val_losses':       val_losses,
    'val_epochs':       val_epochs,
    'best_val':         best_val,
}, RESULT_PATH)
print(f'\nSaved: {RESULT_PATH}')
