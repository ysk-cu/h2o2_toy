#!/usr/bin/env python
# coding: utf-8
#
# nn_toy_1192k_oh_train.py
# OH surrogate, 1192 K / 1.95 atm  (Hong cond. 1 -> Burke Fig. S3a)
# Mixture: 2216 ppm H2O2 / 1364 ppm H2O / 682 ppm O2 / Ar
#
# Effective OH target points (3), chosen by information content:
#   1. rise        (~50% up the rise)    -> constrains k1 (formation, R22)
#   2. peak        (argmax OH)           -> constrains k2/k1 ratio  ([OH]peak diagnostic)
#   3. mid-decay   (33% down the decay)  -> constrains k2 (removal, R26)
# At 1192 K the OH decay is k2-controlled (k5/R5 is minor below ~1176 K), so the
# two decay points are what give k2 its identifiability here.
#
# Active parameters (MUST match the H2O surrogate for a joint fit):
#   x[0]=lnA_R22  x[1]=Ea_R22  x[2]=lnA_R26  x[3]=Ea_R26
#
# NOTE 1 (consistency): LN_F and SIGMA_E below define the x-normalization. They MUST
#   be identical to the values used in nn_toy_1192k_h2o_infer.py, or the shared x in
#   the joint MAP step is meaningless. LN_F = 10 is kept here only for drop-in
#   compatibility with the existing H2O surrogate; it is far wider than the physical
#   prior. For the MAP step, retrain BOTH surrogates with LN_F = ln(physical 2-sigma
#   factor) ~ 0.4-0.7 and carry the true prior sigma into the residual.
# NOTE 2 (sigma_obs): the observational uncertainty sigma_log = 0.05 (both species)
#   is used in the inference residual, NOT in training. Here the surrogate is fit to
#   the Cantera log-targets; its log-error should land well under 0.05 (Zhang Table 1
#   accuracy already guarantees this).
# NOTE 3 (activation): GELU is used instead of ReLU so the surrogate has a smooth
#   dy/dx for the downstream gradient-based MAP optimizer (ReLU kinks aggravate the
#   identifiability ridge). Ideally retrain the H2O surrogate with GELU too.

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

IDX_R22 = 21   # H2O2(+M) <=> OH + OH (+M)  — falloff   (k1)
IDX_R26 = 25   # H2O2 + OH <=> HO2 + H2O    — Arrhenius (k2, single-channel here)

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

# ── Parameter normalization (keep identical to the H2O surrogate) ─────────────

LN_F        = np.log(10)        # A = A0 * exp(x * LN_F);  see NOTE 1
SIGMA_E     = 5000.0    # Ea(cal/mol) = Ea0 + x * SIGMA_E
PARAM_NAMES = ['lnA_R22', 'Ea_R22', 'lnA_R26', 'Ea_R26']
INPUT_DIM   = 4

# ── Multi-scale Sobol sampling ────────────────────────────────────────────────

TOTAL_SAMPLES = 40000
SIGMA_LIST    = (0.1, 0.3, 0.5)
RATIO_LIST    = (1/6, 1/6, 2/3)
# RATIO_LIST    = (1/4, 1/4, 1/2)
 
# ── Shock-tube condition (Hong cond. 1 / Fig. S3) ─────────────────────────────

T_INITIAL = 1192
P_INITIAL = 1.95 * ct.one_atm
INITIAL_X = {'H2O2': 2216e-6, 'H2O': 1364e-6, 'O2': 682e-6,
             'AR':   1.0 - (2216 + 1364 + 682) * 1e-6}

# constant pressure–enthalpy reactor (matches Hong's constant P-H assumption)
DT_FINE = 1e-7      # 0.1 us resolution for nominal target selection
N_FINE  = 10000     # 1.0 ms window
T_FINE  = np.linspace(DT_FINE, DT_FINE * N_FINE, N_FINE)

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

CHECKPOINT_PATH = 'ckpt_1192k_oh_train2.pt'
RESULT_PATH     = 'result_1192k_oh_train2.pt'
SIGMA_REQS      = {0.1: (0.01, 0.02), 0.3: (0.02, 0.05), 0.5: (0.03, 0.10)}
LOG_EPS         = 1e-12

TARGET_LABELS = ['rise (50% up)', '[OH]_peak', 'OH @ 2·t_peak', 'OH @ 0.8ms']


# ── Nominal OH profile & effective target-time selection ──────────────────────

def run_nominal_oh():
    gas = ct.Solution(YAML_FILE)
    gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
    reactor = ct.IdealGasConstPressureReactor(gas, energy='on')
    net     = ct.ReactorNet([reactor])
    oh_idx  = gas.species_index('OH')
    profile = np.empty(N_FINE)
    for k in range(N_FINE):
        net.advance(T_FINE[k])
        profile[k] = reactor.thermo.X[oh_idx]
    return profile


def find_oh_targets(profile, t_sim):
    """Return 6 fixed nominal times for the target probes."""
    i_peak  = int(np.argmax(profile))
    oh_peak = float(profile[i_peak])
    oh_init = float(profile[0])
    t_peak  = float(t_sim[i_peak])
    rise_gap = oh_peak - oh_init

    i1 = int(np.argmin(np.abs(profile[:i_peak] - (oh_init + 0.50 * rise_gap))))

    times = np.array([
        t_sim[i1],    # P1: rise (50%)
        t_peak,       # P2: [OH]_peak
        2.0 * t_peak, # P3: OH @ 2·t_peak
        8e-4          # P4: OH @ 0.8ms 
    ])
    dt_guard = (t_sim[1] - t_sim[0]) * 2
    for k in range(1, len(times)):
        if times[k] <= times[k - 1]:
            times[k] = times[k - 1] + dt_guard
    return times


print('\nRunning nominal OH profile (1 ms, const-P) ...')
_nom  = run_nominal_oh()
_i_pk = int(np.argmax(_nom))
print(f'  OH_peak = {_nom[_i_pk]*1e6:.1f} ppm  at t = {T_FINE[_i_pk]*1e3:.4f} ms')

TARGET_TIMES = find_oh_targets(_nom, T_FINE)
N_TARGETS    = len(TARGET_TIMES)


print('\nSelected OH target probes (nominal):')
_nom_vals = [
    np.interp(TARGET_TIMES[0], T_FINE, _nom) * 1e6,   # rise OH [ppm]
    _nom[_i_pk] * 1e6,                                 # [OH]_peak [ppm]
    np.interp(TARGET_TIMES[2], T_FINE, _nom) * 1e6,   # OH @ 2·t_peak [ppm]
    np.interp(TARGET_TIMES[3], T_FINE, _nom) * 1e6,   # OH @ 0.8ms [ppm]
]
_units = ['ppm', 'ppm', 'ppm', 'ppm']
for j, (lbl, val, unit) in enumerate(zip(TARGET_LABELS, _nom_vals, _units)):
    print(f'  {j+1}. {lbl:<22}  nominal ~ {val:.2f} {unit}')


# ── Sobol sampling ─────────────────────────────────────────────────────────────

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


# ── Cantera simulation — 4 target probes (rise/OH_peak/2×t_peak/0.8ms) ──────

def run_single(x_vec):
    try:
        gas = ct.Solution(YAML_FILE)
        # R22 (k1) — falloff low-rate
        new_A_R22  = NOMINAL_A_R22 * np.exp(x_vec[0] * LN_F)
        new_Ea_R22 = (NOMINAL_EA_R22_cal + x_vec[1] * SIGMA_E) * 4184.0
        rxn22 = gas.reaction(IDX_R22)
        rxn22.rate.low_rate = ct.Arrhenius(new_A_R22, NOMINAL_B_R22, new_Ea_R22)
        gas.modify_reaction(IDX_R22, rxn22)
        # R26 (k2) — simple Arrhenius
        new_A_R26  = NOMINAL_A_R26 * np.exp(x_vec[2] * LN_F)
        new_Ea_R26 = (NOMINAL_EA_R26_cal + x_vec[3] * SIGMA_E) * 4184.0
        rxn26 = gas.reaction(IDX_R26)
        rxn26.rate = ct.Arrhenius(new_A_R26, NOMINAL_B_R26, new_Ea_R26)
        gas.modify_reaction(IDX_R26, rxn26)

        gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
        reactor = ct.IdealGasConstPressureReactor(gas, energy='on')
        net     = ct.ReactorNet([reactor])
        oh_idx  = gas.species_index('OH')

        y_out = np.empty(N_TARGETS)
        for k, t in enumerate(TARGET_TIMES):
            net.advance(t)
            y_out[k] = reactor.thermo.X[oh_idx]
        return True, y_out
    except Exception:
        return False, None


# ── Checkpoint / run simulations ──────────────────────────────────────────────

if os.path.exists(CHECKPOINT_PATH):
    ckpt     = torch.load(CHECKPOINT_PATH, weights_only=False)
    raw_y    = ckpt['y_list']; raw_x = ckpt['x_list']
    raw_y    = [np.asarray(y)[:N_TARGETS] if np.asarray(y).ndim == 1 and len(y) >= N_TARGETS else y for y in raw_y]
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
                    'target_times': TARGET_TIMES}, CHECKPOINT_PATH)

torch.save({'last_index': TOTAL_SAMPLES - 1, 'y_list': y_list, 'x_list': x_list,
            'l_list': l_list, 'n_failed': n_failed,
            'target_times': TARGET_TIMES}, CHECKPOINT_PATH)

X_raw = np.asarray(x_list); Y_raw = np.asarray(y_list); L_raw = np.asarray(l_list)
print(f'\nShapes: X={X_raw.shape}  Y={Y_raw.shape}  (failed={n_failed})')


# ── Log transform & split ─────────────────────────────────────────────────────

Y_log = np.log(Y_raw + LOG_EPS)        # log-MSE matches constant sigma_log obs model
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


# ── Architecture — GELU, 4 inputs -> N_TARGETS outputs ────────────────────────

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
            loss.backward(); optimizer.step()
        else:
            with torch.no_grad():
                loss = criterion(model(xb), yb)
        total += loss.item() * len(xb)
    return total / len(loader.dataset)

print('\nTraining (1192 K OH, 5-pt: rise/OH_peak/2×t_peak/5×t_peak/0.1ms, ReLU) ...')
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
print('  1192 K OH — 5-pt (rise/OH_peak/2×t_peak/5×t_peak/0.5ms)  |  Zhang Table 1')
print(f'{"="*72}')
all_pass = True
for k, sigma in enumerate(SIGMA_LIST, 1):
    sel    = (L_test == sigma)
    r      = rel_err[sel]
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

print('\nPer-target mean relative error (%):')
for j, (lbl, t_tgt) in enumerate(zip(TARGET_LABELS, TARGET_TIMES)):
    print(f'  {j+1}. {lbl:<22} t={t_tgt*1e3:.4f} ms   err={rel_err[:, j].mean()*100:.2f}%')


# ── Save ──────────────────────────────────────────────────────────────────────

torch.save({
    'model_state':   model.state_dict(),
    'test_indices':  list(test_ds.indices),
    'target_times':  TARGET_TIMES,
    'target_labels': TARGET_LABELS,
    'hidden_dim':    HIDDEN_DIM,
    'n_targets':     N_TARGETS,
    'ln_f':          LN_F,
    'sigma_e':       SIGMA_E,
    'train_losses':  train_losses,
    'val_losses':    val_losses,
    'val_epochs':    val_epochs,
    'best_val':      best_val,
}, RESULT_PATH)
print(f'\nSaved: {RESULT_PATH}')
