#!/usr/bin/env python
# coding: utf-8
#
# nn_toy_1192k_joint_infer.py
# Stack the OH and H2O surrogates into ONE MAP estimate of the shared rate
# parameters x = [lnA_R22, Ea_R22, lnA_R26, Ea_R26]  (k1 A/Ea, k2 A/Ea).
#
# Why stacking: H2O constrains k1 but is blind to k2 (flat ridge); OH constrains
# the k1/k2 balance (opposite-sign sensitivities). Only the joint fit pins both.
# The posterior covariance Sigma* = (J^T Sigma_obs^-1 J + Sigma_prior^-1)^-1 is
# computed from the FULL stacked Jacobian; it cannot be recovered from two
# separate single-species fits.
#
# Observation model: log-space, sigma_log = 0.05 for every OH and H2O point.
#   residual = [ (g_OH(x)  - logOH_obs )/0.05 ;
#               (g_H2O(x) - logH2O_obs)/0.05 ;
#                x / sigma_prior_x ]            <- MAP prior block (optional)
# Both surrogates already OUTPUT log(species), so no exp() is needed here.

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares

OH_RESULT  = 'result_1192k_oh_train.pt'     # from nn_toy_1192k_oh_train.py  (GELU)
H2O_RESULT = 'result_1192k_h2o_infer.pt'    # from nn_toy_1192k_h2o_infer.py (ReLU)

SIG_LOG  = 0.05          # observational uncertainty in log space (both species)
INPUT_DIM = 4
PARAM_NAMES = ['lnA_R22 (k1 A)', 'Ea_R22 (k1 Ea)', 'lnA_R26 (k2 A)', 'Ea_R26 (k2 Ea)']

# x-normalization — MUST match what the surrogates were trained with
LN_F    = 10.0
SIGMA_E = 5000.0

# Physical prior (2-sigma) used ONLY to scale the prior block and report reduction.
# Convert physical uncertainty -> sigma in x-units:
#   A-factor: x = ln(A/A0)/LN_F, so a 2-sigma factor f_2s gives 1-sigma_x = ln(f_2s)/(2*LN_F)
#   Ea:       x = (Ea-Ea0)/SIGMA_E, so 1-sigma_x = sigma_Ea_phys / SIGMA_E
F2S_A      = 2.0      # 2-sigma uncertainty FACTOR on each A (placeholder; set physically)
SIG_EA_PHYS = 4000.0  # 1-sigma on each Ea, cal/mol (placeholder; set physically)
SIGMA_PRIOR_X = np.array([
    np.log(F2S_A) / (2 * LN_F),   # k1 A
    SIG_EA_PHYS / SIGMA_E,        # k1 Ea
    np.log(F2S_A) / (2 * LN_F),   # k2 A
    SIG_EA_PHYS / SIGMA_E,        # k2 Ea
])

USE_PRIOR = True          # True -> MAP; False -> data-only (still well-posed once stacked)
SYNTHETIC = True          # True -> self-test from surrogates; False -> use real obs below
X_TRUE    = np.array([0.05, 0.0, -0.05, 0.0])   # synthetic "truth" for the self-test


# ── Surrogate definition (activation MUST match how each net was trained) ──────
# The activation has no learnable params, so loading a ReLU-trained state_dict into
# a GELU net silently gives WRONG outputs. Match each net to its training script.

class SurrogateNN(nn.Module):
    def __init__(self, n_out, hidden=16, act='gelu'):
        super().__init__()
        a = nn.GELU() if act == 'gelu' else nn.ReLU()
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, hidden), a, nn.Linear(hidden, n_out))
    def forward(self, x):
        return self.net(x)


def load_surrogate(path, act):
    ckpt   = torch.load(path, weights_only=False)
    tt     = np.asarray(ckpt['target_times'])
    n_out  = int(ckpt.get('n_targets', len(tt)))
    hidden = int(ckpt.get('hidden_dim', 16))
    net = SurrogateNN(n_out=n_out, hidden=hidden, act=act)
    net.load_state_dict(ckpt['model_state'])
    net.eval()
    return net, tt


oh_net,  oh_times  = load_surrogate(OH_RESULT,  act='gelu')   # trained with GELU
h2o_net, h2o_times = load_surrogate(H2O_RESULT, act='relu')   # trained with ReLU
print(f'OH  surrogate: {len(oh_times)} targets at (ms) {np.round(oh_times*1e3,3)}')
print(f'H2O surrogate: {len(h2o_times)} targets at (ms) {np.round(h2o_times*1e3,3)}')


def predict_log(net, x):
    """Return the surrogate's log-species vector for a single x (numpy)."""
    with torch.no_grad():
        return net(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).numpy().ravel()


# ── Observations (log space) ──────────────────────────────────────────────────

if SYNTHETIC:
    rng = np.random.default_rng(0)
    logOH_obs  = predict_log(oh_net,  X_TRUE) + rng.normal(0, SIG_LOG, len(oh_times))
    logH2O_obs = predict_log(h2o_net, X_TRUE) + rng.normal(0, SIG_LOG, len(h2o_times))
    print(f'\n[synthetic self-test] X_TRUE = {X_TRUE}')
else:
    # ---- REAL DATA HOOK -------------------------------------------------------
    # Digitize OH and H2O from Hong Fig. S3, interpolate onto the SAME target
    # times the surrogates were trained on, convert mole fraction -> log.
    #   t_oh_exp, oh_exp_ppm  = ...        # digitized OH curve
    #   t_h2o_exp, h2o_exp_ppm = ...       # digitized H2O curve
    #   logOH_obs  = np.log(np.interp(oh_times,  t_oh_exp,  oh_exp_ppm*1e-6))
    #   logH2O_obs = np.log(np.interp(h2o_times, t_h2o_exp, h2o_exp_ppm*1e-6))
    raise NotImplementedError('Fill in the REAL DATA HOOK, then set SYNTHETIC=False.')


# ── Stacked residual ──────────────────────────────────────────────────────────

def make_residual(use_oh, use_h2o, use_prior):
    def residual(x):
        parts = []
        if use_oh:
            parts.append((predict_log(oh_net,  x) - logOH_obs)  / SIG_LOG)
        if use_h2o:
            parts.append((predict_log(h2o_net, x) - logH2O_obs) / SIG_LOG)
        if use_prior:
            parts.append(x / SIGMA_PRIOR_X)
        return np.concatenate(parts)
    return residual


def solve(use_oh, use_h2o, label):
    res = make_residual(use_oh, use_h2o, USE_PRIOR)
    sol = least_squares(res, x0=np.zeros(INPUT_DIM), method='trf')
    cov = np.linalg.inv(sol.jac.T @ sol.jac)      # residuals whitened -> this is Sigma*
    sig = np.sqrt(np.diag(cov))
    print(f'\n{"="*68}\n  {label}\n{"="*68}')
    print(f'  cost = {0.5*np.sum(sol.fun**2):.4g}   (data pts: '
          f'{(len(oh_times) if use_oh else 0)+(len(h2o_times) if use_h2o else 0)})')
    print(f'  {"param":<18}{"x*":>9}{"post_sig_x":>12}{"reduction":>11}')
    for n, xv, sv, sp in zip(PARAM_NAMES, sol.x, sig, SIGMA_PRIOR_X):
        red = 1 - sv / sp if USE_PRIOR else float('nan')
        print(f'  {n:<18}{xv:>9.3f}{sv:>12.3f}{red*100:>10.1f}%')
    return sol.x, sig


# ── Run: show that OH is what resolves k2 ─────────────────────────────────────

solve(False, True,  'H2O ONLY   (k2 unconstrained -> ridge)')
solve(True,  False, 'OH ONLY    (constrains k1/k2 ratio)')
x_star, sig_star = solve(True, True, 'JOINT  (stacked OH + H2O)  <-- USE THIS')


# ── Report the joint solution in physical units ───────────────────────────────

print(f'\n{"="*68}\n  Physical parameters at the joint MAP estimate\n{"="*68}')
import cantera as ct
g = ct.Solution('chem_cti_toy_model_og.yaml')
mol = ct.UnitSystem({"length":"cm","mass":"g","time":"s","quantity":"mol",
    "pressure":"dyn / cm^2","energy":"erg","temperature":"K","current":"A",
    "activation-energy":"cal / mol"})
A0_22 = g.reaction(21).rate.low_rate.pre_exponential_factor
Ea0_22 = mol.convert_activation_energy_to(f"{g.reaction(21).rate.low_rate.activation_energy} J/kmol","cal / mol")
A0_26 = g.reaction(25).rate.pre_exponential_factor
Ea0_26 = mol.convert_activation_energy_to(f"{g.reaction(25).rate.activation_energy} J/kmol","cal / mol")

A22  = A0_22 * np.exp(x_star[0]*LN_F);  Ea22 = Ea0_22 + x_star[1]*SIGMA_E
A26  = A0_26 * np.exp(x_star[2]*LN_F);  Ea26 = Ea0_26 + x_star[3]*SIGMA_E
print(f'  k1 (R22): A = {A22:.3e}  (x{np.exp(x_star[0]*LN_F):.3f})   Ea = {Ea22:.0f} cal/mol')
print(f'  k2 (R26): A = {A26:.3e}  (x{np.exp(x_star[2]*LN_F):.3f})   Ea = {Ea26:.0f} cal/mol')
if SYNTHETIC:
    print(f'\n  recovery error |x* - X_TRUE| = {np.abs(x_star - X_TRUE)}')
    print('  (joint should recover X_TRUE within ~the posterior sigma)')
