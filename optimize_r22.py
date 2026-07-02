"""
optimize_r22.py — Standalone Zhang Eq. 8 optimization + Laplace posterior

Two ways to use:
  1. From a notebook/script after training:
        from optimize_r22 import save_training_meta, run_optimization
        save_training_meta(Y_MEAN, Y_STD, T_SCALE)        # call once after training
        result, x_opt, Sigma = run_optimization(model, Y_MEAN, Y_STD, T_SCALE)

  2. From the command line (loads model + meta saved by save_training_meta):
        python optimize_r22.py
        python optimize_r22.py --model best_nn_surrogate_r22_relu_test.pt
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import cantera as ct
from scipy.optimize import least_squares

# ── Default file paths ────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = 'best_nn_surrogate_r22_relu_test.pt.bak'

# ── NN architecture (must match training) ─────────────────────────────────────
_NN_INPUT_DIM  = 3   # INPUT_DIM(2) + 1 time
_HIDDEN_DIM    = 16
_NN_OUTPUT_DIM = 1


class SurrogateNN(nn.Module):
    def __init__(self, input_dim=_NN_INPUT_DIM, hidden_dim=_HIDDEN_DIM, output_dim=_NN_OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, t):
        B = x.size(0)
        if t.dim() == 1:
            T_STEPS = t.size(0)
            t_exp = t.view(1, T_STEPS, 1).expand(B, -1, -1)
        else:
            T_STEPS = t.size(1)
            t_exp = t.unsqueeze(-1)
        x_exp = x.unsqueeze(1).expand(-1, T_STEPS, -1)
        xt = torch.cat([x_exp, t_exp], dim=-1)
        return self.net(xt).squeeze(-1)


# ── Physics / simulation constants ────────────────────────────────────────────
YAML_FILE   = 'chem_cti_toy_model_og.yaml'
IDX_R22     = 21
INPUT_DIM   = 2
PARAM_NAMES = ['lnA_R22', 'Ea_R22']
LN_F        = 10.0     # 2σ prior on ln(A)
SIGMA_E     = 5000.0   # 2σ prior on Ea [cal/mol]
LAMBDA      = 4.0      # regularization weight
NOISE_FLOOR = 1e-6     # clip floor before log
SIGMA_OB_LN = 0.05     # observational uncertainty in log-space (Burke 2013 Table 2)

T_INITIAL   = 1057
P_INITIAL   = 1.83 * ct.one_atm
INITIAL_X   = {'H2O2': 860e-6, 'H2O': 663e-6, 'O2': 332e-6,
                'AR':   1.0 - (860 + 663 + 332) * 1e-6}
DT_MAX      = 1e-6
TIME_STEPS  = 6000
T_SIM       = np.linspace(DT_MAX, DT_MAX * TIME_STEPS, TIME_STEPS)

EXP_CSV     = 'Hong et.al_og.csv'


# ── Metadata helpers ──────────────────────────────────────────────────────────

def save_training_meta(y_mean, y_std, t_scale, model_path=DEFAULT_MODEL_PATH):
    """Save z-score stats alongside the model. Call once after training."""
    meta_path = model_path.replace('.pt', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({'y_mean': float(y_mean), 'y_std': float(y_std),
                   't_scale': float(t_scale)}, f, indent=2)
    print(f'Saved training meta → {meta_path}')


def load_training_meta(model_path=DEFAULT_MODEL_PATH):
    meta_path = model_path.replace('.pt', '_meta.json')
    with open(meta_path) as f:
        return json.load(f)


# ── Nominal kinetic parameters (read from Cantera, not hardcoded) ─────────────

def _load_nominal_params():
    mol_units = ct.UnitSystem({
        "length": "cm", "mass": "g", "time": "s",
        "quantity": "mol", "pressure": "dyn / cm^2", "energy": "erg",
        "temperature": "K", "current": "A", "activation-energy": "cal / mol"})
    gas     = ct.Solution(YAML_FILE)
    A       = gas.reaction(IDX_R22).rate.low_rate.pre_exponential_factor
    B       = gas.reaction(IDX_R22).rate.low_rate.temperature_exponent
    Ea_si   = gas.reaction(IDX_R22).rate.low_rate.activation_energy
    Ea_cal  = mol_units.convert_activation_energy_to(f"{Ea_si} J/kmol", "cal / mol")
    return A, B, Ea_cal


# ── Main entry point ──────────────────────────────────────────────────────────

def run_optimization(model, t_scale=None, out_prefix='opt_r22', verbose=True):
    """
    Run Zhang Eq. 8 TRF optimization + Laplace posterior uncertainty.
    """
    model.eval()
    t_scale = T_SIM[-1] if t_scale is None else t_scale

    NOMINAL_A, NOMINAL_B, NOMINAL_EA_cal = _load_nominal_params()

    # ── Experimental data ─────────────────────────────────────────────────────
    df_exp = pd.read_csv(EXP_CSV)
    t_exp  = df_exp['time'].values * 1e-3                        # ms → s
    y_exp  = (df_exp['x_h2o'].values * 0.75 + 400) * 1e-6       # ppm → mole fraction
    if verbose:
        print(f'Exp points: {len(t_exp)}')

    # ── Core functions (close over model, t_scale) ────────────

    def nn_predict(x_vec, t_query=None):
        x_t = torch.tensor(np.asarray(x_vec).reshape(1, -1), dtype=torch.float32)
        t_t = torch.tensor((T_SIM if t_query is None else t_query) / t_scale, dtype=torch.float32)
        with torch.no_grad():
            return model(x_t, t_t).numpy().ravel()

    def compute_jacobian(x_vec, t_query):
        x_t = torch.tensor(np.asarray(x_vec).reshape(1, -1),
                        dtype=torch.float32, requires_grad=True)
        t_t = torch.tensor(t_query / t_scale, dtype=torch.float32)

        def func(xx):
            return model(xx, t_t).squeeze(0)

        J = torch.autograd.functional.jacobian(func, x_t)
        return J.detach().numpy().reshape(len(t_query), INPUT_DIM)  

    def objective(x_vec):
        """Residual vector r; least_squares minimizes 0.5*||r||^2 (Zhang Eq. 8)."""
        y_pred_log = nn_predict(x_vec, t_query=t_exp)
        ln_y_exp = np.log(np.clip(y_exp, NOISE_FLOOR, None))
        data_res = (y_pred_log - ln_y_exp) / SIGMA_OB_LN
        reg_res  = np.sqrt(LAMBDA) * np.asarray(x_vec)
        return np.concatenate([data_res, reg_res])

    def objective_jac(x_vec):
        J_log = compute_jacobian(x_vec, t_exp)
        J_data = J_log / SIGMA_OB_LN
        J_reg = np.sqrt(LAMBDA) * np.eye(INPUT_DIM)
        return np.vstack([J_data, J_reg])


    # ── TRF bounded least squares ─────────────────────────────────────────────
    result = least_squares(
        objective, np.zeros(INPUT_DIM),
        jac=objective_jac,
        bounds=([-1.0] * INPUT_DIM, [1.0] * INPUT_DIM),
        method='trf')
    x_opt = result.x

    if verbose:
        print(f'\nConverged: {result.success}  (status {result.status}: {result.message})')
        for name, xv in zip(PARAM_NAMES, x_opt):
            at_bound = '   <-- AT BOUND' if abs(abs(xv) - 1.0) < 1e-4 else ''
            print(f'  {name:12s}: x* = {xv:+.8f}{at_bound}')
        print(f'Final cost 0.5||r||^2 : {result.cost:.4e}')

    # ── Laplace posterior (Gauss-Newton Hessian) ──────────────────────────────
    J_res      = result.jac                    # (N_exp + INPUT_DIM, INPUT_DIM)
    H_gn       = J_res.T @ J_res
    Sigma_star = np.linalg.inv(H_gn)
    L_chol     = np.linalg.cholesky(Sigma_star)

    J_sim_log = compute_jacobian(x_opt, T_SIM)
    JL = J_sim_log @ L_chol
    pred_var_ln = np.sum(JL**2, axis=1)
    pred_std    = np.sqrt(pred_var_ln)

    # ── Cantera ground-truth runs ─────────────────────────────────────────────
    def _cantera_run(x_vec):
        gas = ct.Solution(YAML_FILE)
        new_A    = NOMINAL_A * np.exp(x_vec[0] * LN_F)
        new_Ea_J = (NOMINAL_EA_cal + x_vec[1] * SIGMA_E) * 4184.0
        rxn = gas.reaction(IDX_R22)
        rxn.rate.low_rate = ct.Arrhenius(new_A, NOMINAL_B, new_Ea_J)
        gas.modify_reaction(IDX_R22, rxn)
        gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
        net = ct.ReactorNet([ct.IdealGasReactor(gas, energy='on')])
        h2o_idx = gas.species_index('H2O')
        profile = np.empty(TIME_STEPS)
        for i in range(TIME_STEPS):
            net.advance(net.time + DT_MAX)
            profile[i] = net.reactors[0].thermo.X[h2o_idx]
        return profile

    y_opt_ct = _cantera_run(x_opt)
    y_nom_ct = _cantera_run(np.zeros(INPUT_DIM))
    y_opt_nn = np.exp(nn_predict(x_opt))

    # ── Main result plot ──────────────────────────────────────────────────────
    t_ms     = T_SIM * 1e3
    t_exp_ms = t_exp * 1e3
    y_upper  = y_opt_ct * np.exp( 2 * pred_std) * 1e6
    y_lower  = y_opt_ct * np.exp(-2 * pred_std) * 1e6

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_exp_ms, y_exp * 1e6, 'o', mfc='none', mec='k',
            label='Hong et al. (exp)', zorder=5)
    ax.plot(t_ms, y_nom_ct * 1e6, 'r--', lw=4,  label='Cantera, nominal')
    ax.plot(t_ms, y_opt_ct * 1e6, 'b-',  lw=2,  label='Cantera, optimized')
    ax.plot(t_ms, y_opt_nn * 1e6, 'g:',  lw=1.5, label='NN-RS at x*')
    ax.fill_between(t_ms, y_lower, y_upper, color='b', alpha=0.18,
                    label='NN-RS posterior 2σ')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'H$_2$O mole fraction [ppm]')
    ax.set_xlim([0, 6])
    ax.legend(loc='lower right', frameon=False)
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_title(f'MSI-NN: {INPUT_DIM}-parameter optimization (R22)')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_result.png', dpi=150)
    if verbose:
        print(f'Saved: {out_prefix}_result.png')

    # ── Correlation matrix plot ───────────────────────────────────────────────
    diag_std = np.sqrt(np.diag(Sigma_star))
    Corr = Sigma_star / np.outer(diag_std, diag_std)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(Corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(INPUT_DIM))
    ax.set_xticklabels(PARAM_NAMES, rotation=30, ha='right')
    ax.set_yticks(range(INPUT_DIM))
    ax.set_yticklabels(PARAM_NAMES)
    for i in range(INPUT_DIM):
        for j in range(INPUT_DIM):
            ax.text(j, i, f'{Corr[i, j]:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if abs(Corr[i, j]) > 0.6 else 'black')
    plt.colorbar(im, ax=ax, label='correlation')
    ax.set_title('Posterior correlation matrix')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_corr.png', dpi=150)
    if verbose:
        print(f'Saved: {out_prefix}_corr.png')

    # ── Physical-units summary ────────────────────────────────────────────────
    phys_scales    = np.array([LN_F, SIGMA_E])
    post_std_phys  = np.sqrt(np.diag(Sigma_star)) * phys_scales
    prior_std_phys = 0.5 * phys_scales              # prior std = 0.5 in normalized space

    A_opt  = NOMINAL_A * np.exp(x_opt[0] * LN_F)
    Ea_opt = NOMINAL_EA_cal + x_opt[1] * SIGMA_E

    if verbose:
        print('\n' + '=' * 70)
        print(f'  A  R22 : nom = {NOMINAL_A:.3e}   opt = {A_opt:.3e}   x* = {x_opt[0]:+.4f}')
        print(f'           2σ : ×÷{np.exp(2*post_std_phys[0]):.2f}   '
              f'(prior ×÷{np.exp(2*prior_std_phys[0]):.2f})')
        print(f'  Ea R22 : nom = {NOMINAL_EA_cal:.0f} cal/mol   '
              f'opt = {Ea_opt:.0f} cal/mol   x* = {x_opt[1]:+.4f}')
        print(f'           2σ : ±{2*post_std_phys[1]:.0f} cal/mol   '
              f'(prior ±{2*prior_std_phys[1]:.0f} cal/mol)')
        print('=' * 70)

    return result, x_opt, Sigma_star


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    model_path = DEFAULT_MODEL_PATH
    out_prefix = 'opt_r22'

    model = SurrogateNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    print(f'Loaded model: {model_path}')

    run_optimization(model,
                     t_scale=T_SIM[-1],
                     out_prefix=out_prefix)
