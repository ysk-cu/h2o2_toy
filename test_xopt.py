import numpy as np
import torch
import cantera as ct

# Load the model
ckpt = torch.load('result_1192k_oh_gelu_train.pt', weights_only=False)
import torch.nn as nn
class SurrogateNN(nn.Module):
    def __init__(self, hidden=16, n_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_out),
        )
    def forward(self, x):
        return self.net(x)

model = SurrogateNN()
model.load_state_dict(ckpt['model_state'])
model.eval()

x_opt = [-0.079, 0.025, 0.027, -0.019]
x_t = torch.tensor(np.asarray(x_opt).reshape(1, -1), dtype=torch.float32)
with torch.no_grad():
    y_nn_log = model(x_t).squeeze(0).numpy()
print("NN raw log(OH):", y_nn_log)
print("NN OH ppm:", np.exp(y_nn_log) * 1e6)

# Cantera
YAML_FILE = 'chem_cti_toy_model_og.yaml'
gas = ct.Solution(YAML_FILE)
IDX_R22 = 21
IDX_R26 = 25
LN_F = 10
SIGMA_E = 5000.0

NOMINAL_A_R22 = gas.reaction(IDX_R22).rate.low_rate.pre_exponential_factor
NOMINAL_B_R22 = gas.reaction(IDX_R22).rate.low_rate.temperature_exponent
NOMINAL_EA_R22_si = gas.reaction(IDX_R22).rate.low_rate.activation_energy
NOMINAL_EA_R22_cal = NOMINAL_EA_R22_si / 4184.0 * 1000  # J/kmol -> cal/mol

NOMINAL_A_R26 = gas.reaction(IDX_R26).rate.pre_exponential_factor
NOMINAL_B_R26 = gas.reaction(IDX_R26).rate.temperature_exponent
NOMINAL_EA_R26_si = gas.reaction(IDX_R26).rate.activation_energy
NOMINAL_EA_R26_cal = NOMINAL_EA_R26_si / 4184.0 * 1000

new_A_R22  = NOMINAL_A_R22 * np.exp(x_opt[0] * LN_F)
new_Ea_R22 = (NOMINAL_EA_R22_cal + x_opt[1] * SIGMA_E) * 4184.0
rxn22 = gas.reaction(IDX_R22)
rxn22.rate.low_rate = ct.Arrhenius(new_A_R22, NOMINAL_B_R22, new_Ea_R22)
gas.modify_reaction(IDX_R22, rxn22)

new_A_R26  = NOMINAL_A_R26 * np.exp(x_opt[2] * LN_F)
new_Ea_R26 = (NOMINAL_EA_R26_cal + x_opt[3] * SIGMA_E) * 4184.0
rxn26 = gas.reaction(IDX_R26)
rxn26.rate = ct.Arrhenius(new_A_R26, NOMINAL_B_R26, new_Ea_R26)
gas.modify_reaction(IDX_R26, rxn26)

gas.TPX = 1192, 1.95 * ct.one_atm, {'H2O2': 2216e-6, 'H2O': 1364e-6, 'O2': 682e-6, 'AR': 1.0 - (2216 + 1364 + 682) * 1e-6}
reactor = ct.IdealGasConstPressureReactor(gas, energy='on')
net = ct.ReactorNet([reactor])
oh_idx = gas.species_index('OH')
target_times = ckpt['target_times']

y_ct_ppm = []
for t in target_times:
    net.advance(t)
    y_ct_ppm.append(reactor.thermo.X[oh_idx] * 1e6)
print("Cantera OH ppm:", y_ct_ppm)
