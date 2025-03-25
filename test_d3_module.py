import torch
import ase.io

from torch_dftd.nn.dftd3_module import DFTD3Module
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from torch_dftd.torch_dftd3_module import TorchDFTD3TorchCalculator

from torch_dftd.dftd3_xc_params import get_dftd3_default_params
from ase import units
from ase.optimize import LBFGS, FIRE
from ase.units import Bohr

##
# --- parameters ---
dft = "d3"
damping = "bj"
xc: str = "pbe"
old: bool = False
device: str = "cuda:0"
cutoff: float = 40.0 * Bohr
cnthr: float = 40.0 * Bohr
abc: bool = False
# --- torch dftd3 specific params ---
dtype: torch.dtype = torch.float64
bidirectional: bool = True
cutoff_smoothing: str = "none"

# make dftd3 nn.module
# params = get_dftd3_default_params(damping, xc, old=old)
# d3_module = DFTD3Module(params, 
#                       cutoff=cutoff, 
#                       cnthr=cnthr, 
#                       abc=abc,
#                       dtype=dtype, 
#                       bidirectional=bidirectional, 
#                       cutoff_smoothing=cutoff_smoothing
#                       )
# d3_module.to(device)

# # make d3 calculator that we always use
# original_d3_cal = TorchDFTD3Calculator(device="cpu", damping=damping, 
#                                        dtype=torch.float32, xc="pbe", cutoff=cutoff)

# new d3 module
new_d3_module = TorchDFTD3TorchCalculator(
    device="cuda:0", damping=damping, 
    dtype=torch.float32, xc="pbe", cutoff=cutoff
)

#
from ase.io import read
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
import numpy as np

fin = "/home/coder/project/plot_dispersion/data/pure_solvent_100_1-5_rho=1.1.xyz"
config = ase.io.read(fin, index="-1")
config.set_pbc(True)
al = [config,]

Zs = np.unique(al[0].get_atomic_numbers())
configs = [data.config_from_atoms(atoms) for atoms in al]
z_table = utils.AtomicNumberTable(list(Zs))

data_loader = torch_geometric.dataloader.DataLoader(
    dataset=[
        data.AtomicData.from_config(
            config, z_table=z_table, cutoff=cutoff
        )
        for config in configs
    ],
    batch_size=1,
    shuffle=False,
    drop_last=False,
)

datas = []
for dd in data_loader:
    datas.append(dd.to_dict())

datas[0]
try_in = datas[0]
pos = try_in['positions'].to(device)
Z = torch.tensor(config.get_atomic_numbers(), device=device)
pbc = torch.tensor(config.pbc, device = device)
edge_index = try_in['edge_index']
S = try_in['unit_shifts']
damping = damping

if any(pbc):
    cell = try_in['cell'].to(device)
else:
    cell = None

if cell is None:
    shift_pos = S
else:
    shift_pos = torch.mm(S.to(device), cell.detach())



data_1 = {
    'positions': pos.to(device),
    'Z' : Z.to(device),
    'cell': cell.to(device),
    'pbc': torch.tensor(pbc).to(device),
    'edge_index': edge_index.to(device),
    'unit_shifts': S.to(device),
}

e4 = new_d3_module(data_1)["energy"]
# 
#assert e1 == e2 == e3 == e4


# -3.8232408211206472
# -3.8232408211206472
# -3.8232407986300667 # ase nlist from torchdftd3
# -3.8232408211206472

# # pymatgen nlist from torch
# -3.8232408211206472
# -3.8232408211206472
# -3.8232411456106354
# -3.8232408211206472

# compile as .pt
from torch import jit
model_scripted = torch.jit.script(new_d3_module)
model_scripted.save("test_d3.pt")

# load and try evaluate
loaded_model = torch.jit.load("test_d3.pt")
output = loaded_model(data_1)

al_james = read("/home/coder/project/finetune/make_model/combined.xyz", ":")
al = al_water + al_in + al_james
total_iterations = len(al)

f = open("ats_data_noncubic.txt", "w")
f.write("{}\n".format(total_iterations))
for k in range(total_iterations):
    at = al[k]
    cellT = np.transpose(at.cell)
    at.wrap()
    at.set_calculator(calculator)    
    f.write("{} {} {} {}\n".format(len(at), at.get_potential_energy(), at.get_forces()[0][0], at.get_stress()[0]))
    for j in range(3):
        f.write("{} {} {} {}\n".format(cellT[j][0], cellT[j][1], cellT[j][2], int(at.pbc[j])))
    f.write(" ".join([str(i) for i in at.get_atomic_numbers()]) + "\n")
    for i in range(len(at)):
        f.write(" ".join([str(j) for j in at[i].position]) + "\n")