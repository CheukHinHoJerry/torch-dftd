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
device: str = "cuda"
#cutoff: float = 10.0 * Bohr
cutoff: float = 40.0 * Bohr
cnthr: float = 40.0 * Bohr
abc: bool = True
# --- torch dftd3 specific params ---
#dtype: torch.dtype = torch.float64
bidirectional: bool = True
cutoff_smoothing: str = "none"

# # make d3 calculator that we always use
calculator = TorchDFTD3Calculator(device=device, damping=damping, 
                                dtype=torch.float64, xc=xc, cutoff=cutoff,
                                )

# new d3 module
new_d3_module = TorchDFTD3TorchCalculator(
    device=device, damping=damping, 
    dtype=torch.float64, xc=xc, cutoff=cutoff, do_check=True
)

#
from ase.io import read
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
import numpy as np

al_in = read("/home/coder/project/finetune/make_model/peg16_500K_DFT.xyz", ":")[:5]

for at in al_in:
    at.set_cell(np.eye(3)*100)
    at.set_pbc([False, False, False])

al = read("/home/coder/project/finetune/make_model/water_boxes.xyz", ":")[:5] #+ al_in
#al = read("/home/coder/project/torch-dftd/MOF-2_relax_mace_mp0_small.cif", ":")
total_iterations = len(al)

f = open("d3_ats_data_mof_test_cut.txt", "w")
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


# compile to torch .pt
from torch import jit
model_scripted = torch.jit.script(new_d3_module)
model_scripted.save(f"test_d3_{device}_check_{cutoff}_abc=true.pt")
