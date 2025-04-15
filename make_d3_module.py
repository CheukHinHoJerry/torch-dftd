import torch

from torch_dftd.nn.dftd3_module import DFTD3Module
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from torch_dftd.torch_dftd3_module import TorchDFTD3TorchCalculator
from torch_dftd.dftd3_xc_params import get_dftd3_default_params

from ase.units import Bohr

##
# --- parameters ---
dft = "d3"
damping = "bj"
xc: str = "pbe"
old: bool = False # default
device: str = "cpu" #"cuda:0"
cutoff: float = 40.0 * Bohr
cnthr: float = 40.0 * Bohr # default
abc: bool = True
do_check: bool = True
# --- torch dftd3 specific params ---
#dtype: torch.dtype = torch.float64
#bidirectional: bool = True
#cutoff_smoothing: str = "none"

## ===

# new d3 module
new_d3_module = TorchDFTD3TorchCalculator(
    device=device, damping=damping, 
    dtype=torch.float64, xc=xc, cutoff=cutoff,
    do_check=True
)

#

model_scripted = torch.jit.script(new_d3_module)
model_scripted.save("test_d3_test.pt")