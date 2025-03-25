import torch
import ase.io

from torch_dftd.nn.dftd3_module import DFTD3Module
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from torch_dftd.torch_dftd3_module import TorchDFTD3TorchCalculator

from torch_dftd.dftd3_xc_params import get_dftd3_default_params
from ase import units
from ase.units import Bohr

from ase.io import read
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
import numpy as np
import timeit

from cProfile import Profile
from pstats import SortKey, Stats
##
# --- parameters ---
dft = "d3"
damping = "bj"
xc: str = "pbe"
old: bool = False
device: str = "cpu"
cutoff: float = 40.0 * Bohr
cnthr: float = 40.0 * Bohr
abc: bool = False
# --- torch dftd3 specific params ---
#dtype: torch.dtype = torch.float64
bidirectional: bool = True
cutoff_smoothing: str = "none"

# --- benchmark parameters ---
nwarmup = 5
nsample = 5
profile_code = False

# --- example data ---
al = read("/home/coder/project/torch-dftd/ethanol900.xyz")

##

# make d3 calculator that we always use
calculator = TorchDFTD3Calculator(device=device, damping=damping, 
                                dtype=torch.float64, xc=xc, cutoff=cutoff,
                                )

# new d3 module - do check
new_d3_module = TorchDFTD3TorchCalculator(
    device=device, damping=damping, 
    dtype=torch.float64, xc=xc, cutoff=cutoff, do_check=True
)

new_d3_module2 = TorchDFTD3TorchCalculator(
    device=device, damping=damping, 
    dtype=torch.float64, xc=xc, cutoff=cutoff, do_check=False
)


# get input dictionary
input_dict = calculator._build_nlist(atoms=al)
input_dict["positions"] = input_dict['pos']
input_dict["unit_shifts"] = input_dict['shift_pos']
print(new_d3_module(input_dict))


# warm up
for _ in range(nwarmup):
    new_d3_module(input_dict)

for _ in range(nwarmup):
    new_d3_module2(input_dict)

# benchmark functions
def test_docheck():
    for _ in range(nsample):
        new_d3_module(input_dict)

def test_docheck2():
    for _ in range(nsample):
        new_d3_module2(input_dict)

if profile_code:
    with Profile() as profile:
        print(f"{test_docheck() = }")
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.CUMULATIVE)
            .print_stats()
        )

t1 = timeit.timeit("test_docheck()", globals=globals(), number=1)
print("time for doing check: ", t1)

if profile_code:
    with Profile() as profile:
        print(f"{test_docheck2() = }")
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.CUMULATIVE)
            .print_stats()
        )

t2 = timeit.timeit("test_docheck2()", globals=globals(), number=1)
print("time for doing check2: ", t2)
