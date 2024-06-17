
"""
first testings for the nlist with skin layet
TODO: repeated the tests from mace + d3 by using eam/other empirical potential (instead of mace) and check
with the convergence of 3 different cases:
(i) reference (ii) usual skin nlist (iii) one that does not update
with deterministic md wrapped in a tests with certain tolerance
"""

from typing import List

import numpy as np
import pytest
import torch
import copy

from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.calculators.dftd3 import DFTD3
from ase.calculators.emt import EMT
from ase import units
import ase.md

from torch_dftd.testing.damping import damping_method_list, damping_xc_combination_list
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

# ---- testing utilities

def ref_atoms():
    """Initialization"""
    mol = molecule("CH3CH2OCH3")

    slab = fcc111("Au", size=(2, 1, 3), vacuum=80.0)
    slab.set_cell(
        slab.get_cell().array @ np.array([[1.0, 0.1, 0.2], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]])
    )
    slab.pbc = np.array([True, True, True])

    large_bulk = bulk("Pt", "fcc") * (4, 4, 4)

    atoms_dict = {"mol": mol, "slab": slab, "large": large_bulk}

    # TODO: also test on other Atoms
    return atoms_dict["mol"]

##
    
def test_skin_counters():
    """
    Test different counters involved
    """
    # ---- dftd3 correction parameters
    count_test = 123
    every = 2
    delay = 10
    check = False
    ##
    atoms = ref_atoms()
    damping = "bj"
    old = False
    device = "cpu"
    xc = "pbe"
    torch_dftd3_calc = TorchDFTD3Calculator(
            device="cuda", damping="bj", dtype=torch.float32, xc="pbe", cutoff= 40 * units.Bohr, 
            every = every, delay = delay, check = check, skin = 2.0 * units.Bohr,
            )
    atoms.set_calculator(torch_dftd3_calc)
    
    dyn = ase.md.verlet.VelocityVerlet(atoms, 0.5 * units.fs)
    dyn.run(count_test)
    
    Nrebuilds_prev = torch_dftd3_calc.Nrebuilds
    torch_dftd3_calc.reset_counter()
    Nrebuilds_new = torch_dftd3_calc.Nrebuilds     
    print(Nrebuilds_prev == count_test // delay)   
    assert Nrebuilds_prev == count_test // delay and \
            (Nrebuilds_new == 0 and torch_dftd3_calc.count_rebuild == 0 and torch_dftd3_calc.count_check == 0)


    