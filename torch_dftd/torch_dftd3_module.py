import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes
from ase.units import Bohr
from torch import Tensor
from torch_dftd.dftd3_xc_params import get_dftd3_default_params
from torch_dftd.functions.edge_extraction import calc_edge_index
from torch_dftd.nn.dftd2_module import DFTD2Module
from torch_dftd.nn.dftd3_module import DFTD3Module

import torch.nn as nn

class TorchDFTD3TorchCalculator(nn.Module):
    """ase compatible DFTD3 calculator using pytorch

    Args:
        dft (Calculator or None): base dft calculator can be set here
        atoms (Atoms):
        damping (str): damping method. "zero", "bj", "zerom", "bjm"
        xc (str): exchange correlation functional
        old (bool): Use DFTD2 method when `True`, DFTD3 method is used when `False`
        device (str): torch device. Ex. "cuda:0" to use GPU ID 0
        cutoff (float): cutoff distance in angstrom. Default value is 95bohr := 50 angstrom.
        cnthr (float): coordination number cutoff distance in angstrom.
            Default value is 40bohr := 21 angstrom.
        abc (bool): ATM 3-body interaction
        dtype (dtype): internal calculation is done in this precision.
        bidirectional (bool): calculated `edge_index` is bidirectional or not.
        cutoff_smoothing (str): cutoff smoothing makes gradient smooth at `cutoff` distance
        **kwargs:
    """

    def __init__(
        self,
        dft: Optional[Calculator] = None,
        atoms: Atoms = None,
        damping: str = "zero",
        xc: str = "pbe",
        old: bool = False,
        device: str = "cpu",
        cutoff: float = 95.0 * Bohr,
        cnthr: float = 40.0 * Bohr,
        abc: bool = False,
        # --- torch dftd3 specific params ---
        dtype: torch.dtype = torch.float32,
        bidirectional: bool = True,
        cutoff_smoothing: str = "none",
        # --- neighborlist specific params ---
        every: int = -1,  # time step that consider rebuild
        delay: int = -1,  # delay build neighbor list until this time step since last rebuilt
        check: bool = False,  # If true, rebuild if min(ats.positions - cached_position) > skin / 2, else must rebuild after "delay"
        skin: float = None,  # skin parameter for checking rebuild, in angstrom
        **kwargs,
    ):
        super(TorchDFTD3TorchCalculator, self).__init__()
        self.dft = dft
        self.params = get_dftd3_default_params(damping, xc, old=old)
        self.damping = damping
        self.abc = abc
        self.old = old
        self.device = torch.device(device)
        # if old:
        #     self.dftd_module: torch.nn.Module = DFTD2Module(
        #         self.params,
        #         cutoff=cutoff,
        #         dtype=dtype,
        #         bidirectional=bidirectional,
        #         cutoff_smoothing=cutoff_smoothing,
        #     )
        # else:
        self.dftd_module = DFTD3Module(
            self.params,
            cutoff=cutoff,
            cnthr=cnthr,
            abc=abc,
            dtype=dtype,
            bidirectional=bidirectional,
            cutoff_smoothing=cutoff_smoothing,
        )
        self.dftd_module.to(device)
        self.dtype = dtype
        self.cutoff = cutoff
        self.bidirectional = bidirectional
        # 
        # compatibility with ase calculator
        #self.results: Dict[str, torch.Tensor] = {}
        # --- skin nlist ---
        self.Nrebuilds = 0  # record number of rebuilding nlist
        if every != -1 and delay != -1 and skin != None:
            self.use_skin = True
            #
            self.every = every
            self.delay = delay
            self.check = check
            if check and skin == None:
                self.skin = 0.2
            else:
                self.skin = skin
            # enlarge the nlist by skin
            self.cutoff += self.skin
            # counter and caching tools
            self.count_rebuild = 0  # count steps since last rebuild
            self.count_check = 0  # count steps since last checking
            self.cache_input_dicts = dict()
            self.rebuild = True
        else:
            self.use_skin = False

    # ref: https://github.com/ACEsuit/mace/blob/main/mace/calculators/lammps_mace.py
    def forward(
        self,
        input_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # 
        # wrap the `data` as torch-dftd equivalent format
        # in torch_dftd3_calculator.py
        # data is from libSG
        # old_pos = copy.deepcopy(data['positions']),
        pos=input_data['positions'].to(self.device)
        Z = input_data['Z'].to(self.device)
        cell = input_data['cell'].to(self.device)
        pbc = input_data['pbc'].to(self.device)
        edge_index = input_data['edge_index'].to(self.device)
        S = input_data["unit_shifts"].to(self.device)

        # transform S here
        # if any(pbc):
        #     cell = cell.to(self.device)
        # else:
        #     cell = None

        # if cell is None:
        #     shift_pos = S
        # else:
        # assume cell is always given
        cell = cell.to(self.device)
        shift_pos = torch.mm(S, cell.detach())

        results = self.dftd_module.calc_energy_and_forces(
        #results = self.dftd_module.calc_energy_batch(
                pos = pos,
                Z = Z,
                cell = cell,
                pbc = pbc,
                edge_index=edge_index,
                shift_pos=shift_pos,
                damping=self.damping,
            )[0]
        #self.results["energy"] = results#["energy"]
        #self.results["free_energy"] = self.results["energy"]

        # skip
        # Referenced DFTD3 impl.
        # if self.dft is not None:
        #     try:
        #         efree = self.dft.get_potential_energy(force_consistent=True)
        #         self.results["free_energy"] += efree
        #     except PropertyNotImplementedError:
        #         pass

        #if "forces" in results:
        #self.results["forces"] = results["forces"]
        #if "stress" in results:
        #self.results["stress"] = results["stress"]
        
        return {
            "energy": results["energy"],
            "forces": results["forces"],
            "stress": results["stress"],
        }