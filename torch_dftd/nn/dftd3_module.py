import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from ase.units import Bohr
from torch import Tensor
from torch_dftd.functions.dftd3 import d3_autoang, d3_autoev, edisp
from torch_dftd.functions.distance import calc_distances
from torch_dftd.nn.base_dftd_module import BaseDFTDModule

# >>> Bohr
# 0.5291772105638411


class DFTD3Module(BaseDFTDModule):
    """DFTD3Module

    Args:
        params (dict): xc-dependent parameters. alp, s6, rs6, s18, rs18.
        cutoff (float): cutoff distance in angstrom. Default value is 95bohr := 50 angstrom.
        cnthr (float): coordination number cutoff distance in angstrom.
            Default value is 40bohr := 21 angstrom.
        abc (bool): ATM 3-body interaction
        dtype (dtype): internal calculation is done in this precision.
        bidirectional (bool): calculated `edge_index` is bidirectional or not.
        n_chunks (int): number of times to split c6 computation to reduce peak memory.
        do_check (bool): whether to do check on the input interatom distance again (int): number of times to split c6 computation to reduce peak memory.
    """

    def __init__(
        self,
        params: Dict[str, float],
        cutoff: float = 95.0 * 0.5291772105638411,
        cnthr: float = 40.0 * 0.5291772105638411,
        abc: bool = False,
        dtype=torch.float32,
        bidirectional: bool = False,
        cutoff_smoothing: str = "none",
        n_chunks: Optional[int] = None,
        do_check: bool = False,
    ):
        super(DFTD3Module, self).__init__()

        # relative filepath to package folder
        d3_filepath = str(Path(os.path.abspath(__file__)).parent / "params" / "dftd3_params.npz")
        d3_params = np.load(d3_filepath)
        c6ab = torch.tensor(d3_params["c6ab"], dtype=dtype)
        r0ab = torch.tensor(d3_params["r0ab"], dtype=dtype)
        rcov = torch.tensor(d3_params["rcov"], dtype=dtype)
        r2r4 = torch.tensor(d3_params["r2r4"], dtype=dtype)
        # (95, 95, 5, 5, 3) c0, c1, c2 for coordination number dependent c6ab term.
        self.register_buffer("c6ab", c6ab)
        self.register_buffer("r0ab", r0ab)  # atom pair distance (95, 95)
        self.register_buffer("rcov", rcov)  # atom covalent distance (95)
        self.register_buffer("r2r4", r2r4)  # (95,)

        if cnthr > cutoff:
            print(
                f"WARNING: cnthr {cnthr} is larger than cutoff {cutoff}. "
                f"cutoff distance is used for cnthr"
            )
            cnthr = cutoff
        self.params = params
        self.cutoff = cutoff
        self.cnthr = cnthr
        self.abc = abc
        self.dtype = dtype
        self.bidirectional = bidirectional
        self.cutoff_smoothing = cutoff_smoothing
        self.n_chunks = n_chunks
        self.do_check = do_check

    def calc_energy_batch(
        self,
        Z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        cell: Tensor,
        pbc: Tensor,
        shift_pos: Tensor,
        damping: str = "zero",
    ) -> Tensor:
        """Forward computation to calculate atomic wise dispersion energy"""
        shift_pos = pos.new_zeros((edge_index.size()[1], 3, 3)) if shift_pos is None else shift_pos
        pos_bohr = pos / 0.52917726  # angstrom -> bohr
        # if cell is None:
        #     cell_bohr: Tensor = None
        # else:
        cell_bohr = cell / 0.52917726  # angstrom -> bohr
        shift_bohr = shift_pos / 0.52917726  # angstrom -> bohr
        r = calc_distances(pos_bohr, edge_index, cell_bohr, shift_bohr)
        # E_disp (n_graphs,): Energy in eV unit
        c6ab: torch.Tensor = self.c6ab 
        r0ab: torch.Tensor = self.r0ab 
        rcov: torch.Tensor = self.rcov 
        r2r4: torch.Tensor = self.r2r4
        E_disp = 27.21138505 * edisp(
            Z,
            r,
            edge_index,
            c6ab,
            r0ab,
            rcov,
            r2r4,
            params=self.params,
            cutoff=self.cutoff / 0.5291772105638411,
            cnthr=self.cnthr / 0.5291772105638411,
            #batch=None, #batch,
            #batch_edge=None,#batch_edge,
            shift_pos=shift_bohr,
            damping=damping,
            cutoff_smoothing=self.cutoff_smoothing,
            bidirectional=self.bidirectional,
            abc=self.abc,
            pos=pos_bohr,
            cell=cell_bohr,
            n_chunks=self.n_chunks,
            do_check=self.do_check
        )
        return E_disp
