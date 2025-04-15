from typing import Optional

import torch
from torch import Tensor


def calc_distances(
    pos: Tensor,
    edge_index: Tensor,
    cell: Tensor,
    shift_pos: Tensor,
    eps:float = 1e-20,
) -> Tensor:
    """Distance calculation function.

    Args:
        pos (Tensor): (n_atoms, 3) atom positions.
        edge_index (Tensor): (2, n_edges) edge_index for graph.
        cell (Tensor): cell size, None for non periodic system.
        shift_pos (Tensor): (n_edges, 3) position shift vectors of edges owing to the periodic boundary. It should be length unit.
        eps (float): Small float value to avoid NaN in backward when the distance is 0.

    Returns:
        Dij (Tensor): (n_edges, ) distances of edges

    """
    assert cell is not None
    idx_i = edge_index[0, :]
    idx_j = edge_index[1, :]
    # calculate interatomic distances
    Ri = pos[idx_i]
    Rj = pos[idx_j]
    #if cell is not None:
    Rj += shift_pos
    # eps is to avoid Nan in backward when Dij = 0 with sqrt.
    Dij = torch.sqrt(torch.sum((Ri - Rj) ** 2, dim=-1) + eps)
    return Dij
