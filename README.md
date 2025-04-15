# torch-dftd
pytorch implementation of dftd2 [1] & dftd3 [2, 3]

 - Blog: [Open sourcing pytorch implementation of DFTD3](https://tech.preferred.jp/en/blog/oss-pytorch-dftd3/)

## Install

```bash
# Install from pypi
pip install torch-dftd

# Install from source (for developers)
git clone https://github.com/pfnet-research/torch-dftd
pip install -e .
```

## Quick start

```python
from ase.build import molecule
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

atoms = molecule("CH3CH2OCH3")
# device="cuda:0" for fast GPU computation.
calc = TorchDFTD3Calculator(atoms=atoms, device="cpu", damping="bj")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(f"energy {energy} eV")
print(f"forces {forces}")
```

## [Experimental] Using skin-type neighborlist
Recomputing the neighbor list at every step can be slow, particularly when the cutoff for the D3 correction is large. To address this, a naive implementation of a neighbor list with a skin layer is provided, featuring hyperparameters similar to those in [LAMMPS](https://docs.lammps.org/neigh_modify.html#description). You can simply replace your calculator initialization with the following:

```python
calc = TorchDFTD3Calculator(atoms=atoms, device="cpu", damping="bj",
 every = 2, delay = 10, check = True, skin = 2.0) # skin-type nlist parameters
```
Here, the additional parameters `every`, `delay`, `check`, and `skin` allow reusing the neighbor list as specified in the [LAMMPS](https://docs.lammps.org/neigh_modify.html#description) documentation. The `skin` parameter defines the buffer region around the cutoff distance and has the same unit as specified in the `atoms` object, which is typically in angstroms.

This part of the code is released under [Academic Software Licsense (ASL)](https://github.com/ACEsuit/ACE.jl/blob/main/ASL.md) .

## Compile as torch.pt file
To use the d3 calculator in C++, an easiest way is to compile the d3 `nn.Module` as a `.pt` file. This branch is specifially for such purpose that fixes compilation issue that arises with the main branch. To create a `.pt` file, modify the parameters in `torch-dftd/make_d3_module.py` and run

```
python make_d3_module.py
```

a `test_d3.pt` file will then be generated. At this point, you may need to compile the correpsonding `.pt` for each `device` but this will be fixed in the near future.


## Dependency

The library is tested under following environment.
 - python: 3.10
 - CUDA: 12.2
```bash
torch==2.0.1
ase==3.22.1
# Below is only for 3-body term
cupy-cuda12x==12.2.0
pytorch-pfn-extras==0.7.3
```

## Development tips
### Formatting & Linting
[pysen](https://github.com/pfnet/pysen) is used to format the python code of this repository.<br/>
You can simply run below to get your code formatted :)
```bash
# Format the code
$ pysen run format
# Check the code format
$ pysen run lint
```

### CUDA Kernel function implementation with cupy
[cupy](https://github.com/cupy/cupy) supports users to implement CUDA kernels within python code, 
and it can be easily linked with pytorch tensor calculations.<br/>
Element wise kernel is implemented and used in some pytorch functions to accelerate speed with GPU.

See [document](https://docs.cupy.dev/en/stable/user_guide/kernel.html) for details about user defined kernel.

## Citation

Please always cite original paper of DFT-D2 [1] or DFT-D3 [2, 3].<br/>
Also, please cite the paper [4] if you used this software for your publication.

DFT-D2:<br/>
[1] S. Grimme, J. Comput. Chem, 27 (2006), 1787-1799.
DOI: [10.1002/jcc.20495](https://doi.org/10.1002/jcc.20495)

DFT-D3:<br/>
[2] S. Grimme, J. Antony, S. Ehrlich and H. Krieg, J. Chem. Phys, 132 (2010), 154104.
DOI: [10.1063/1.3382344](https://doi.org/10.1063/1.3382344)

If BJ-damping is used in DFT-D3:<br/> 
[3] S. Grimme, S. Ehrlich and L. Goerigk, J. Comput. Chem, 32 (2011), 1456-1465.
DOI: [10.1002/jcc.21759](https://doi.org/10.1002/jcc.21759)

[4] [PFP: Universal Neural Network Potential for Material Discovery](https://arxiv.org/abs/2106.14583)

```text
@misc{takamoto2021pfp,
      title={PFP: Universal Neural Network Potential for Material Discovery}, 
      author={So Takamoto and Chikashi Shinagawa and Daisuke Motoki and Kosuke Nakago and Wenwen Li and Iori Kurata and Taku Watanabe and Yoshihiro Yayama and Hiroki Iriguchi and Yusuke Asano and Tasuku Onodera and Takafumi Ishii and Takao Kudo and Hideki Ono and Ryohto Sawada and Ryuichiro Ishitani and Marc Ong and Taiki Yamaguchi and Toshiki Kataoka and Akihide Hayashi and Takeshi Ibuka},
      year={2021},
      eprint={2106.14583},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```
