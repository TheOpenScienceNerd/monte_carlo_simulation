[![ORCID: Monks](https://img.shields.io/badge/Tom_Monks_ORCID-0000--0003--2631--4481-brightgreen)](https://orcid.org/0000-0003-2631-4481)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Monte Carlo Simulation in Python

Examples and tutorials for conducting monte-carlo simulation in Python.  

## License

The materials have been made available under an [MIT license](LICENCE).  The materials are as-is with no liability for the author. Please provide credit if you reuse the code in your own work.

## Citation

Please feel free to use or adapt the code for your own work. But if so then a citation would be very much appreciated! 

```bibtex
@software{opensciencenerd_montecarlo,
author = {Monks, Thomas },
license = {MIT},
title = {{An introduction to monte-carlo simulation in Python}},
url = {https://github.com/TheOpenScienceNerd/replications-algorithm}
}
```

## Installation instructions

### Installing dependencies

All dependencies can be found in [`binder/environment.yml`]() and are pulled from conda-forge.  To run the code locally, we recommend installing [miniforge](https://github.com/conda-forge/miniforge);

> miniforge is Free and Open Source Software (FOSS) alternative to Anaconda and miniconda that uses conda-forge as the default channel for packages. It installs both conda and mamba (a drop in replacement for conda) package managers.  We recommend mamba for faster resolving of dependencies and installation of packages. 

navigating your terminal (or cmd prompt) to the directory containing the repo and issuing the following command:

```bash
mamba env create -f binder/environment.yml
```

Activate the mamba environment using the following command:

```bash
mamba activate mc
```

Run Jupyter-lab

```bash
jupyter-lab
```

## Repo overview

```
.
├── binder
│   └── environment.yml
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── 01_mc_investment_decision.ipynb
├── 02_mc_newsvendor.ipynb
├── newsvendor.py
└── README.md
```

* `binder/environment.yml` - contains the conda environment if you wish to work the models.
* `01_mc_investment_decision.ipynb` - an investment decision problem from Pidd (2004).
* `02_mc_newsvendor.ipynb` - a simple multi-period newsvendor problem monte carlo simulation
* `newsvendor.py` - module containing newsvendor problem code
* `CHANGES.md` - changelog with record of notable changes to project between versions.
* `CITATION.cff` - citation information for the package.
* `LICENSE` - details of the MIT permissive license of this work.

