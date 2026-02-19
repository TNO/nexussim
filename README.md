# Badges

These are examples of badges you might want to add to your README:  
Please update the URLs accordingly.

[![Built Status](https://api.cirrus-ci.com/github/<USER>/nexussim.svg?branch=main)](https://cirrus-ci.com/github/<USER>/nexussim)  
[![ReadTheDocs](https://readthedocs.org/projects/nexussim/badge/?version=latest)](https://nexussim.readthedocs.io/en/stable/)  
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/nexussim/main.svg)](https://coveralls.io/r/<USER>/nexussim)  
[![PyPI-Server](https://img.shields.io/pypi/v/nexussim.svg)](https://pypi.org/project/nexussim/)  
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/nexussim.svg)](https://anaconda.org/conda-forge/nexussim)  
[![Monthly Downloads](https://pepy.tech/badge/nexussim/month)](https://pepy.tech/project/nexussim)  
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/nexussim)  
[![PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

---

# nexussim

NexusSim is a cutting-edge simulator designed to assess the performance of applications
deployed across the cloud-fog-edge nexus. It empowers researchers and developers to
model, analyze, and optimize interactions in distributed systems, uncovering insights
into resource usage and application behavior across all layers of the computing
continuum.

---

## Making Changes & Contributing

External contributors should follow the guidelines in
[CONTRIBUTING.rst](CONTRIBUTING.rst)

## Developers' section

Practical information for TNO developers

### Installation

After cloning the project, make sure you put the following in place:

1. Create a python environment based on `python3.11`.  This version is the main version
   for development. Using a python3.11 installation and within the root of the cloned
   repository, run:

```bash
python -m venv .venv 
```
If you are not using `python3.11` or you want to install virtual environments with other
python, look at this [stackoverflow_query](https://stackoverflow.com/questions/70422866/how-to-create-a-venv-with-a-different-python-version) 

2. Install pydynaa.  `pydynaa` comes in wheels for specific python versions and OS support (linux, MacOS).  Get the correct wheel file and pip install it.  

TODO: Include here in the future the instructions to get it from package repositories in ci.gitlab.

3. Make your local repository an editable installation of the `nexussim` package:

```bash
python -m pip install --verbose --editable .
```

## Examples

The `examples` folder, contains several examples on how to create and run nexussim models.  They are executable python scripts.

<!-- This project uses [pre-commit](https://pre-commit.com/). Please make sure to install it before making any changes:

```bash
pip install pre-commit
cd nexussim
pre-commit install 
``` -->
