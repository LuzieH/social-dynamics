# Setup
This code has been loosely tested with `python >= 3.7`. To setup the package clone the repo, open a terminal inside the cloned folder and run:
```
pip install -r requirements.txt
```
Followed by
```
pip install .
```
which will run the package installer.

# Usage
Basic usage of the code is fairly simple and straight forward, simply run `main.py` passing the parameters to be used for output folder (`root_dir`<sup>[1](#1)</sup>) and gin configuration to be used (`gin_files`<sup>[1](#1)</sup> and `gin_bindings`<sup>[2](#2)</sup>).

As an example, in the folder above the cloned repo open a terminal and run:
```
python social-dynamics/main.py --root_dir="experiments_results" --gin_files="social-dynamics/configs/base_config.gin"
```

For more information on how to use Gin Config you may visit the wiki of this repository.
\
\
\
<a name="1">1</a>: Required parameter

<a name="2">2</a>: Non required parameter

# Notebooks
A simple notebook to explore and visualize the results of a given experiment is provided at `jupyter_notebooks/Results.ipynb`.


