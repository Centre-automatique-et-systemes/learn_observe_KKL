# Learning to observe: neural network-based KKL observers

### To run the code:
- create a directory (further named dir), `cd dir`
- clone the repo in dir/repo
- unzip Data.zip in dir/Data
- create a virtual environment in dir (with pip: `python3 -m venv 
  venv`), source it (`source venv/bin/activate`)
- go to dir/repo, then run `pip install -e .` to install the package
- install interpolation repo: in dir, `git clone https://github.com/aliutkus/torchinterp1d`, `cd torchinterp1d`, `pip install -e .`

### Content
The directory `learn_KKL` contains the main files: `system.py` contains the 
dynamical systems considered (Van der Pol...), `luenberger_observer.py` 
contains the KKL observer (architecture of the encoder and decoder, forward 
functions to train and use them...), and `learner.py` contains a utility 
class for training the observer based on pytorch lightning. The user is 
encouraged to add their dynamical systems in `system.py`, and to write their 
own learner class if they need more advanced functionalities.

**Tutorials** are provided in the directory `jupyter_notebooks`. It contains 
four base cases: two systems (Van der Pol and the reverse Duffing oscillator)
and two designs for the observer (autoencoder and supervised learning). The 
user is encouraged to first run the tutorials in order to understand how the 
toolbox is structured.

### To reproduce the results of the paper:
Supervised learning with dependency on w_c: run `python 
experiments_noise/0_supervised_revduffing.py` for the reverse Duffing 
experiments,
`python experiments/5_supervised_qqs2_meas1.py` for the 
Qube experiments.
The final plots for our gain tuning criterion were obtained in Matlab by 
running `criterion.m` on the data obtained from the previous scripts, since 
there are no native python functions for computing the H-infinity and H-2 
norms in our criterion (the plot given by python is only an approximation).

Autoencoder with D optimized jointly: run `python 
experiments_noise/2_ae-d_revduffing.
py` for the reverse Duffing oscillator or `python 
experiments_noise/3_ae-d_vanderpol.
py` for the Saturated Van der Pol.

### If you use this toolbox, please cite:
```
@article{paper,  
author={M. {Buisson-Fenet} and L. {Bahr} and and F. {Di Meglio}},  
title={Towards gain tuning for numerical KKL observers}
}
```
