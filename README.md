# Learning to observe: neural network-based KKL observers

### To run the code:
- create a directory (further named dir), `cd dir`
- clone the repo in dir/repo
- unzip `Data/QQS2_data_diffx0.zip` in dir/repo/Data
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

The **experiments** folder contains scripts running numerical KKL on 
different systems. The **experiments_noise** folders contains similar 
scripts, but for taking sensitivity to measurement noise into account. This 
is done by learning the transformations as functions of a parameter 
$\omega_c$ (supervised setting) then computing an empirical gain tuning 
criterion, or jointly optimizing $D$ (unsupervised setting). It is also 
possible to run the scripts of the **experiments** folder sequentially for 
many values of $D$, then evaluate the gain criterion a posteriori using 
`experiments_noise/eval_qqs2_results_individual.py`.

The **Data** folder contains a zip file with the Quanser Qube data: 
unzip it in `Data/QQS2_data_diffx0` to reproduce the Qube results.

### To reproduce the results of the paper:
Supervised learning with dependency on w_c: run `python 
experiments_noise/0_supervised_revduffing.py` for the reverse Duffing 
experiments,
, `python 
experiments_noise/1_supervised_saturated_vanderpol.py` for the Saturated Van der 
Pol experiments.
For the Qube experiments, run `./launch_script1.sh`, which launches `python 
experiments/5_supervised_qqs2_meas1.py` sequentially for many 
indepedent values of $\omega_c$ instead of learning the transformation as 
a function of $\omega_c$.
Then run `python experiments_noise/eval_qqs2_results_individual.py` to compute 
the 
gain tuning criterion and test trajectories over these many independent 
values of $\omega_c$.
The final plots for our gain tuning criterion were obtained in Matlab by 
running `criterion.m` on the data saved by the previous scripts, since 
there are no native python functions for computing the H-infinity and H-2 
norms in our criterion (the plot given by python is only an approximation).
Running `python experiments_noise/eval_qqs2_results_individual.plot_crit(...)
` on that criterion computed by Matlab then yields the final plots in the paper.

Autoencoder with D optimized jointly: run `python 
experiments_noise/2_ae-d_revduffing.
py` for the reverse Duffing oscillator or `python 
experiments_noise/3_ae-d_vanderpol.
py` for the Saturated Van der Pol.

### If you use this toolbox, please cite:
```
@article{paper,  
author={M. {Buisson-Fenet} and L. {Bahr} and V. {Morgenthaler} and F. {Di 
Meglio}},  
title={Towards gain tuning for numerical KKL observers},
journal = {arXiv preprint arXiv:2204.00318},
url = {http://arxiv.org/abs/2204.00318},
year = {2022}
}
```
