# Learning to observe: neural network-based KKL observers

### To run the code:
- create a directory (further named dir), `cd dir`
- clone the repo in dir/repo
- create a virtual environment for this directory (with pip: `python3 -m venv 
  venv`), source it (`source venv/bin/activate`)
- install interpolation repo: `git clone https://github.com/aliutkus/torchinterp1d` into dir, `cd torchinterp1d`, `pip install -e .`
- go to the repo, then run `pip install -e .` to install the package

### Contents
The directory `learn_KKL` contains the main files: `system.py` contains the 
dynamical systems considered (Van der Pol...), `luenberger_observer.py` 
contains the KKL observer (architecture of the encoder and decoder, forward 
functions to train and use them...), and `learner.py` contains a utility 
class for training the observer based on pytorch lightning. The user is 
encouraged to add their dynamical systems in `system.py`, and to write their 
own learner class if they need more advanced functionalities.

**Tutorials** are provided in the directory `jupyter_notebooks`. It contains 
four base cases: two systems (Van der Pol and the inverse Duffing oscillator)
and two designs for the observer (autoencoder and supervised learning). The 
user is encouraged to first run the tutorials in order to understand how the 
toolbox is structured.

### To reproduce the results of the paper:

### If you use this toolbox, please cite:
