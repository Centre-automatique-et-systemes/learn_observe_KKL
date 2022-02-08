from learn_KKL.learner_noise import LearnerNoise
from learn_KKL.system import QuanserQubeServo2
from learn_KKL.system import VanDerPol
from learn_KKL.luenberger_observer_noise import LuenbergerObserverNoise

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
import seaborn as sb
from scipy import signal
import matplotlib.pyplot as plt
import torch.optim as optim
import sys
import torch
import learn_KKL.utils as utils
import numpy.linalg as linalg
from torch import nn

sys.path.append("../")

if __name__ == "__main__":
    system = QuanserQubeServo2()
    tsim = [0, 20]
    dt = 1e-2
    x_0 = torch.tensor([-np.pi+0.01, -np.pi+0.01, 0.0, 0.0])
    # x_0 = torch.tensor([-3.,0.])
    tq, sol = system.simulate(x_0, tsim, dt)
    print(sol)
