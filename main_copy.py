from learn_KKL.learner import Learner
# from learn_KKL.system import RevDuffing
from learn_KKL.system import VanDerPol
from learn_KKL.luenberger_observer import LuenbergerObserver
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
import pandas as pd
# from learn_KKL.utils import generate_mesh
from torch import nn

sys.path.append("../")


sb.set_style("whitegrid")

if __name__ == "__main__":
    system = VanDerPol()

    # Instantiate the observer
    # D = torch.diag_embed(torch.tensor([ -9.3700,  -9.3700, -11.8325]))
    observer = LuenbergerObserver(dim_x=2, dim_y=1, method='Supervised',
                                recon_lambda=0.8, wc=0.3, activation=nn.Tanh())
    observer.set_dynamics(system)
    # Generate (x_i, z_i) data by running system backward, then system + observer
    # forward in time

    # data = pd.read_csv('/Users/lukasbahr/Downloads/exp_0/training_data.csv')
    # data = torch.tensor(data.values[:,1:])
    # # data = observer.generate_data_svl(np.array([[-1, 1.], [-1., 1.]]), 72000)
    # # grid too large leads to underflow when simulating backward in time

    # val_data = pd.read_csv('/Users/lukasbahr/Downloads/exp_0/validation_data.csv')
    # val_data = torch.tensor(val_data.values[:,1:])
    data = observer.generate_data_svl(np.array([[-1.0, 1.0], [-1.0, 1.0]]), 72000)
    # grid too large leads to underflow when simulating backward in time
    data, val_data = train_test_split(data, test_size=0.3, shuffle=True)

    # Train the inverse transformation using pytorch-lightning and the learner class
    # Options for training
    trainer_options={'max_epochs': 20}
    optimizer_options = {'weight_decay': 1e-6}
    scheduler_options = {'mode': 'min', 'factor': 0.1, 'patience': 3,
                        'threshold': 1e-4, 'verbose': True}
    stopper = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss', min_delta=5e-4, patience=3, verbose=False, mode='min')
    # Instantiate learner
    learner_T_star = Learner(observer=observer, system=system, training_data=data,
                            validation_data=val_data, method='T_star',
                            batch_size=10, lr=1e-3, optimizer=optim.Adam,
                            optimizer_options=optimizer_options,
                            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                            scheduler_options=scheduler_options)
    # Define logger and checkpointing
    logger = TensorBoardLogger(save_dir=learner_T_star.results_folder + '/tb_logs')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    trainer = pl.Trainer(
        callbacks=[stopper, checkpoint_callback], **trainer_options, logger=logger,
        log_every_n_steps=1, check_val_every_n_epoch=3)

    # To see logger in tensorboard, copy the following output name_of_folder
    print(f"Logs stored in {learner_T_star.results_folder}/tb_logs")

    # Train and save results
    trainer.fit(learner_T_star)

    

    # To see logger in tensorboard, copy the following output name_of_folder
    print(f'Logs stored in {learner_T_star.results_folder}/tb_logs')
    # which should be similar to jupyter_notebooks/runs/method/exp_0/tb_logs/
    # Then type this in terminal:
    # tensorboard --logdir=name_of_folder --port=8080

    # Train and save results
    trainer.fit(learner_T_star)
    learner_T_star.save_results(limits=np.array([[-1., 1.], [-1.,1.]]),
                                nb_trajs=10, tsim=(0, 60), dt=1e-2,
                                checkpoint_path=checkpoint_callback.best_model_path)
