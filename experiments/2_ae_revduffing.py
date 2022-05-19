# -*- coding: utf-8 -*-

# Import base utils
from mimetypes import init
import torch
from torch import nn
import numpy as np
import pathlib
import sys

# In order to import learn_KKL we need to add the working dir to the system path
working_path = str(pathlib.Path().resolve())
sys.path.append(working_path)

# Import KKL observer
from learn_KKL.learner import Learner
from learn_KKL.system import RevDuffing
from learn_KKL.luenberger_observer import LuenbergerObserver
from learn_KKL.utils import generate_mesh

# Import learner utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import torch.optim as optim
import sys


if __name__ == "__main__":

    ##########################################################################
    # Setup observer #########################################################
    ##########################################################################
    # Learning method
    learning_method = "Autoencoder"
    num_hl = 5
    size_hl = 50
    activation = nn.ReLU()
    recon_lambda = 0.1

    # Define system
    system = RevDuffing()

    # Define data params
    x_limits = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    num_samples = 5000
    init_wc = 0.2

    # Solver options
    solver_options = {'method': 'rk4', 'options': {'step_size': 1e-3}}

    # Create the observer (autoencoder design)
    observer = LuenbergerObserver(
        dim_x=system.dim_x,
        dim_y=system.dim_y,
        method=learning_method,
        activation=activation,
        num_hl=num_hl,
        size_hl=size_hl,
        solver_options=solver_options,
        wc=init_wc,
        recon_lambda=recon_lambda
    )
    observer.set_dynamics(system)

    # Generate the data
    data = generate_mesh(x_limits, num_samples, method="LHS")
    data, val_data = train_test_split(data, test_size=0.3, shuffle=True)

    ##########################################################################
    # Setup learner ##########################################################
    ##########################################################################

    # Trainer options
    num_epochs = 100
    trainer_options = {"max_epochs": num_epochs}
    batch_size = 100
    init_learning_rate = 1e-3

    # Optim options
    optim_method = optim.Adam
    optimizer_options = {"weight_decay": 1e-8}

    # Scheduler options
    scheduler_method = optim.lr_scheduler.ReduceLROnPlateau
    scheduler_options = {
        "mode": "min",
        "factor": 0.1,
        "patience": 1,
        "threshold": 1e-4,
        "verbose": True,
    }
    stopper = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss", min_delta=5e-4, patience=3, verbose=False,
        mode="min"
    )

    # Instantiate learner
    learner = Learner(
        observer=observer,
        system=system,
        training_data=data,
        validation_data=val_data,
        method=learning_method,
        batch_size=batch_size,
        lr=init_learning_rate,
        optimizer=optim_method,
        optimizer_options=optimizer_options,
        scheduler=scheduler_method,
        scheduler_options=scheduler_options,
    )

    # Define logger and checkpointing
    logger = TensorBoardLogger(save_dir=learner.results_folder + "/tb_logs")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(
        callbacks=[stopper, checkpoint_callback],
        **trainer_options,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=3,
    )

    # To see logger in tensorboard, copy the following output name_of_folder
    print(f"Logs stored in {learner.results_folder}/tb_logs")
    # which should be similar to jupyter_notebooks/runs/method/exp_0/tb_logs/
    # Then type this in terminal:
    # tensorboard --logdir=name_of_folder

    # Train and save results
    trainer.fit(learner)

    ##########################################################################
    # Generate plots #########################################################
    ##########################################################################

    learner.save_results(limits=x_limits, nb_trajs=10, tsim=(0, 50), dt=1e-2,
                         checkpoint_path=checkpoint_callback.best_model_path)