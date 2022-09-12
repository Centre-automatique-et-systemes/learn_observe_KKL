# -*- coding: utf-8 -*-

import pathlib
import sys

import numpy as np
# Import base utils
from torch import nn

# In order to import learn_KKL we need to add the working dir to the system path
working_path = str(pathlib.Path().resolve())
sys.path.append(working_path)

# Import KKL observer
from learn_KKL.learnerV2 import Learner
from learn_KKL.system import SaturatedVanDerPol
from learn_KKL.luenberger_observer import LuenbergerObserver

# Import learner utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import torch.optim as optim

if __name__ == "__main__":
    ##########################################################################
    # Setup observer #########################################################
    ##########################################################################

    # Learning method
    learning_method = "Supervised"
    num_hl = 5
    size_hl = 50
    activation = nn.SiLU()

    # Define system
    system = SaturatedVanDerPol()

    # Define data params
    wc = 0.2
    x_limits = np.array([[-2.7, 2.7], [-2.7, 2.7]])
    num_samples = 50000

    # Solver options
    solver_options = {'method': 'rk4', 'options': {'step_size': 1e-3}}

    # Instantiate observer object
    observer = LuenbergerObserver(
        dim_x=system.dim_x,
        dim_y=system.dim_y,
        wc=wc,
        method=learning_method,
        activation=activation,
        num_hl=num_hl,
        size_hl=size_hl,
        solver_options=solver_options,
    )
    observer.set_dynamics(system)

    # Generate training data and validation data
    data = observer.generate_data_svl(x_limits, num_samples, method="LHS", k=10)
    data, val_data = train_test_split(data, test_size=0.3, shuffle=True)

    ##########################################################################
    # Setup learner ##########################################################
    ##########################################################################

    # Trainer options
    num_epochs = 100
    trainer_options = {"max_epochs": num_epochs}
    batch_size = 100
    init_learning_rate = 5e-3

    # Optim options
    optim_method = optim.Adam
    optimizer_options = {"weight_decay": 1e-8}

    # Scheduler/stopper options
    scheduler_method = optim.lr_scheduler.ReduceLROnPlateau
    scheduler_options = {
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
        "threshold": 1e-4,
        "verbose": True,
    }
    stopper = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False,
        mode="min"
    )

    # Instantiate learner for T
    learner_T = Learner(
        observer=observer,
        system=system,
        training_data=data,
        validation_data=val_data,
        method="T",
        batch_size=batch_size,
        lr=init_learning_rate,
        optimizer=optim_method,
        optimizer_options=optimizer_options,
        scheduler=scheduler_method,
        scheduler_options=scheduler_options,
    )

    # Define logger and checkpointing
    logger = TensorBoardLogger(save_dir=learner_T.results_folder + "/tb_logs")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(
        callbacks=[stopper, checkpoint_callback],
        **trainer_options,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=3,
    )

    # To see logger in tensorboard, copy the following output name_of_folder
    print(f"Logs stored in {learner_T.results_folder}/tb_logs")

    # Train the transformation function using the learner class
    trainer.fit(learner_T)

    ##########################################################################
    # Generate plots #########################################################
    ##########################################################################

    learner_T.save_results(limits=x_limits, nb_trajs=10, tsim=(0, 60), dt=1e-2,
                           checkpoint_path=checkpoint_callback.best_model_path)

    # Scheduler/stopper options
    scheduler_method = optim.lr_scheduler.ReduceLROnPlateau
    scheduler_options = {
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
        "threshold": 1e-4,
        "verbose": True,
    }
    stopper = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False,
        mode="min"
    )

    # Instantiate learner for T_star
    learner_T_star = Learner(
        observer=observer,
        system=system,
        training_data=data,
        validation_data=val_data,
        method="T_star",
        batch_size=batch_size,
        lr=init_learning_rate,
        optimizer=optim_method,
        optimizer_options=optimizer_options,
        scheduler=scheduler_method,
        scheduler_options=scheduler_options,
    )

    # Define logger and checkpointing
    logger = TensorBoardLogger(
        save_dir=learner_T_star.results_folder + "/tb_logs")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(
        callbacks=[stopper, checkpoint_callback],
        **trainer_options,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=3,
    )

    # To see logger in tensorboard, copy the following output name_of_folder
    print(f"Logs stored in {learner_T_star.results_folder}/tb_logs")

    # Train the transformation function using the learner class
    trainer.fit(learner_T_star)

    ##########################################################################
    # Generate plots #########################################################
    ##########################################################################

    learner_T_star.save_results(
        limits=x_limits, nb_trajs=10, tsim=(0, 60), dt=1e-2,
        checkpoint_path=checkpoint_callback.best_model_path)
