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
from learn_KKL.learner import Learner
from learn_KKL.system import SaturatedVanDerPol
from learn_KKL.luenberger_observer import LuenbergerObserver
from learn_KKL.utils import generate_mesh

# Import learner utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import torch.optim as optim

if __name__ == "__main__":

    TRAIN = True

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
    system = SaturatedVanDerPol()  # saturation only used for heatmap

    # Define data params
    x_limits = np.array([[-2.7, 2.7], [-2.7, 2.7]])
    num_samples = 70000
    init_wc = 2.

    # Solver options
    solver_options = {'method': 'rk4', 'options': {'step_size': 1e-3}}

    if TRAIN:
        # Create the observer (autoencoder design)
        observer = LuenbergerObserver(
            dim_x=system.dim_x,
            dim_y=system.dim_y,
            method=learning_method,
            activation=activation,
            num_hl=num_hl,
            size_hl=size_hl,
            solver_options=solver_options,
            # D='diag',  # TODO
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
        init_learning_rate = 5e-4

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
            monitor="val_loss", min_delta=1e-4, patience=3, verbose=False,
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
    else:
        # Load learner
        path = "runs/SaturatedVanDerPol/Autoencoder/N7e4_wc2"
        learner_path = path + "/learner.pkl"
        import dill as pkl

        with open(learner_path, "rb") as rb_file:
            learner = pkl.load(rb_file)
        learner.results_folder = path
        observer = learner.model
        verbose = False
        save = True

    ##########################################################################
    # Generate plots #########################################################
    ##########################################################################

    learner.save_results(limits=x_limits, nb_trajs=10, tsim=(0, 60), dt=1e-2,
                         checkpoint_path=checkpoint_callback.best_model_path)
