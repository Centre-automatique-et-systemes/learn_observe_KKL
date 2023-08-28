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
from learn_KKL.learner_noise import LearnerNoise
from learn_KKL.system import SaturatedVanDerPol, VanDerPol, OldSaturatedVanDerPol
from learn_KKL.luenberger_observer_noise import LuenbergerObserverNoise

# Import learner utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import torch.optim as optim

if __name__ == "__main__":

    TRAIN = False

    ##########################################################################
    # Setup observer #########################################################
    ##########################################################################

    # Learning method
    learning_method = "Supervised_noise"
    num_hl = 5
    size_hl = 50
    activation = nn.SiLU()

    # Define system
    system = SaturatedVanDerPol()

    # Define data params
    wc_arr = np.linspace(0.03, 1., 100)
    x_limits = np.array([[-2.7, 2.7], [-2.7, 2.7]])
    num_samples = wc_arr.shape[0] * 5000

    # Solver options
    solver_options = {'method': 'rk4', 'options': {'step_size': 1e-3}}

    if TRAIN:
        # Instantiate observer object
        observer = LuenbergerObserverNoise(
            dim_x=system.dim_x,
            dim_y=system.dim_y,
            wc_array=wc_arr,
            method=learning_method,
            activation=activation,
            num_hl=num_hl,
            size_hl=size_hl,
            solver_options=solver_options,
        )
        observer.set_dynamics(system)

        # Generate training data and validation data
        data = observer.generate_data_svl(x_limits, wc_arr, num_samples,
                                          method="LHS")
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
        learner_T = LearnerNoise(
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
        logger = TensorBoardLogger(
            save_dir=learner_T.results_folder + "/tb_logs")
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

        learner_T.save_results(
            checkpoint_path=checkpoint_callback.best_model_path, )

        learner_T.save_plot(
            "Train_loss.pdf",
            "Training loss over time",
            "log",
            learner_T.train_loss.detach(),
        )
        learner_T.save_plot(
            "Val_loss.pdf",
            "Validation loss over time",
            "log",
            learner_T.val_loss.detach(),
        )

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
        learner_T_star = LearnerNoise(
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
        logger = TensorBoardLogger(save_dir=learner_T_star.results_folder + "/tb_logs")
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

        learner_T_star.save_results(
            checkpoint_path=checkpoint_callback.best_model_path)

        learner_T_star.save_plot(
            "Train_loss.pdf",
            "Training loss over time",
            "log",
            learner_T_star.train_loss.detach(),
        )
        learner_T_star.save_plot(
            "Val_loss.pdf",
            "Validation loss over time",
            "log",
            learner_T_star.val_loss.detach(),
        )

    else:
        # Load learner
        path = "runs/SaturatedVanDerPol/Supervised_noise/T_star/" \
               "exp_100_wc0.03-1_-2727_rk41e-3_2"
        learner_path = path + "/learner.pkl"
        import dill as pkl

        with open(learner_path, "rb") as rb_file:
            learner_T_star = pkl.load(rb_file)
        learner_T_star.results_folder = path
        observer = learner_T_star.model
        verbose = False
        save = True

    ##########################################################################
    # Generate plots #########################################################
    ##########################################################################

    # Params
    nb = int(np.min([len(learner_T_star.training_data), 10000]))
    idx = np.random.choice(np.arange(len(learner_T_star.training_data)),
                           size=(nb,), replace=False)
    verbose = False

    learner_T_star.save_pdf_training(learner_T_star.training_data[idx], verbose)

    # Gain criterion
    print('Computing our gain-tuning criterion can take some time but saves '
          'intermediary data in a subfolder xzi_mesh: if you have already run '
          'this script, set save to False and path to this subfolder.')
    save = False
    path = ''
    if save:
        mesh = learner_T_star.model.generate_data_svl(
            x_limits, wc_arr, 10000 * len(wc_arr), method="uniform", stack=False
        )
    else:
        mesh = torch.randn((10, learner_T_star.model.dim_x +
                           learner_T_star.model.dim_z, 1))
    # learner_T_star.save_rmse_wc(mesh, wc_arr, verbose)
    # learner_T_star.plot_sensitiviy_wc(mesh, wc_arr, verbose, save=save,
    #                                   path=path)

    # Trajectories
    std_array = [0.0, 0.25, 0.5]
    wc_arr = np.array([0.03, 0.2, 1.])
    tsim = (0, 30)
    dt = 1e-2
    x_0 = torch.tensor([0.1, 0.1])
    z_0 = learner_T_star.model.encoder(
        torch.cat((x_0.expand(len(wc_arr), -1),
                   torch.as_tensor(wc_arr).reshape(-1, 1)), dim=1))
    # if save:
    #     mesh = learner_T_star.model.generate_data_svl(
    #         x_limits, wc_arr, 10000 * len(wc_arr), method="uniform", stack=False
    #     )
    # else:
    #     mesh = torch.randn((10, learner_T_star.model.dim_x +
    #                         learner_T_star.model.dim_z, 1))
    #
    # learner_T_star.plot_traj_rmse(x_0, wc_arr, verbose, tsim, dt, std=0.25)
    #
    # for std in std_array:
    #     learner_T_star.save_trj(
    #        x_0, wc_arr, 0, verbose, tsim, dt, std=std#, z_0=z_0
    #     )
    #     learner_T_star.plot_traj_error(
    #         x_0, wc_arr, 0, verbose, tsim, dt, std=std#, z_0=z_0
    #     )

    # Heatmap
    if not save:
        mesh = learner_T_star.model.generate_data_svl(
            x_limits, wc_arr, 10000, method="uniform", stack=False
        )
    learner_T_star.save_pdf_heatmap(mesh, verbose)
    learner_T_star.save_invert_heatmap(mesh, verbose)

    # # Criterion
    # from eval_qqs2_results_individual import plot_crit
    # import os
    # plot_crit(os.path.join(path, 'xzi_mesh'), N=10000, verbose=verbose)
