# -*- coding: utf-8 -*-

# Import base utils
import copy
import pathlib
import sys

import numpy as np
import torch
from torch import nn

# In order to import learn_KKL we need to add the working dir to the system path
working_path = str(pathlib.Path().resolve())
sys.path.append(working_path)

# Import KKL observer
from learn_KKL.learner import Learner
from learn_KKL.system import QuanserQubeServo2
from learn_KKL.luenberger_observer import LuenbergerObserver
from learn_KKL.utils import generate_mesh, RMSE
from learn_KKL.filter_utils import EKF_ODE, interpolate_func, \
    dynamics_traj_observer

# Import learner utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torchdiffeq import odeint
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt

# Script to learn a KKL observer from simulations of the Quanser Qube 2,
# test it on experimental data, and compare with EKF.
# Quanser Qube: state (theta, alpha, thetadot, alphadot)

# For continuous data need theta in [-pi, pi] and alpha in [0, 2pi]: only
# managed to train KKL for angles that stay in this range! Rigorously should
# take extended state (x1, x2) = (cos(theta), sin(theta)) and (x3, x4) = (
# cos(alpha), sin(alpha)) to avoid this issue and (hopefully) train on whole
# state-space. Numerical results are only local for now.

if __name__ == "__main__":

    ##########################################################################
    # Setup observer #########################################################
    ##########################################################################
    # Learning method
    learning_method = "Supervised"
    num_hl = 5
    size_hl = 50
    activation = nn.ReLU()
    recon_lambda = 0.1

    # Define system
    system = QuanserQubeServo2()

    # Define data params (same characteristics as experimental data)
    dt = 0.004
    tsim = (0, 2000 * dt)
    num_initial_conditions = 20
    init_wc = 3.
    # x_limits = np.array([[0., 0.1], [0., 0.1], [0., 0.1], [0., 0.1]])
    x_limits = np.array([[0.0, 0.001], [0.05, 0.1], [0.0, 0.001], [0.0, 0.001]])

    # Solver options
    solver_options = {'method': 'rk4', 'options': {'step_size': 1e-3}}

    # Create the observer
    observer = LuenbergerObserver(
        dim_x=system.dim_x, dim_y=system.dim_y, method=learning_method,
        wc=init_wc, recon_lambda=recon_lambda
    )
    observer.set_dynamics(system)

    # Generate data
    data = observer.generate_trajectory_data(
        x_limits, num_initial_conditions, method="LHS", tsim=tsim,
        stack=False, dt=dt
    )
    theta = data[:, :, 0]
    alpha = data[:, :, 1]
    # Map to [-pi,pi]
    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    # alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi
    # Map to [0, 2pi]
    # theta = theta % (2 * np.pi)
    alpha = alpha % (2 * np.pi)
    data[:, :, 0] = theta
    data[:, :, 1] = alpha
    data_ordered = copy.deepcopy(data)
    data = torch.cat(torch.unbind(data, dim=1), dim=0)
    data, val_data = train_test_split(data, test_size=0.3, shuffle=False)

    print(data.shape)

    ##########################################################################
    # Setup learner ##########################################################
    ##########################################################################

    # Trainer options
    num_epochs = 100
    trainer_options = {"max_epochs": num_epochs}
    batch_size = 20
    init_learning_rate = 1e-2

    # Optim options
    optim_method = optim.Adam
    optimizer_options = {"weight_decay": 1e-6}

    # Scheduler options
    scheduler_method = optim.lr_scheduler.ReduceLROnPlateau
    scheduler_options = {
        "mode": "min",
        "factor": 0.5,
        "patience": 3,
        "threshold": 1e-4,
        "verbose": True,
    }
    stopper = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss", min_delta=5e-4, patience=10, verbose=False,
        mode="min"
    )

    # Instantiate learner
    learner = Learner(
        observer=observer,
        system=system,
        training_data=data,
        validation_data=val_data,
        method='T_star',
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
        check_val_every_n_epoch=3
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

    # To see logger in tensorboard, copy the following output name_of_folder
    print(f"Logs stored in {learner.results_folder}/tb_logs")

    # Plot training data (as trajectories)
    plt.plot(data_ordered[..., 0], 'x')
    plt.title(r'Training data: $\theta$')
    plt.savefig(os.path.join(learner.results_folder, 'Train_theta.pdf'))
    plt.clf()
    plt.close('all')
    plt.plot(data_ordered[..., 1], 'x')
    plt.title(r'Training data: $\alpha$')
    plt.savefig(os.path.join(learner.results_folder, 'Train_alpha.pdf'))
    plt.clf()
    plt.close('all')
    plt.plot(data_ordered[..., 2], 'x')
    plt.title(r'Training data: $\dot{\theta}$')
    plt.savefig(os.path.join(learner.results_folder, 'Train_thetadot.pdf'))
    plt.clf()
    plt.close('all')
    plt.plot(data_ordered[..., 3], 'x')
    plt.title(r'Training data: $\dot{\alpha}$')
    plt.savefig(os.path.join(learner.results_folder, 'Train_alphadot.pdf'))
    plt.clf()
    plt.close('all')

    with torch.no_grad():
        # Save training and validation data
        idx = np.random.choice(
            np.arange(len(learner.training_data)), size=(10000,)
        )  # subsampling for plots

        specs_file = learner.save_specifications()

        learner.save_pkl("/learner.pkl", learner)

        learner.save_csv(
            learner.training_data.cpu().numpy(),
            os.path.join(learner.results_folder, "training_data.csv"),
        )
        learner.save_csv(
            learner.validation_data.cpu().numpy(),
            os.path.join(learner.results_folder, "validation_data.csv"),
        )
        # Loss plot over time
        learner.save_plot(
            "Train_loss.pdf",
            "Training loss over time",
            "log",
            learner.train_loss.detach(),
        )
        learner.save_plot(
            "Val_loss.pdf", "Validation loss over time", "log", learner.val_loss.detach(),
        )
        xmesh = generate_mesh(x_limits, 10, method="uniform")
        learner.save_random_traj(x_mesh=xmesh, num_samples=10, nb_trajs=10,
                                 verbose=False, tsim=tsim, dt=dt)

    ##########################################################################
    # Test trajectory ########################################################
    ##########################################################################

    # # Load learner  # TODO
    # path = "runs/QuanserQubeServo2/Supervised/T_star/x0_00.1"
    # learner_path = path + "/learner.pkl"
    # import dill as pkl
    # with open(learner_path, "rb") as rb_file:
    #     learner = pkl.load(rb_file)
    # learner.results_folder = path
    # observer = learner.model

    # Experiment
    fileName = 'example_csv_fin4'
    filepath = '../Data/QQS2_data_diffx0/' + fileName + '.csv'
    exp = np.genfromtxt(filepath, delimiter=',')
    exp = exp[1:2001, 1:-1]
    exp_copy = copy.deepcopy(exp)
    exp[:, 0], exp[:, 1] = exp_copy[:, 1], exp_copy[:, 0]
    exp[:, 2], exp[:, 3] = exp_copy[:, 3], exp_copy[:, 2]
    # Map to [0, 2pi]
    exp[:, 1] = exp[:, 1] % (2 * np.pi)
    exp = torch.from_numpy(exp)

    # Observer
    measurement = torch.unsqueeze(exp[..., 1], 1)
    tq = torch.arange(tsim[0], tsim[1], dt)
    y = torch.cat((tq.unsqueeze(1), measurement[:, 0].unsqueeze(1)), dim=1)
    estimation = observer.predict(y, tsim, dt).detach()
    theta = estimation[..., 0]
    alpha = estimation[..., 1]
    # Map to [-pi,pi]
    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    # alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi
    # Map to [0, 2pi]
    # theta = theta % (2 * np.pi)
    alpha = alpha % (2 * np.pi)
    estimation[..., 0] = theta
    estimation[..., 1] = alpha

    # Compare both
    os.makedirs(os.path.join(learner.results_folder, fileName), exist_ok=True)
    rmse = RMSE(exp, estimation, dim=0)
    for i in range(estimation.shape[1]):
        plt.plot(tq, exp[:, i], label=rf'$x_{i + 1}$')
        plt.plot(tq, estimation[:, i], '--', label=rf'$\hat{{x}}_{i + 1}$')
        plt.title(rf'Test trajectory for $\omega_c$ = {init_wc:0.2g}, RMSE = '
                  rf'{rmse[i]:0.2g}')
        plt.xlabel(rf"$t$")
        plt.ylabel(rf"$x_{i + 1}$")
        plt.legend()
        plt.savefig(
            os.path.join(learner.results_folder, fileName, f'Traj{i}.pdf'),
            bbox_inches="tight"
        )
        plt.clf()
        plt.close('all')

    # Compare EKF
    x0 = exp[0].unsqueeze(0)
    dyn_config = {'prior_kwargs': {
        'n': x0.shape[1],
        'observation_matrix': torch.tensor([[0., 1., 0., 0.]]),
        'EKF_process_covar': 1e-1 * torch.eye(x0.shape[1]),
        'EKF_init_covar': torch.tensor([1e-4, 1e-3, 1e-2, 1e-1]) * torch.eye(
            x0.shape[1]),
        'EKF_meas_covar': 1e-3 * torch.eye(measurement.shape[1])}}
    EKF_observer = EKF_ODE('cpu', dyn_config)
    y_func = interpolate_func(x=y, t0=tq[0], init_value=measurement[0])
    controller = lambda t, kwargs, t0, init_control, impose_init: 0.
    x0_estim = torch.cat((
        torch.zeros(1, 1), measurement[0].unsqueeze(1), torch.zeros(1, 2),
        torch.unsqueeze(torch.flatten(dyn_config['prior_kwargs'][
                                          'EKF_init_covar']), 0)), dim=1)
    xtraj = dynamics_traj_observer(
        x0=x0_estim, u=controller, y=y_func, t0=tq[0],
        dt=dt, init_control=0., version=EKF_observer, t_eval=tq, GP=system,
        kwargs=dyn_config)
    estimation_EKF = xtraj[:, :exp.shape[1]]
    rmse_EKF = RMSE(exp, estimation_EKF, dim=0)
    for i in range(estimation.shape[1]):
        plt.plot(tq, exp[:, i], label=rf'$x_{i + 1}$')
        plt.plot(tq, estimation[:, i], '--', label=rf'$\hat{{x}}_{i + 1}$')
        plt.plot(tq, estimation_EKF[:, i], '-.', label=rf'$\hat{{x}}_{i + 1}^{{EKF}}$')
        plt.xlabel(rf"$t$")
        plt.ylabel(rf"$x_{i + 1}$")
        plt.title(rf'RMSE = {rmse[i]:0.2g} for KKL, RMSE = '
                  rf'{rmse_EKF[i]:0.2g} for EKF')
        plt.legend()
        plt.savefig(
            os.path.join(learner.results_folder, fileName, f'Traj_EKF{i}.pdf'),
            bbox_inches="tight"
        )
        plt.clf()
        plt.close('all')
