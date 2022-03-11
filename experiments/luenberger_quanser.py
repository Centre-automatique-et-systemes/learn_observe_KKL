# -*- coding: utf-8 -*-

# In order to import learn_KKL we need to add the working dir to the system path
import pathlib

working_path = str(pathlib.Path().resolve())
import sys

sys.path.append(working_path)

from learn_KKL.learner import Learner
from learn_KKL.system import QuanserQubeServo2
from learn_KKL.luenberger_observer import LuenbergerObserver
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import seaborn as sb
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import os
import numpy as np


sb.set_style("whitegrid")

if __name__ == "__main__":
    system = QuanserQubeServo2()

    observer = LuenbergerObserver(
        dim_x=system.dim_x, dim_y=system.dim_y, method="Supervised", wc=3, dim_z=8
    )
    observer.set_dynamics(system)

    tsim = (0, 10)
    dt = 1e-2
    num_initial_conditions = 100
    num_samples = 50000

    # x_limits = np.array([[0.0, 0.001], [0.0, 0.1], [0.0, 0.001], [0.0, 0.001]])
    x_limits = np.array([[-0.4, 0.4], [0.0, 2*np.pi], [-30., 30.], [-15.0, 15.0]])

    # data = observer.generate_trajectory_data(
    #     x_limits, num_initial_conditions, method="LHS", tsim=tsim, stack=True
    # )
    data = observer.generate_data_svl(x_limits, num_samples)
    data, val_data = train_test_split(data, test_size=0.2, shuffle=True)

    # Train the forward transformation using pytorch-lightning and the learner class
    # Options for training
    trainer_options = {"max_epochs": 50}
    optimizer_options = {"weight_decay": 1e-6}
    scheduler_options = {
        "mode": "min",
        "factor": 0.5,
        "patience": 3,
        "threshold": 1e-4,
        "verbose": True,
    }
    stopper = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min"
    )
    # Instantiate learner
    learner_T_star = Learner(
        observer=observer,
        system=system,
        training_data=data,
        validation_data=val_data,
        method="T_star",
        batch_size=20,
        lr=1e-2,
        optimizer=optim.Adam,
        optimizer_options=optimizer_options,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
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

    # Train and save results
    trainer.fit(learner_T_star)

    # To see logger in tensorboard, copy the following output name_of_folder
    print(f"Logs stored in {learner_T_star.results_folder}/tb_logs")
    # which should be similar to jupyter_notebooks/runs/method/exp_0/tb_logs/

    with torch.no_grad():
        # Save training and validation data
        idx = np.random.choice(
            np.arange(len(learner_T_star.training_data)), size=(10000,)
        )  # subsampling for plots

        specs_file = learner_T_star.save_specifications()

        learner_T_star.save_pkl("/learner.pkl", learner_T_star)

        learner_T_star.save_csv(
            learner_T_star.training_data.cpu().numpy(),
            os.path.join(learner_T_star.results_folder, "training_data.csv"),
        )
        learner_T_star.save_csv(
            learner_T_star.validation_data.cpu().numpy(),
            os.path.join(learner_T_star.results_folder, "validation_data.csv"),
        )
        # Loss plot over time
        learner_T_star.save_plot(
            "Train_loss.pdf",
            "Training loss over time",
            "log",
            learner_T_star.train_loss.detach(),
        )
        learner_T_star.save_plot(
            "Val_loss.pdf", "Validation loss over time", "log", learner_T_star.val_loss.detach(),
        )

    x_0 = torch.tensor([0.0, 0.1, 0.0, 0.0]) + abs(np.random.randn(4) * 0.01)
    tq, simulation = system.simulate(x_0, tsim, dt)

    measurement = system.h(simulation)
    # Save these test trajectories
    # Need to figure out how to interpolate y in parallel for all
    # trajectories!!!
    y = torch.cat((tq.unsqueeze(1), measurement[:, 0].unsqueeze(1)), dim=1)
    estimation = observer.predict(y, tsim, dt).detach()

    for i in range(simulation.shape[1]):
        plt.scatter(tq, simulation[:, i].detach().numpy(), label=rf"$x$")
        plt.scatter(
            tq,
            estimation[:, i].detach().numpy(),
            label=rf"$\hat{{x}}$",
            linestyle="dashed",
        )
        plt.legend()
        plt.xlabel(rf"$t$")
        plt.ylabel(rf"$x$")
        plt.show()
