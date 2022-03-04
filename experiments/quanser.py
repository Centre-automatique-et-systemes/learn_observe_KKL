# -*- coding: utf-8 -*-

# In order to import learn_KKL we need to add the working dir to the system path
import pathlib; working_path = str(pathlib.Path().resolve())
import sys; sys.path.append(working_path)

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


sb.set_style("whitegrid")

if __name__ == "__main__":
    system = QuanserQubeServo2()

    observer = LuenbergerObserver(
        dim_x=system.dim_x,
        dim_y=system.dim_y,
        method="Supervised",
        wc=1.5,
        recon_lambda=0.7,
        D="direct",
    )

    observer.set_dynamics(system)

    tsim = (0, 10)
    dt = 1e-2
    x_0 = torch.tensor(
        [[0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]]
    )
    _, data = observer.simulate_system(x_0, tsim, dt)

    data, val_data = train_test_split(data[:, 0, :], test_size=0.01, shuffle=True)

    # Train the forward transformation using pytorch-lightning and the learner class
    # Options for training
    trainer_options = {"max_epochs": 30}
    optimizer_options = {"weight_decay": 1e-6}
    scheduler_options = {
        "mode": "min",
        "factor": 0.1,
        "patience": 3,
        "threshold": 1e-4,
        "verbose": True,
    }
    stopper = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss", min_delta=5e-4, patience=3, verbose=False, mode="min"
    )
    # Instantiate learner
    learner_T_star = Learner(
        observer=observer,
        system=system,
        training_data=data,
        validation_data=val_data,
        method="T_star",
        batch_size=10,
        lr=1e-3,
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

    # To see logger in tensorboard, copy the following output name_of_folder
    print(f"Logs stored in {learner_T_star.results_folder}/tb_logs")

    # Train and save results
    trainer.fit(learner_T_star)

    tq, simulation = system.simulate(x_0, tsim, dt)
    measurement = system.h(simulation)
    # Save these test trajectories
    # Need to figure out how to interpolate y in parallel for all
    # trajectories!!!
    y = torch.cat((tq.unsqueeze(1), measurement[:,0].unsqueeze(1)), dim=1)
    estimation = observer.predict(y, tsim, dt).detach()

    plt.plot(tq, simulation.detach().numpy(), label=rf"$x$")
    plt.plot(tq, estimation.detach().numpy(), label=rf"$\hat{{x}}$")
    plt.legend()
    plt.xlabel(rf"$t$")
    plt.ylabel(rf"$x$")

    plt.show()

    # limits = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    # learner_T.save_results(
    #     limits=limits,
    #     wc_arr_train=wc_arr,
    #     nb_trajs=10,
    #     t_sim=(0, 40),
    #     dt=1e-2,
    #     checkpoint_path=checkpoint_callback.best_model_path,
    # )

    # learner_T.save_plot(
    #     "Train_loss.pdf",
    #     "Training loss over time",
    #     "log",
    #     learner_T.train_loss.detach(),
    # )
    # learner_T.save_plot(
    #     "Val_loss.pdf", "Validation loss over time", "log", learner_T.val_loss.detach()
    # )

    # idx = np.random.choice(np.arange(len(learner_T.training_data)), size=(10000,))
    # verbose = False
    # num_samples = 70000
    # mesh = learner_T.model.generate_data_svl(
    #     limits, wc_arr, num_samples, method="uniform", stack=False
    # )

    # learner_T.save_pdf_training(learner_T.training_data[idx], verbose)
    # learner_T.save_trj(
    #     torch.tensor([0.225, -0.131]), wc_arr, 5, verbose, (0, 70), 1e-2, var=0.0
    # )
    # learner_T.save_pdf_heatmap(mesh, verbose)

    # mesh = learner_T.model.generate_data_svl(
    #     limits, wc_arr, 1 * 100, method="LHS", stack=False
    # )
    # learner_T.save_rmse_wc(mesh, wc_arr, verbose)
    # learner_T.plot_sensitiviy_wc(mesh, wc_arr, verbose)
