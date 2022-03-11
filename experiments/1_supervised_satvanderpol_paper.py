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
from learn_KKL.system import SaturatedVanDerPol
from learn_KKL.luenberger_observer_noise import LuenbergerObserverNoise

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
    learning_method = "Supervised_noise"
    num_hl = 5
    size_hl = 50
    activation = nn.SiLU()

    # Define system
    system = SaturatedVanDerPol(limit=100.)

    # Define data params
    # wc_arr = np.linspace(0.1, 2., 50)
    wc_arr = np.array([0.1])
    x_limits = np.array([[-2.7, 2.7], [-2.7, 2.7]])
    num_samples = wc_arr.shape[0] * 10000

    # Instantiate observer object
    observer = LuenbergerObserverNoise(
        dim_x=system.dim_x,
        dim_y=system.dim_y,
        wc_array=wc_arr,
        method=learning_method,
        activation=activation,
        num_hl=num_hl,
        size_hl=size_hl,
    )
    observer.set_dynamics(system)

    # TODO
    from learn_KKL.system import VanDerPol
    syst = VanDerPol()
    obs = LuenbergerObserverNoise(
        dim_x=system.dim_x,
        dim_y=system.dim_y,
        wc_array=wc_arr,
        method=learning_method,
        activation=activation,
        num_hl=num_hl,
        size_hl=size_hl,
    )
    obs.set_dynamics(syst)
    from learn_KKL.utils import generate_mesh
    from scipy import linalg
    x_limits = np.array([[-1, 1], [-1., 1]])
    # mesh = generate_mesh(limits=x_limits, num_samples=num_samples,
    #                      method="uniform")
    mesh = torch.tensor([[2.7, 2.7]])
    num_samples = mesh.shape[0]  # in case regular grid: changed
    wc = 0.1
    D, F = obs.set_DF(wc)
    k = 1#10
    t_c = k / min(abs(linalg.eig(D)[0].real))
    #
    # Simulation with saturation
    y_0 = torch.zeros((num_samples, obs.dim_x + obs.dim_z))  # TODO
    y_1 = y_0.clone()
    # Simulate only x system backward in time
    tsim = (0, -t_c)
    dt = 1e-2
    y_0[:, : obs.dim_x] = mesh
    tbw, data_bw_sat = observer.simulate_system(y_0, tsim, -dt, only_x=True)
    # Simulate both x and z forward in time starting from the last point
    # from previous simulation
    tsim = (-t_c, 20)
    y_1[:, : obs.dim_x] = data_bw_sat[-1, :, : obs.dim_x]
    tfw, data_fw_sat = observer.simulate_system(y_1, tsim, dt)
    # #
    # # Simulation without saturation
    # y_0 = torch.zeros((num_samples, obs.dim_x + obs.dim_z))  # TODO
    # y_1 = y_0.clone()
    # # Simulate only x system backward in time
    # tsim = (0, -t_c)
    # dt = 1e-2
    # y_0[:, : obs.dim_x] = mesh
    # _, data_bw = obs.simulate_system(y_0, tsim, -dt, only_x=True)
    # # Simulate both x and z forward in time starting from the last point
    # # from previous simulation
    # tsim = (-t_c, 0)
    # y_1[:, : obs.dim_x] = data_bw[-1, :, : obs.dim_x]
    # tq, data_fw = obs.simulate_system(y_1, tsim, dt)
    # print(y_0, y_1, t_c)
    # print(data_fw.shape, data_bw.shape)
    import matplotlib.pyplot as plt
    # plt.plot(tq, torch.flip(data_bw[:, 0, 0], (0, )))
    # plt.plot(tq, data_fw[:, 0, 0])
    plt.plot(torch.flip(tbw, (0,)), torch.flip(data_bw_sat[:, 0, 0], (0,)))
    plt.plot(tfw, data_fw_sat[:, 0, 0])
    plt.show()
    # plt.plot(tq, torch.flip(data_bw[:, 0, 1], (0, )))
    # plt.plot(tq, data_fw[:, 0, 1])
    plt.plot(torch.flip(tbw, (0,)), torch.flip(data_bw_sat[:, 0, 1], (0,)))
    plt.plot(tfw, data_fw_sat[:, 0, 1])
    plt.show()
    # plt.plot(tq, data_fw[:, 0, 2])
    # plt.plot(tq, data_fw[:, 0, 3])
    # plt.plot(tq, data_fw[:, 0, 4])
    plt.plot(tfw, data_fw_sat[:, 0, 2])
    plt.plot(tfw, data_fw_sat[:, 0, 3])
    plt.plot(tfw, data_fw_sat[:, 0, 4])
    plt.show()
    # d = obs.generate_data_svl(x_limits, wc_arr, num_samples,
    #                           method="LHS")
    # d, val_d = train_test_split(d, test_size=0.3, shuffle=True)

    # Generate training data and validation data
    data = observer.generate_data_svl(x_limits, wc_arr, num_samples, method="LHS")
    data, val_data = train_test_split(data, test_size=0.3, shuffle=True)

    ##########################################################################
    # Setup learner ##########################################################
    ##########################################################################

    # Define transformation function [T, T_star]
    transformation_function = "T_star"

    # Trainer options
    num_epochs = 100
    trainer_options = {"max_epochs": num_epochs}
    batch_size = 100
    init_learning_rate = 5e-3

    # Optim options
    optim_method = optim.Adam
    # optimizer_options = {"weight_decay": 1e-4}  # smoother transfo?
    optimizer_options = {}

    # Scheduler options
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

    # Instantiate learner
    learner_T_star = LearnerNoise(
        observer=observer,
        system=system,
        training_data=data,
        validation_data=val_data,
        method=transformation_function,
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

    # TODO
    # # Train the transformation function using the learner class
    # trainer.fit(learner_T_star)
    #
    # ##########################################################################
    # # Generate plots #########################################################
    # ##########################################################################
    #
    # learner_T_star.save_results(checkpoint_path=checkpoint_callback.best_model_path,)
    #
    # learner_T_star.save_plot(
    #     "Train_loss.pdf",
    #     "Training loss over time",
    #     "log",
    #     learner_T_star.train_loss.detach(),
    # )
    # learner_T_star.save_plot(
    #     "Val_loss.pdf",
    #     "Validation loss over time",
    #     "log",
    #     learner_T_star.val_loss.detach(),
    # )

    # Params
    nb = int(np.min([len(learner_T_star.training_data), 10000]))
    idx = np.random.choice(np.arange(len(learner_T_star.training_data)),
                           size=(nb,), replace=False)
    verbose = False
    x_limits = np.array([[-2.7, 2.7], [-2.7, 2.7]])

    learner_T_star.save_pdf_training(learner_T_star.training_data[idx], verbose)

    # path = "runs/SaturatedVanDerPol/Supervised_noise/T_star/exp_10_0.1-2_ok1" \
    #        "/"  # TODO
    # import dill as pkl
    # learner_path = path + "/learner.pkl"
    # with open(learner_path, "rb") as rb_file:
    #     learner_T_star = pkl.load(rb_file)
    # learner_T_star.results_folder = path
    # x_limits = np.array([[-2.7, 2.7], [-2.7, 2.7]])
    # wc_arr = np.linspace(0.1, 2., 10)
    # # wc_arr = np.array([0.6])
    # verbose = False

    # # Gain criterion
    # print('Computing our gain-tuning criterion can take some time but saves '
    #       'intermediary data in a subfolder zi_mesh: if you have already run '
    #       'this script, set save to False and path to this subfolder.')
    # save = True
    # path = ''
    # if save:
    #     mesh = learner_T_star.model.generate_data_svl(
    #         x_limits, wc_arr, 10000 * len(wc_arr), method="uniform", stack=False
    #     )
    # else:
    #     mesh = torch.randn((10, learner_T_star.model.dim_x +
    #                        learner_T_star.model.dim_z, 1))
    # learner_T_star.save_rmse_wc(mesh, wc_arr, verbose)
    # learner_T_star.plot_sensitiviy_wc(mesh, wc_arr, verbose, save=save,
    #                                   path=path)

    # x_mesh = torch.tensor([[1.0, 1.0]] * (len(wc_arr) + 1))  # TODO
    # print(wc_arr)
    # learner_T_star.save_random_traj(x_mesh, wc_arr, 1, verbose, (0, 20),
    #                                 1e-2, std=0.5)

    # # Trajectories
    # std_array = [0.0, 0.25, 0.5]
    # wc_arr = np.array([2., 0.1, 1.1556])
    # tsim = (0, 20)
    # dt = 1e-2
    # x_0 = torch.tensor([1.0, 1.0])
    #
    # learner_T_star.plot_rmse_error(x_0, wc_arr, verbose, tsim, dt, std=0.25)
    #
    # for std in std_array:
    #     learner_T_star.save_trj(
    #        x_0, wc_arr, 0, verbose, tsim, dt, var=std
    #     )
    #     learner_T_star.plot_traj_error(
    #         x_0, wc_arr, 0, verbose, tsim, dt, var=std
    #     )
    #
    # # Heatmap
    # mesh = learner_T_star.model.generate_data_svl(
    #     x_limits, wc_arr, 10000, method="uniform", stack=False
    # )
    #
    # # learner.plot_sensitiviy_wc(mesh, wc_arr, verbose)
    # learner_T_star.save_pdf_heatmap(mesh, verbose)
