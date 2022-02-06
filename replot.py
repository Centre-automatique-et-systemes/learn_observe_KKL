'''
Project: Oncycle
File: replot.py
File Created: Monday, 6th December 2021 10:09:18 am
Author: Lukas Bahr
-----
Copyright 2021 - 2022, Oncycle
'''

import sys

sys.path.append("../")
import dill as pkl
import numpy as np
import torch
import os

from numpy import genfromtxt
import os

import random as rd

from learn_KKL.learner import Learner

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from torch.nn import functional

from learn_KKL.utils import RMSE

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


def plot_traj_error_seperate_t_star():
    path_0_46 = "/Users/lukasbahr/master_thesis/learn_observe_KKL/runs/VanDerPol/Supervised/T_star/exp_0_46/Test_trajectories/Traj_0/"
    path_0_9 = "/Users/lukasbahr/master_thesis/learn_observe_KKL/runs/VanDerPol/Supervised/T_star/wc_0_9/Test_trajectories/Traj_0/"

    simulation_0_46 = np.genfromtxt(path_0_46 + "True_traj_0.csv", delimiter=",")
    simulation_0_9 = np.genfromtxt(path_0_9 + "True_traj_0.csv", delimiter=",")

    measurement_0_46 = np.genfromtxt(path_0_46 + "Estimated_traj_0.csv", delimiter=",")
    measurement_0_9 = np.genfromtxt(path_0_9 + "Estimated_traj_0.csv", delimiter=",")

    wc_arr = np.array([0.9, 0.46])
    # wc_arr = np.array([0.46, 0.9])
    tq_ = simulation_0_46[:, 0]
    tq = np.arange(0, 20, 1e-2)

    simulation = np.concatenate(
        (
            np.expand_dims(simulation_0_9[:, 1:], axis=0),
            np.expand_dims(simulation_0_46[:, 1:], axis=0),
        )
    )
    estimation = np.concatenate(
        (
            np.expand_dims(measurement_0_9[:, 1:], axis=0),
            np.expand_dims(measurement_0_46[:, 1:], axis=0),
        )
    )

    simulation = torch.from_numpy(simulation)
    estimation = torch.from_numpy(estimation)

    results_folder = "/Users/lukasbahr/master_thesis/learn_observe_KKL/runs/VanDerPol/Supervised/T_star/"

    # plot_traj_error(tq, simulation, estimation, wc_arr, 0, verbose, results_folder)
    plot_rsme_error(tq, simulation, estimation ,verbose, results_folder, wc_arr)

def plot_rsme_error(tq, simulation, estimation ,verbose, results_folder, w_c_arr):
    traj_folder = os.path.join(results_folder, "Test_trajectories_error")

    # Save these test trajectories
    os.makedirs(traj_folder, exist_ok=True)
    traj_error = 0.0

    plot_style = ["-", "--", "-."]

    for i in range(2):
        error = RMSE(simulation[i, :, :], estimation[i, :, :])
        traj_error += error

        current_traj_folder = os.path.join(traj_folder, f"Traj_{i}")
        os.makedirs(current_traj_folder, exist_ok=True)

        filename = f"RMSE_{i}.txt"
        with open(os.path.join(current_traj_folder, filename), "w") as f:
            print(error.cpu().numpy(), file=f)

        # for i in range(simulation.shape[1]):
        name = "Traj" + str(1) + ".pdf"
        plt.plot(
            tq,
            RMSE(estimation[i, :, :], simulation[i, :, :], 1).cpu().numpy(),
            plot_style[i],
            linewidth=0.8,
            markersize=1,
            label=rf"$\omega_c = {round(float(w_c_arr[i]), 2)}$",
        )

    plt.legend(loc=1)
    plt.grid(visible=True)
    plt.title(rf"Test trajectory RMSE error")
    plt.xlabel(rf"$t$")
    plt.ylabel(rf"$\hat{{x}}-x$")
    plt.savefig(os.path.join(current_traj_folder, name), bbox_inches="tight")
    if verbose:
        plt.show()
    plt.clf()
    plt.close("all")

    filename = "RMSE_traj.txt"
    with open(os.path.join(traj_folder, filename), "w") as f:
        print(traj_error, file=f)


def plot_traj_error(
    tq, simulation, estimation, w_c_arr, nb_trajs, verbose, results_folder
):
    # Estimation over the test trajectories with T_star

    nb_trajs += w_c_arr.shape[0]

    traj_folder = os.path.join(results_folder, "Test_trajectories_error")

    # Save these test trajectories
    os.makedirs(traj_folder, exist_ok=True)
    traj_error = 0.0

    plot_style = ["-", "--", "-."]

    for i in range(nb_trajs):
        # TODO run predictions in parallel for all test trajectories!!!
        # Need to figure out how to interpolate y in parallel for all
        # trajectories!!!

        if i < len(w_c_arr):
            w_c = w_c_arr[i]
        else:
            w_c = rd.uniform(0.2, 1.5)

        error = RMSE(simulation, estimation)
        traj_error += error

        current_traj_folder = os.path.join(traj_folder, f"Traj_{i}")
        os.makedirs(current_traj_folder, exist_ok=True)

        filename = f"RMSE_{i}.txt"
        with open(os.path.join(current_traj_folder, filename), "w") as f:
            print(error.cpu().numpy(), file=f)

        # for i in range(simulation.shape[1]):
        name = "Traj" + str(1) + ".pdf"
        plt.plot(
            tq,
            estimation[i, :, 1].cpu().numpy() - simulation[i, :, 1].cpu().numpy(),
            plot_style[i],
            linewidth=0.8,
            markersize=1,
            label=rf"$\omega_c = {round(float(w_c_arr[i]), 2)}$",
        )

    plt.legend(loc=1)
    plt.grid(visible=True)
    plt.title(rf"Test trajectory error")
    plt.xlabel(rf"$t$")
    plt.ylabel(rf"$\hat{{x}}_{ 2}-x_{2}$")
    plt.savefig(os.path.join(current_traj_folder, name), bbox_inches="tight")
    if verbose:
        plt.show()
    plt.clf()
    plt.close("all")

    filename = "RMSE_traj.txt"
    with open(os.path.join(traj_folder, filename), "w") as f:
        print(traj_error, file=f)


if __name__ == "__main__":
    # Load learner from learner.pkl
    # path = ['0.1', '0.14677993', '0.21544347', '0.31622777', '0.46415888', '0.68129207', '1.']
    path = "runs/VanDerPol/Supervised/T_star/exp_0_46"
    # path = "runs/RevDuffing/Supervised/T_star/211220_100_wc/"

    # for i in path:
    learner_path = path + "/learner.pkl"

    with open(learner_path, "rb") as rb_file:
        learner = pkl.load(rb_file)

    # learner.results_folder = "runs/RevDuffing/Supervised/T_star/211220_100_wc/"
    learner.results_folder = path

    limits = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    num_samples = 70000
    # wc_arr = np.array([0.9090909090909091, 0.4636363636363636])
    # wc_arr = np.logspace(-1.5,0,100)
    # wc_arr = np.array([1.0, 0.03162277660168379, 0.11103363181676379])
    # wc_arr = np.array([ 0.03162277660168379, 0.11103363181676379, 1.0])
    # wc_arr = np.linspace(0.1,1.,100)
    wc_arr = np.array([0.11])
    # # wc_arr = np.array([float(i)])
    # mesh = learner.model.generate_data_svl(limits, wc_arr, num_samples, method='uniform', stack=False)
    # learner.save_pdf_heatmap(mesh, verbose)

    # mesh = learner.model.generate_data_svl(limits, wc_arr, 2000*100, method='LHS', stack=False)
    # learner.save_rmse_wc(mesh, wc_arr, verbose)
    # learner.plot_sensitiviy_wc(mesh, wc_arr, verbose)
    # learner.save_trj(torch.tensor([1., 1.]), wc_arr, 1, verbose, (0, 50), 1e-2, var=0.5)
    # wc_arr = np.logspace(-1.5,0,10)
    # limits = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    # learner.save_plot(
    #     "Train_loss.pdf",
    #     "Training loss over time",
    #     "log",
    #     learner.train_loss.detach(),
    # )
    # learner.save_plot(
    #     "Val_loss.pdf", "Validation loss over time", "log", learner.val_loss.detach()
    # )

    # idx = np.random.choice(np.arange(len(learner.training_data)), size=(10000,))
    verbose = False

    # # learner.save_pdf_training(learner.training_data[idx], verbose)
    # learner.save_trj(
    #     torch.tensor([0.225, -0.131]), wc_arr, 0, verbose, (0, 70), 1e-2, var=0.5
    # )
    # learner.save_trj(
    # torch.tensor([1.0, 1.0]), wc_arr, 0, verbose, (0, 20), 1e-2, var=0.5
    # )
    # learner.plot_traj_error(torch.tensor([1.0, 1.0]), wc_arr, 0, verbose, (0, 50), 1e-2, var=0.5)
    # learner.save_pdf_heatmap(mesh, verbose)

    # mesh = learner.model.generate_data_svl(
    # limits, wc_arr, 200000, method="LHS", stack=False
    # )
    # learner.save_rmse_wc(mesh, wc_arr, verbose)
    # learner.plot_sensitiviy_wc(mesh, wc_arr, verbose)

    plot_traj_error_seperate_t_star()