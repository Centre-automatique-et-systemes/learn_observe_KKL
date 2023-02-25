from pathlib import Path

import matplotlib.pyplot as plt
import os

import scipy.linalg
import torch
import pandas as pd
import numpy as np
import dill as pkl
import seaborn as sb
from functorch import vmap, jacfwd, jacrev

from learn_KKL.utils import compute_h_infinity, RMSE
from learn_KKL.filter_utils import interpolate_func
from learn_KKL.system import QuanserQubeServo2

# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
# https://jwalton.info/Matplotlib-latex-PGF/
# https://stackoverflow.com/questions/12322738/how-do-i-change-the-axis-tick-font-in-a-matplotlib-plot-when-rendering-using-lat
plt.rcdefaults()
# For manuscript
sb.set_style('whitegrid')
plot_params = {
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.serif': 'Palatino',
    'font.size': 16,
    "pgf.preamble": "\n".join([
        r'\usepackage{bm}',
    ]),
    'text.latex.preamble': [r'\usepackage{amsmath}',
                            r'\usepackage{amssymb}',
                            r'\usepackage{cmbright}'],
}
plt.rcParams.update(plot_params)
# # Previous papers
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{cmbright}')
# plt.rc('font', family='serif')
# plt.rcParams.update({'font.size': 22})
# sb.set_style('whitegrid')

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Script to evaluate the gain-tuning criterion for the Qube, given a series
# of experiments for individual values of wc rather than a transformation
# that directly depends on wc as usual for experiments_noise

def sensitivity_norm(model, x, z, wc, save=True, path='', version=9):
    print('Python version of our gain-tuning criterion: the estimation of '
          'the H-infinity norm is not very smooth and we have replaced '
          'the H-2 norm by a second H-infinity norm, hence, the Matlab '
          'script criterion.m should be used instead to compute the final '
          'criterion as it was in the paper.')
    if save:
        with torch.no_grad():
            # Compute dTdx over grid
            dTdx = vmap(jacfwd(model.encoder))(x)
            dTdx = dTdx[:, :, : model.dim_x]
            idx_max = torch.argmax(torch.linalg.matrix_norm(dTdx, ord=2))
            Tmax = dTdx[idx_max]

            # Compute dTstar_dz over grid
            dTstar_dz = vmap(jacrev(model.decoder))(z)
            dTstar_dz = dTstar_dz[:, :, : model.dim_z]
            idxstar_max = torch.argmax(
                torch.linalg.matrix_norm(dTstar_dz, ord=2))
            Tstar_max = dTstar_dz[idxstar_max]

        # Save this data
        file = pd.DataFrame(Tmax)
        file.to_csv(os.path.join(path, f'Tmax_wc{wc:0.2g}.csv'),
                    header=False)
        file = pd.DataFrame(Tstar_max)
        file.to_csv(os.path.join(path, f'Tstar_max_wc{wc:0.2g}.csv'),
                    header=False)
        file = pd.DataFrame(dTdx.flatten(1, -1))
        file.to_csv(os.path.join(path, f'dTdx_wc{wc:0.2g}.csv'),
                    header=False)
        file = pd.DataFrame(dTstar_dz.flatten(1, -1))
        file.to_csv(os.path.join(path, f'dTstar_dz_wc{wc:0.2g}.csv'),
                    header=False)
    else:
        # Load intermediary data
        df = pd.read_csv(os.path.join(path, f'Tmax_wc{wc:0.2g}.csv'),
                         sep=',', header=None)
        Tmax = torch.from_numpy(df.drop(df.columns[0], axis=1).values)
        df = pd.read_csv(os.path.join(path, f'dTdx_wc{wc:0.2g}.csv'),
                         sep=',', header=None)
        dTdx = torch.from_numpy(
            df.drop(df.columns[0], axis=1).values).reshape(
            (-1, Tmax.shape[0], Tmax.shape[1]))
        df = pd.read_csv(os.path.join(path, f'Tstar_max_wc{wc:0.2g}.csv'),
                         sep=',', header=None)
        Tstar_max = torch.from_numpy(df.drop(df.columns[0], axis=1).values)
        df = pd.read_csv(os.path.join(path, f'dTstar_dz_wc{wc:0.2g}.csv'),
                         sep=',', header=None)
        dTstar_dz = torch.from_numpy(
            df.drop(df.columns[0], axis=1).values).reshape(
            (-1, Tstar_max.shape[0], Tstar_max.shape[1]))

    if version == 1:
        C = np.eye(model.dim_z)
        sv = torch.tensor(
            compute_h_infinity(model.D.numpy(), model.F.numpy(), C, 1e-10))
        product = torch.linalg.matrix_norm(Tstar_max, ord=2) * sv
        return torch.cat(
            (torch.linalg.matrix_norm(Tstar_max, ord=2).unsqueeze(0),
             sv.unsqueeze(0), product.unsqueeze(0)), dim=0
        )
    elif version == 2:
        C = np.eye(model.dim_z)
        sv = torch.tensor(compute_h_infinity(
            model.D.numpy(), model.F.numpy(),
            np.dot(Tstar_max.detach().numpy(),
                   C), 1e-3))
        product = sv
        return torch.cat(
            (torch.linalg.matrix_norm(Tstar_max, ord=2).unsqueeze(0),
             sv.unsqueeze(0), product.unsqueeze(0)), dim=0
        )
    elif version == 3:
        C = np.eye(model.dim_x)
        sv = torch.tensor(compute_h_infinity(
            np.dot(np.dot(Tstar_max.detach().numpy(),
                          model.D.numpy()), Tmax.numpy()),
            np.dot(Tstar_max.detach().numpy(), model.F.numpy()),
            C, 1e-3))
        return torch.cat(
            (torch.linalg.matrix_norm(Tmax, ord=2).unsqueeze(0),
             torch.linalg.matrix_norm(Tstar_max, ord=2).unsqueeze(0),
             sv.unsqueeze(0)), dim=0
        )
    elif version == 9:
        # TODO right norms are implemented but results less smooth than Matlab!
        C = np.eye(model.dim_z)
        l2_norm = torch.linalg.norm(
            torch.linalg.matrix_norm(dTstar_dz, dim=(1, 2), ord=2))
        sv1 = torch.tensor(
                    compute_h_infinity(model.D.numpy(), model.F.numpy(), C, 1e-10))
        # sv2 = torch.tensor(
        #     compute_h_infinity(model.D.numpy(), np.eye(model.dim_z), C,
        #                        1e-10))
        # obs_sys = control.matlab.ss(model.D.numpy(), model.F.numpy(), C, 0)
        # sv1 = control.h2syn()
        # A1 = model.D.numpy()
        # Q1 = - model.F.numpy() @ model.F.numpy().T
        # P1 = scipy.linalg.solve_continuous_lyapunov(A1, Q1)
        # sv1 = torch.as_tensor((C @ P1 @ C.T).trace())
        A2 = model.D.numpy()
        Q2 = - np.eye(model.dim_z)
        P2 = scipy.linalg.solve_continuous_lyapunov(A2, Q2)
        sv2 = torch.as_tensor((C @ P2 @ C.T).trace())
        product = l2_norm * (sv1 + sv2)
        return torch.cat(
            (l2_norm.unsqueeze(0), sv1.unsqueeze(0),
             product.unsqueeze(0)), dim=0
        )
    else:
        raise NotImplementedError(f'Gain-tuning criterion version '
                                  f'{version} is not implemented.')

# Loop over values of wc: load individual model for each
def plot_sensitiviy_wc(exp_folder, exp_subfolders, verbose, dim_z, save=True,
                       path=''):
    errors = np.zeros((len(exp_subfolders), 3))
    if path == '':
        path = os.path.join(exp_folder, 'xzi_mesh')
        os.makedirs(path, exist_ok=True)

    w_c_array = []
    D_arr = torch.zeros(0, dim_z, dim_z)
    for j in range(len(exp_subfolders)):
        # Load model
        learner_path = os.path.join(exp_folder, exp_subfolders[j],
                                    "learner.pkl")
        with open(learner_path, "rb") as rb_file:
            learner = pkl.load(rb_file)
        learner.results_folder = exp_folder

        # Retrieve params
        dt = 0.04
        data_tsim = (0, 4)  # for generating test data
        traj_data = learner.traj_data
        x_limits = learner.x0_limits
        wc = learner.model.wc
        w_c_array.append(wc)
        print(wc, learner.model.D)

        # Generate mesh on which to compute gradients, including wc
        if traj_data:
            mesh = learner.model.generate_trajectory_data(
                x_limits, 500, method="LHS", tsim=data_tsim,
                stack=True, dt=dt
            )
        else:
            mesh = learner.model.generate_data_svl(
                x_limits, 10000, method="LHS")

        if save:
            x_mesh = mesh[:, learner.x_idx_in]
            z_mesh = mesh[:, learner.z_idx_in]

            file = pd.DataFrame(mesh)
            file.to_csv(os.path.join(
                path, f'xzi_data_wc{wc:0.2g}.csv'), header=False)
            for i in learner.x_idx_out[:-1] + learner.z_idx_out[:-1]:
                plt.scatter(mesh[:, i], mesh[:, i + 1])
                plt.savefig(os.path.join(
                    path, f'xzi_data_wc{wc:0.2g}_{i}.pdf'),
                    bbox_inches='tight')
                plt.xlabel(rf'$x_{i}$')
                plt.xlabel(rf'$x_{i + 1}$')
                plt.clf()
                plt.close('all')
        else:
            # Load mesh that was saved in path
            df = pd.read_csv(os.path.join(
                path, f'xzi_data_wc{wc:0.2g}.csv'), sep=',',
                header=None)
            mesh = torch.from_numpy(df.drop(df.columns[0], axis=1).values)
            x_mesh = mesh[:, learner.x_idx_in]
            z_mesh = mesh[:, learner.z_idx_in]

        errors[j] = sensitivity_norm(learner.model, x_mesh, z_mesh, wc,
                                     save=save, path=path)
        D_arr = torch.cat((D_arr, torch.unsqueeze(learner.model.D, 0)))

    w_c_array = torch.as_tensor(w_c_array)
    # errors = functional.normalize(errors)
    learner.save_csv(
        D_arr.flatten(1, -1),
        os.path.join(path, "D_arr.csv"),
    )
    learner.save_csv(
        w_c_array,
        os.path.join(path, "w_c_array.csv"),
    )
    learner.save_csv(
        np.concatenate((np.expand_dims(w_c_array, 1), errors), axis=1),
        os.path.join(path, "sensitivity.csv"),
    )
    name = "sensitivity_wc.pdf"
    blue = "tab:blue"
    orange = "tab:orange"
    green = "tab:green"

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(r"$\omega_c$")
    # ax1.set_ylabel('exp', color=color)
    ax1.tick_params(axis='y')
    # line_1 = ax1.plot(
    #     w_c_array,
    #     errors[:, 0],
    #     label=r"$\frac{1}{N}\max_{z_i} \left| \frac{\partial \mathcal{T}^*}{\partial z} (z_i) \right|_{l^2}$",
    #     color=blue
    # )
    # line_2 = ax1.plot(w_c_array, errors[:, 1], label=r"$\left| G \right|_\infty$", color=green)

    ax2 = ax1.twinx()

    N = len(mesh)
    line_3 = ax1.plot(
        w_c_array,
        errors[:, -1] / N,
        label=r"$\frac{\alpha(\omega_c)}{n}$",
        color=orange,
    )
    ax2.tick_params(axis='y')
    fig.tight_layout()

    # added these three lines
    # lns = line_3 + line_1 + line_2
    lns = line_3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=1)
    ax1.grid(False)

    # plt.title("Gain tuning criterion")

    plt.savefig(os.path.join(learner.results_folder, name),
                bbox_inches="tight")

    if verbose:
        plt.show()

    plt.clf()
    plt.close("all")

# Use crit.csv from Matlab: utility function to plot criterion from Matlab
def plot_crit(folder, N, verbose=False):
    # Read file
    df = pd.read_csv(os.path.join(folder, 'crit.csv'), sep=',', header=None)
    errors = df.values
    w_c_array = errors[:, 0]

    plt.plot(
        w_c_array,
        errors[:, -1] / N,
        label=r"$\frac{\alpha(\omega_c)}{n}$",
        color='orange',
    )
    plt.xlabel(r'$\omega_c$')
    # plt.title("Gain tuning criterion")
    plt.legend()
    plt.savefig(os.path.join(folder, 'crit.pdf'), bbox_inches="tight")
    if verbose:
        plt.show()
    plt.clf()
    plt.close("all")

# Save test trajectories for different values of noise std and wc
def test_trajs(exp_folder, exp_subfolders, test_array, std_array, x0,
               verbose=False, true_traj=None, true_traj_compare=False):
    with torch.no_grad():
        for std in std_array:
            std_folder = os.path.join(exp_folder, f'Test_trajs_{std}')
            os.makedirs(std_folder, exist_ok=True)
            for j in test_array:
                # Load model
                learner_path = os.path.join(exp_folder, exp_subfolders[j],
                                            "learner.pkl")
                with open(learner_path, "rb") as rb_file:
                    learner = pkl.load(rb_file)
                # Retrieve params
                dt = 0.04
                tsim = (0, 8)  # for generating test data
                wc = learner.model.wc
                traj_folder = os.path.join(std_folder, f'wc{wc:0.2g}')

                # learner.save_trj(init_state=x0, verbose=verbose, tsim=tsim,
                #                  dt=dt, std=std, traj_folder=traj_folder,
                #                  z_0=None)
                if traj_folder is None:
                    traj_folder = os.path.join(learner.results_folder,
                                               f"Test_trajectories/Traj_{std}")
                if true_traj is None:
                    tq, simulation = learner.system.simulate(x0, tsim, dt)
                elif true_traj_compare == True:
                    simulation = true_traj
                    tq, simu = learner.system.simulate(x0, tsim, dt)
                    meas = learner.model.h(simu)
                else:
                    simulation = true_traj
                    tq = torch.arange(tsim[0], tsim[1], dt)
                measurement = learner.model.h(simulation)
                noise = torch.normal(0, std, size=measurement.shape)
                measurement = measurement.add(noise)
                # Save these test trajectories
                os.makedirs(traj_folder, exist_ok=True)
                traj_error = 0.0

                # TODO run predictions in parallel for all test trajectories!!!
                # Need to figure out how to interpolate y in parallel for all
                # trajectories!!!
                y = torch.cat((tq.unsqueeze(1), measurement), dim=-1)
                estimation, z = learner.model.predict(
                    y, tsim, dt, out_z=True, z_0=None)
                if true_traj_compare:
                    y_simu = torch.cat((tq.unsqueeze(1), meas), dim=-1)
                    estim, z_simu = learner.model.predict(
                        y_simu, tsim, dt, out_z=True, z_0=None)
                error = RMSE(simulation, estimation)
                traj_error += error
                filename = f"RMSE.txt"
                with open(os.path.join(traj_folder, filename), "w") as f:
                    print(error.cpu().numpy(), file=f)

                learner.save_csv(
                    simulation.cpu().numpy(),
                    os.path.join(traj_folder, f"True_traj.csv"),
                )
                learner.save_csv(
                    estimation.cpu().numpy(),
                    os.path.join(traj_folder, f"Estimated_traj.csv"),
                )

                if true_traj_compare:
                    name = "Meas.pdf"
                    plt.plot(tq, measurement.detach().numpy(),
                             label='Experiment')
                    plt.plot(tq, meas.detach().numpy(), '--',
                             label='Simulation')
                    plt.legend(loc=1)
                    plt.grid(visible=True)
                    plt.title('Measurement')
                    plt.xlabel(rf"$t$")
                    plt.ylabel(rf"$y$")
                    plt.savefig(
                        os.path.join(traj_folder, name), bbox_inches="tight"
                    )
                    if verbose:
                        plt.show()
                    plt.clf()
                    plt.close("all")
                    for j in range(z.shape[1]):
                        name = "Traj_z" + str(j) + ".pdf"
                        plt.plot(tq, z[:, j].detach().numpy(),
                                 label=rf"Experiment")
                        plt.plot(tq, z_simu[:, j].detach().numpy(), '--',
                                 label=rf"Simulation")
                        plt.legend(loc=1)
                        plt.grid(visible=True)
                        # plt.title('Observer state over test trajectory')
                        plt.xlabel(rf"$t$")
                        plt.ylabel(rf"$z_{j + 1}$")
                        plt.savefig(
                            os.path.join(traj_folder, name), bbox_inches="tight"
                        )
                        if verbose:
                            plt.show()
                        plt.clf()
                        plt.close("all")

                for j in range(estimation.shape[1]):
                    name = "Traj" + str(j) + ".pdf"
                    if j == 0:
                        plt.plot(tq, measurement.detach().numpy(), '-',
                                 label=r"$y$")
                    plt.plot(tq, simulation[:, j].detach().numpy(), '--',
                             label=rf"$x_{j + 1}$")
                    plt.plot(
                        tq, estimation[:, j].detach().numpy(), '-.',
                        label=rf"$\hat{{x}}_{j + 1}$"
                    )
                    plt.legend(loc=1)
                    plt.grid(visible=True)
                    # plt.title(
                    #     rf"Test trajectory, RMSE = {np.round(error.numpy(), 4)}")
                    plt.xlabel(rf"$t$")
                    plt.ylabel(rf"$x_{j + 1}$")
                    plt.savefig(
                        os.path.join(traj_folder, name), bbox_inches="tight"
                    )
                    if verbose:
                        plt.show()
                    plt.clf()
                    plt.close("all")

                filename = "RMSE_traj.txt"
                with open(os.path.join(traj_folder, filename), "w") as f:
                    print(traj_error, file=f)

# Compute test trajectories for different values of noise std and wc and save
# corresponding error plots
def error_trajs(exp_folder, exp_subfolders, test_array, std_array, x0,
               verbose=False, true_traj=None):
    with torch.no_grad():
        for std in std_array:
            traj_folder = os.path.join(exp_folder, f'Test_trajs_{std}')
            os.makedirs(traj_folder, exist_ok=True)
            plot_style = ["-", "--", "-."]
            for j in range(len(test_array)):
                # Load model
                test_idx = test_array[j]
                learner_path = os.path.join(exp_folder, exp_subfolders[test_idx],
                                            "learner.pkl")
                with open(learner_path, "rb") as rb_file:
                    learner = pkl.load(rb_file)
                # Retrieve params
                dt = 0.04
                tsim = (0, 8)  # for generating test data
                wc = learner.model.wc

                # Compute error
                if j == 0:
                    if true_traj is None:
                        tq, simulation = learner.system.simulate(x0, tsim, dt)
                    else:
                        simulation = true_traj
                        tq = torch.arange(tsim[0], tsim[1], dt)
                    measurement = learner.model.h(simulation)
                    noise = torch.normal(0, std, size=measurement.shape)
                    measurement = measurement.add(noise)
                    y = torch.cat((tq.unsqueeze(1), measurement), dim=-1)
                estimation = learner.model.predict(
                    y, tsim, dt, z_0=None).detach()

                whole_error = RMSE(simulation, estimation)
                filename = f"RMSE_wc{wc:0.2g}.txt"
                with open(os.path.join(traj_folder, filename),
                          "w") as f:
                    print(whole_error.cpu().numpy(), file=f)

                error = RMSE(simulation, estimation, dim=-1)
                learner.save_csv(
                    error,
                    os.path.join(traj_folder, f"Error_traj_wc{wc:0.2g}.csv"),
                )
                name = "RMSE_traj.pdf"
                # name = "Error_traj.pdf"
                plt.plot(
                    tq,
                    error,
                    # torch.sum(estimation-simulation, dim=-1),
                    plot_style[j],
                    linewidth=0.8,
                    markersize=1,
                    label=rf"$\omega_c = {float(wc):0.2g}$",
                )
            plt.legend(loc=1)
            plt.grid(visible=True)
            # plt.title(rf"RMSE over test trajectory")
            plt.xlabel(rf"$t$")
            plt.ylabel(r'$|\hat{x}-x|$')
            # plt.ylabel(rf"$\hat{{x}}-x$")
            plt.savefig(os.path.join(traj_folder, name),
                        bbox_inches="tight")
            if verbose:
                plt.show()
            plt.clf()
            plt.close("all")


if __name__ == "__main__":
    # Enter folder from which to extract the experiments
    ROOT = Path(__file__).parent.parent
    MEASUREMENT = 1  # Either 1, 2, 12
    EXP_FOLDER = ROOT / 'runs' / f'QuanserQubeServo2_meas{MEASUREMENT}' / \
                 'Supervised' / 'T_star' / 'N5000_wc1541'

    # Retrieve list of wc values used in this experiment
    subdirs = [f for f in os.listdir(EXP_FOLDER) if f.startswith('exp')]
    subdirs.sort(key=lambda dir: int(dir.split('exp_')[1]))
    # Load one model to retrieve params
    learner_path = EXP_FOLDER / subdirs[0] / 'learner.pkl'
    with open(learner_path, "rb") as rb_file:
        learner_T_star = pkl.load(rb_file)
    dim_z = learner_T_star.model.dim_z

    # Compute gradients for each wc
    verbose = False
    print('Computing our gain-tuning criterion can take some time but saves '
          'intermediary data in a subfolder xzi_mesh: if you have already run '
          'this script, set save to False and path to that subfolder.')
    save = True
    path = ''
    # plot_sensitiviy_wc(exp_folder=EXP_FOLDER, exp_subfolders=subdirs,
    #                    dim_z=dim_z, verbose=verbose, save=save, path=path)
    plot_crit(os.path.join(EXP_FOLDER, 'xzi_mesh'), N=50000, verbose=verbose)

    # Experimental test traj
    dt_exp = 0.004
    dt = 0.04
    tsim = (0, 8)  # for generating test data
    fileName = 'example_csv_fin4'
    filepath = 'Data/QQS2_data_diffx0/' + fileName + '.csv'
    exp_data = np.genfromtxt(filepath, delimiter=',')
    tq_exp = torch.from_numpy(exp_data[1:2001, -1] - exp_data[1, -1])
    exp_data = exp_data[1:2001, 1:-1]
    system = QuanserQubeServo2()
    exp_data = torch.from_numpy(system.remap_hardware(exp_data))
    t_exp = torch.cat((tq_exp.unsqueeze(1), exp_data), dim=1)
    exp_func = interpolate_func(x=t_exp, t0=tq_exp[0], init_value=exp_data[0])
    tq = torch.arange(tsim[0], tsim[1], dt)
    exp = exp_func(tq)

    # # Test trajectories
    std_array = [0.0, 0.025, 0.05]
    # test_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40])
    test_array = np.array([0, 9, 40])
    # x_0 = torch.tensor([0.1, 0.1, 0., 0.])
    x_0 = torch.tensor(
        [0.0337475773334841, 0.5062136600022615, 1.0925699983931834,
         1.7034324073882772])
    test_trajs(exp_folder=EXP_FOLDER, exp_subfolders=subdirs,
               test_array=test_array, std_array=std_array, x0=x_0,
               verbose=verbose)  # simu test traj
    # test_trajs(exp_folder=EXP_FOLDER, exp_subfolders=subdirs,
    #            test_array=test_array, std_array=std_array, x0=x_0,
    #            verbose=verbose, true_traj=exp)  # exp test traj
    # test_trajs(exp_folder=EXP_FOLDER, exp_subfolders=subdirs,
    #            test_array=test_array, std_array=std_array, x0=x_0,
    #            verbose=verbose, true_traj=exp,
    #            true_traj_compare=True)  # compare simu and exp test traj

    # Error trajectories
    std_array = [0.0, 0.025, 0.05]
    test_array = np.array([0, 9, 40])
    # x_0 = torch.tensor([0.1, 0.1, 0., 0.])
    x_0 = torch.tensor(
        [0.0337475773334841, 0.5062136600022615, 1.0925699983931834,
         1.7034324073882772])
    error_trajs(exp_folder=EXP_FOLDER, exp_subfolders=subdirs,
               test_array=test_array, std_array=std_array, x0=x_0,
               verbose=verbose)  # simu test traj
    # error_trajs(exp_folder=EXP_FOLDER, exp_subfolders=subdirs,
    #             test_array=test_array, std_array=std_array, x0=x_0,
    #             verbose=verbose, true_traj=exp)  # exp test traj
