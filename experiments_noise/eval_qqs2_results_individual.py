from pathlib import Path
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import numpy as np
import dill as pkl
import seaborn as sb
from functorch import vmap, jacfwd, jacrev

from learn_KKL.utils import compute_h_infinity

# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
# https://jwalton.info/Matplotlib-latex-PGF/
plot_params = {
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'axes.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.labelsize': 16,
    "pgf.preamble": "\n".join([
        r'\usepackage{bm}',
    ]),
    'text.latex.preamble': [r'\usepackage{amsmath}',
                            r'\usepackage{amssymb}',
                            r'\usepackage{cmbright}'],
}

sb.set_style("whitegrid")

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
        C = np.eye(model.dim_z)
        l2_norm = torch.linalg.norm(
            torch.linalg.matrix_norm(dTstar_dz, dim=(1, 2), ord=2))
        sv1 = torch.tensor(
            compute_h_infinity(model.D.numpy(), model.F.numpy(), C, 1e-10))
        sv2 = torch.tensor(  # TODO implement H2 norm instead!
            compute_h_infinity(model.D.numpy(), np.eye(model.dim_z), C,
                               1e-10))
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

    plt.title("Gain tuning criterion")

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
    plt.title("Gain tuning criterion")
    plt.legend()
    plt.savefig(os.path.join(folder, 'crit.pdf'), bbox_inches="tight")
    if verbose:
        plt.show()
    plt.clf()
    plt.close("all")

# Save test trajectories for different values of noise std and wc
def test_trajs(exp_folder, exp_subfolders, test_array, std_array, x0,
               verbose=False):
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

            learner.save_trj(init_state=x0, verbose=verbose, tsim=tsim, dt=dt,
                             std=std, traj_folder=traj_folder, z_0=None)


if __name__ == "__main__":
    # Enter folder from which to extract the experiments
    ROOT = Path(__file__).parent.parent
    MEASUREMENT = 2  # Either 1, 2, 12
    EXP_FOLDER = ROOT / 'runs' / f'QuanserQubeServo2_meas{MEASUREMENT}' / \
                 'Supervised' / 'T_star' / 'N5000_wc1540'

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
          'intermediary data in a subfolder zi_mesh: if you have already run '
          'this script, set save to False and path to that subfolder.')
    save = True
    path = ''
    # plot_sensitiviy_wc(exp_folder=EXP_FOLDER, exp_subfolders=subdirs,
    #                    dim_z=dim_z, verbose=verbose, save=save, path=path)
    plot_crit(os.path.join(EXP_FOLDER, 'xzi_mesh'), N=50000, verbose=verbose)

    # Test trajectories
    std_array = [0.0, 0.025, 0.05]
    test_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40])
    x_0 = torch.tensor([0.1, 0.1, 0., 0.])
    test_trajs(exp_folder=EXP_FOLDER, exp_subfolders=subdirs,
               test_array=test_array, std_array=std_array, x0=x_0,
               verbose=verbose)
