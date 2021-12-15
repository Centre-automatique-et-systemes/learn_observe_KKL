import sys ; sys.path.append('../')
import dill as pkl
import numpy as np
import torch

if __name__ == "__main__":
    # Load learner from learner.pkl
    # path = ['0.1', '0.14677993', '0.21544347', '0.31622777', '0.46415888', '0.68129207', '1.']

    # for i in path:
    learner_path = 'runs/VanDerPol/Supervised/T_star/0.01930698/learner.pkl'
    
    with open(learner_path, 'rb') as rb_file:
        learner = pkl.load(rb_file)

    learner.results_folder = 'runs/VanDerPol/Supervised/T_star/0.01930698/'



    limits = np.array([[-1, 1.], [-1., 1.]])
    num_samples = 70000
    wc_arr = np.array([0.01930698])
    # wc_arr = np.array([float(i)])
    mesh = learner.model.generate_data_svl(limits, wc_arr, num_samples,
                                                method='uniform', stack=False)
    verbose = False
    learner.save_pdf_heatmap(mesh, verbose)

    # mesh = learner.model.generate_data_svl(limits, wc_arr, 10000,
                                            # method='LHS', stack=False)
    # learner.save_rmse_wc(mesh, wc_arr, verbose)
    # learner.plot_sensitiviy_wc(mesh, wc_arr, verbose)
    # learner.save_trj(torch.tensor([1., 1.]), wc_arr, 1, verbose, (0, 50), 1e-2, var=0.5)