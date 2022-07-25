# imports
import sys ; sys.path.append('../')
import torch.optim as optim
import torch
import seaborn as sb
import pytorch_lightning as pl
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from learn_KKL.luenberger_observer import LuenbergerObserver
from learn_KKL.system import RevDuffing,VanDerPol
from learn_KKL.learner import Learner
from learn_KKL.raffinement import *
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Set up the system
    system = RevDuffing()

    # Instantiate the observer
    observer = LuenbergerObserver(dim_x=2, dim_y=1, method='Supervised', recon_lambda=0.8, wc=0.03,
                                  activation=torch.nn.SiLU())
    observer.set_dynamics(system)
    # Generate (x_i, z_i) data by running system backward, then system + observer
    # forward in time
    data,grid = observer.generate_data_svl(np.array([[-1, 1.], [-1., 1.]]), [50,50],
                                      method='uniform')
    data, val_data = train_test_split(data, test_size=0.3, shuffle=True)


    # affichage
    print(data.shape)
    im,ax = plt.subplots(1,3,figsize=(15,15))
    for i in range(3):
        ax[i].scatter(data[:,0],data[:,1],c = data[:,2+i],s=1,cmap='jet')
        ax[i].axis('square')
    plt.show()

    # # Train the forward transformation using pytorch-lightning and the learner class
    # # Options for training
    # trainer_options = {'max_epochs': 15}
    # optimizer_options = {'weight_decay': 1e-6}
    # scheduler_options = {'mode': 'min', 'factor': 0.1, 'patience': 3,
    #                      'threshold': 1e-4, 'verbose': True}
    # stopper = pl.callbacks.early_stopping.EarlyStopping(
    #     monitor='val_loss', min_delta=5e-4, patience=3, verbose=False, mode='min')
    # # Instantiate learner
    # learner_T = Learner(observer=observer, system=system, training_data=data,
    #                     validation_data=val_data, method='T', batch_size=10,
    #                     lr=1e-3, optimizer=optim.Adam,
    #                     optimizer_options=optimizer_options,
    #                     scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    #                     scheduler_options=scheduler_options)
    # # Define logger and checkpointing
    # logger = TensorBoardLogger(save_dir=learner_T.results_folder + '/tb_logs')
    # checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    # trainer = pl.Trainer(
    #     callbacks=[stopper, checkpoint_callback], **trainer_options, logger=logger,
    #     log_every_n_steps=1, check_val_every_n_epoch=3)
    #
    # # To see logger in tensorboard, copy the following output name_of_folder
    # print(f'Logs stored in {learner_T.results_folder}/tb_logs')
    # # which should be similar to jupyter_notebooks/runs/method/exp_0/tb_logs/
    # # Then type this in terminal:
    # # tensorboard --logdir=name_of_folder --port=8080
    #
    # # Train and save results
    # trainer.fit(learner_T)
    # # learner_T.save_results(limits=np.array([[-1, 1.], [-1., 1.]]),
    # #                        nb_trajs=10, tsim=(0, 60), dt=1e-2,
    # #                        checkpoint_path=checkpoint_callback.best_model_path)
    # print('apprentissage T terminé')
    #---------------------------------------------------------------------------------------------------
    # Train the inverse transformation using pytorch-lightning and the learner class
    # Options for training
    trainer_options = {'max_epochs': 15}
    optimizer_options = {'weight_decay': 1e-8}
    scheduler_options = {'mode': 'min', 'factor': 0.1, 'patience': 3,
                         'threshold': 1e-4, 'verbose': True}
    stopper = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss', min_delta=5e-4, patience=3, verbose=False, mode='min')
    # Instantiate learner
    learner_T_star = Learner(observer=observer, system=system, training_data=data,
                             validation_data=val_data, method='T_star',
                             batch_size=10, lr=5e-4, optimizer=optim.Adam,
                             optimizer_options=optimizer_options,
                             scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                             scheduler_options=scheduler_options)
    # Define logger and checkpointing
    logger = TensorBoardLogger(save_dir=learner_T_star.results_folder + '/tb_logs')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    trainer = pl.Trainer(
        callbacks=[stopper, checkpoint_callback], **trainer_options, logger=logger,
        log_every_n_steps=1, check_val_every_n_epoch=3)

    # %%
    # To see logger in tensorboard, copy the following output name_of_folder
    print(f'Logs stored in {learner_T_star.results_folder}/tb_logs')
    # which should be similar to jupyter_notebooks/runs/method/exp_0/tb_logs/
    # Then type this in terminal:
    # tensorboard --logdir=name_of_folder --port=8080

    # Train and save results
    trainer.fit(learner_T_star)
    learner_T_star.save_results(limits=np.array([[-1, 1.], [-1., 1.]]),
                                nb_trajs=5, tsim=(0, 60), dt=1e-2,
                                checkpoint_path=checkpoint_callback.best_model_path)

