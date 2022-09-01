# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:17:56 2022

@author: pchauris
"""

"Les outils de raffinement de façon plus ordonnés"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class Cell:
    def __init__(self,index,geometry):
        
        Lx, Ly, Nx, Ny, Ox, Oy = geometry
        px,py = Lx/Nx,Ly/Ny
        x = index[0]*px + px/2 + Ox
        y = index[1]*py + py/2 + Oy
        
        self.index = index
        self.geometry = geometry
        self.center = np.array([x,y])
        self.size = np.array([px,py])
        
    def info(self):
        print('index :',self.index)
        print('center :',self.center)
        print('size :',self.size)
        
    def split_iso(self):
        px,py = self.size
        #create 4 new cells
        C00 = Cell(np.concatenate((self.index,[0,0])),self.geometry)
        C00.size = self.size/2
        C00.center = self.center - np.array([px/4,py/4])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),self.geometry)
        C01.size = self.size/2
        C01.center = self.center + np.array([px/4,-py/4])
        
        C10 = Cell(np.concatenate((self.index,[1,0])),self.geometry)
        C10.size = self.size/2
        C10.center = self.center + np.array([-px/4,py/4])
        
        C11 = Cell(np.concatenate((self.index,[1,1])),self.geometry)
        C11.size = self.size/2
        C11.center = self.center + np.array([px/4,py/4])
        
        return C00,C01,C10,C11
    
    def split_x(self):
        px,py = self.size
        #create 2 new cells allong first axis
        C00 = Cell(np.concatenate((self.index,[0,0])),self.geometry)
        C00.size = np.array([px/2,py])
        C00.center = self.center - np.array([px/4,0])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),self.geometry)
        C01.size = np.array([px/2,py])
        C01.center = self.center + np.array([px/4,0])
        
        return C00,C01
    
    
    def split_y(self):
        px,py = self.size
        #create 2 new cells allong second axis
        C00 = Cell(np.concatenate((self.index,[0,0])),self.geometry)
        C00.size = np.array([px,py/2])
        C00.center = self.center - np.array([0,py/4])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),self.geometry)
        C01.size = np.array([px,py/2])
        C01.center = self.center + np.array([0,py/4])
        
        return C00,C01


def init_grid(geometry):
    grid = []
    Nx,Ny = geometry[2:4]
    for i in range(Nx):
        for j in range(Ny):
            cell = Cell(np.array([i,j]),geometry)
            grid.append(cell)
    return grid


def coordinate(grid):
    X1 = []
    X2 = []
    for cell in grid:
        x1,x2 = cell.center
        X1.append(x1)
        X2.append(x2)
    return np.array(X1),np.array(X2)

def plot_cell(cell):
    x,y = cell.center
    px,py = cell.size
    plt.vlines(x = x-px/2, ymin = y-py/2, ymax = y+py/2,linewidth=1)
    plt.vlines(x = x+px/2, ymin = y-py/2, ymax = y+py/2,linewidth=1)
    plt.hlines(y = y-py/2, xmin = x-px/2, xmax = x+px/2,linewidth=1)
    plt.hlines(y = y+py/2, xmin = x-px/2, xmax = x+px/2,linewidth=1)
    

def crit_sequence(Xi,Z):
    return erreur_gp(Xi,Z)

def alpha_sequence(Xi,Z):
    sequence = crit_sequence(Xi,Z)
    return np.linspace(0,sequence.max(),Xi.size)


def distrib_sequence(Xi,Z):
    alpha = alpha_sequence(Xi,Z)
    crit = crit_sequence(Xi,Z)
    distribution = []
    for j in range(alpha.size):
        dj = np.count_nonzero(crit>alpha[j])
        distribution.append(dj)
    return np.array(distribution)


def auto_threshold(Xi,Z):
    alpha = alpha_sequence(Xi,Z)
    distribution = distrib_sequence(Xi,Z)
    f = alpha*distribution
    #maximum global
    fmax = np.max(f)
    idmax = np.where(f == fmax)
    idmax = idmax[0][0]
    #alpha 
    alphamax = alpha[idmax]
    return alphamax

def iterate_grid(grid,Xi,Z):
    alpha = auto_threshold(Xi,Z)
    crit = crit_sequence(Xi,Z)
    new_grid = grid.copy()
    for cell in grid:
        k = grid.index(cell)        
        # raffinement iso
        if crit[k] > alpha :
            C00,C01,C10,C11 = cell.split_iso()
            new_grid.remove(cell)
            new_grid.insert(k,C11)
            new_grid.insert(k,C10)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
        # if len(new_grid) > 768:
        #     break   
    return new_grid

def erreur_gp(Xi,Z):
    "retourne l'erreur d'estimation de Xi = T*(Z) par un processus gaussien"
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(Xi.size), size=20, replace=False)
    Z_train, x_train = Z[training_indices,:], Xi[training_indices]

    noise_std = 0.1
    x_train_noisy = x_train + rng.normal(loc=0.0, scale=noise_std, size=x_train.shape)
    
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(Z_train, x_train_noisy)
    gaussian_process.kernel_

    mean_prediction, std_prediction = gaussian_process.predict(Z, return_std=True)

    erreur = np.abs(Xi-mean_prediction)
    return erreur
    
    