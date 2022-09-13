# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:08:46 2022

@author: pchauris
"""

"Implémenter le raffinement pour une grille de dimension quelconque n, sur un critère de Rn dans R"

import numpy as np
import matplotlib.pyplot as plt

class Cell:
    def __init__(self,index,geometry):
        L,N,O = geometry
        dimension = np.size(L)
        P = [L[i]/N[i] for i in range(dimension)]
        X = [index[i]*P[i] + P[i]/2 + O[i] for i in range(dimension)]

        self.index = index
        self.geometry = geometry
        self.center = np.array(X)
        self.size = np.array(P)
        
    def info(self):
        print('index :',self.index)
        print('center :',self.center)
        print('size :',self.size)
        
    def split(self):
        P = self.size
        X = self.center
        combi = combinaisons(self.geometry)
        dim = X.size
        # create 2**dim new cells
        new_cells = []
        for i in range(2**dim):
            new_x = X + combi[i]
            new_cell = Cell(np.zeros(dim),self.geometry) # dont care about index, only usefull to create the original uniform grid
            new_cell.center = new_x
            new_cell.size = P/2
            new_cells.append(new_cell)
        return new_cells
    

def indices(geometry):
    L,N,O = geometry
    dim = np.size(L)
    axes = np.array([np.arange(N[i]) for i in range(dim)])
    ind = np.meshgrid(*axes)
    res = np.array([elem for elem in ind])
    res = res.reshape(dim,np.product(N))
    res = [[res[i,j] for i in range(dim)] for j in range(np.product(N))]
    return np.array(res)

def init_grid(geometry):
    indice = indices(geometry)
    grid = [Cell(ind,geometry) for ind in indice]
    return grid

def coordinate(grid):
    pos = [cell.center for cell in grid]
    return np.array(pos)
            
def combinaisons(geometry):
    L,N,O = geometry
    dim = np.size(L)
    P = [L[i]/N[i] for i in range(dim)]
    axes = np.array([[-1,1] for i in range(dim)])
    ind = np.meshgrid(*axes)
    res = np.array([elem for elem in ind])
    res = res.reshape(dim,2**dim).T
    return res*P/4

def raffinement_complet(grid):
    new_grid = []
    for cell in grid:
        new_cells = cell.split()
        for cells in new_cells:
            new_grid.append(cells)
    return new_grid

def alpha_sequence(grid,critere):
    sequence = critere
    return np.linspace(0,sequence.max(),len(grid))

def distrib_sequence(grid,critere):
    alpha = alpha_sequence(grid,critere)
    distribution = [np.count_nonzero(critere>alpha_k) for alpha_k in alpha]
    return np.array(distribution)

def auto_threshold(grid,critere):
    alpha = alpha_sequence(grid,critere)
    distribution = distrib_sequence(grid,critere)
    f = alpha*distribution
    #maximum global
    fmax = np.max(f)
    idmax = np.where(f == fmax)
    idmax = idmax[0][0]
    #alpha 
    alphamax = alpha[idmax]
    return alphamax

def raffinement2(grid,critere):
    "en chaque point x de la grille, raffine si critere(x) > alpha"
    new_grid = []
    alpha = auto_threshold(grid,critere)
    for k in range(len(grid)):
        cell = grid[k]
        if critere[k] > alpha:
            new_cells = cell.split()
            for cells in new_cells:
                new_grid.append(cells)
        else:
            new_grid.append(cell)

    return new_grid

def uniform_grid(geometry,npts):
    L,_,O = geometry
    dim = np.size(L)
    N = int(np.power(npts,1/dim)) + 1
    mesh = np.array([np.linspace(O[i],O[i]+L[i],N) for i in range(dim)])
    mesh = np.meshgrid(*mesh)
    grid = np.array([mesh[i] for i in range(dim)])
    grid = grid.reshape(dim,N**dim)
    while grid[0].size != npts:
        ind = np.random.randint(0,grid[0].size)
        grid = np.array([np.delete(grid[i],ind) for i in range(dim)])
    return grid