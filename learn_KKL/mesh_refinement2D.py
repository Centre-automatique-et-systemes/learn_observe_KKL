# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:57:24 2022

@author: pchauris
"""
#%% imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from findiff import FinDiff
import time
#%% def classe et fonctions

class Cell:
    def __init__(self,index,level,L,x10,x20):
        self.level = level
        self.index = index
        self.L = L
        self.x10 = x10
        self.x20 = x20
    def info(self):
        print('index :',self.index)
        print('level :',self.level)
    def split(self):
        C00 = Cell(np.concatenate((self.index,[0,0])),self.level + 1,self.L,self.x10,self.x20)
        C01 = Cell(np.concatenate((self.index,[0,1])),self.level + 1,self.L,self.x10,self.x20)
        C10 = Cell(np.concatenate((self.index,[1,0])),self.level + 1,self.L,self.x10,self.x20)
        C11 = Cell(np.concatenate((self.index,[1,1])),self.level + 1,self.L,self.x10,self.x20)
        return C00,C01,C10,C11
    def center(self):
        index = self.index
        level = self.level
        L = self.L
        x10 = self.x10
        x20 = self.x20
        x1 = 0
        x2 =  0
        for k in range(level):
            x1 += index[2*k]*L/2**k/N
            x2 += index[2*k+1]*L/2**k/N
        x1 += L/N/2**(k+1) + x10
        x2 += L/N/2**(k+1) + x20
        return np.array([x1,x2])

def cen(grid):
    X1 = []
    X2 = []
    for cell in grid:
        x1,x2 = cell.center()
        X1.append(x1)
        X2.append(x2)
    return np.array(X1),np.array(X2)

def init_grid(N,L,x10,x20):
    grid = []
    for i in range(N):
        for j in range(N):
            grid.append(Cell(np.array([i,j]),1,L,x10,x20))
    return grid

def f(X,Y):
    return np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2)

def emp_grad(cell):
    x1,y1 = cell.center()
    level = cell.level
    dx = L/(N*2**(level-1))
    x0,y0 = x1 - dx/2 , y1 - dx/2
    Dfx = f(x0+dx,y0) - f(x0,y0)
    Dfy = f(x0,y0+dx) - f(x0,y0)
    return np.abs(Dfx/dx),np.abs(Dfy/dx)


def crit_sequence(grid):
    res = []
    for cell in grid:
        gx,gy = emp_grad(cell)
        res.append(np.sqrt(gx**2+gy**2))
        # res.append(max(gx,0))
    return np.array(res)

def alpha_sequence(grid):
    res = crit_sequence(grid)
    return np.linspace(0,res.mean(),len(grid))

def distrib_sequence(grid):
    alpha = alpha_sequence(grid)
    crit = crit_sequence(grid)
    res = []
    for j in range(alpha.size):
        dj = np.count_nonzero(crit>alpha[j])
        res.append(dj)
    return np.array(res)

def auto_threshold(grid):
    alpha = alpha_sequence(grid)
    d = distrib_sequence(grid)
    f = alpha*d
    #maximum global
    fmax = np.max(f)
    idmax = np.where(f == fmax)
    idmax = idmax[0][0]
    #alpha 
    alphamax = alpha[idmax]
    return alphamax

def iterate_grid(grid,alpha):
    new_grid = grid.copy()
    for cell in grid:
        k = grid.index(cell)
        gx,gy = emp_grad(cell)
        if np.sqrt(gx**2+gy**2) > alpha:
        # if max(gx,0) > alpha:
            C00,C01,C10,C11 = cell.split()
            new_grid.remove(cell)
            new_grid.insert(k,C11)
            new_grid.insert(k,C10)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
    return new_grid

def raffinement(grid,niter):
    for _ in range(niter):
        alpha = auto_threshold(grid)
        grid = iterate_grid(grid,alpha)
    return grid
    