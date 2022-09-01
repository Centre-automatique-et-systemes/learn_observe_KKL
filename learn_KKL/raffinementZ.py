# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:11:11 2022

@author: pchauris
"""
import numpy as np
import matplotlib.pyplot as plt

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
    return np.array([np.array(X1),np.array(X2)])

def plot_cell(cell):
    x,y = cell.center
    px,py = cell.size
    plt.vlines(x = x-px/2, ymin = y-py/2, ymax = y+py/2,linewidth=1)
    plt.vlines(x = x+px/2, ymin = y-py/2, ymax = y+py/2,linewidth=1)
    plt.hlines(y = y-py/2, xmin = x-px/2, xmax = x+px/2,linewidth=1)
    plt.hlines(y = y+py/2, xmin = x-px/2, xmax = x+px/2,linewidth=1)
    

def crit_sequence(grid,Z,nx,ny,Coeffs):
    return gradient(grid,Z,nx,ny,Coeffs)


def alpha_sequence(grid,Z,nx,ny,Coeffs):
    res = crit_sequence(grid,Z,nx,ny,Coeffs)
    return np.linspace(0,res.max(),len(grid))


def distrib_sequence(grid,Z,nx,ny,Coeffs):
    alpha = alpha_sequence(grid,Z,nx,ny,Coeffs)
    crit = crit_sequence(grid,Z,nx,ny,Coeffs)
    res = []
    for j in range(alpha.size):
        dj = np.count_nonzero(crit>alpha[j])
        res.append(dj)

    return np.array(res)

def auto_threshold(grid,Z,nx,ny,Coeffs):
    alpha = alpha_sequence(grid,Z,nx,ny,Coeffs)
    d = distrib_sequence(grid,Z,nx,ny,Coeffs)
    f = alpha*d
    #maximum global
    fmax = np.max(f)
    idmax = np.where(f == fmax)
    idmax = idmax[0][0]
    #alpha 
    alphamax = alpha[idmax]
    return alphamax


def iterate_grid(grid,Z,axe):
    
    nx,ny = int(np.sqrt(len(grid))/5),int(np.sqrt(len(grid))/5)
    Coeffs = coeffs(grid,Z,axe,nx,ny)
    
    alpha = auto_threshold(grid,Z,nx,ny,Coeffs)
    new_grid = grid.copy()
    crit_seq = crit_sequence(grid,Z,nx,ny,Coeffs)
    for cell in grid:
        k = grid.index(cell)
        # raffinement iso
        if crit_seq[k] > alpha :
            C00,C01,C10,C11 = cell.split_iso()
            if cell in new_grid:
                new_grid.remove(cell)
            new_grid.insert(k,C11)
            new_grid.insert(k,C10)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
    
    return new_grid

def split_grid(grid,Z,nx,ny,ix,iy):
    "return the sub grid of index (ix,iy) from the grid divided in nx time ny regions"
    sub_grid = []
    sub_Z = []
    for cell in grid:
        ind = grid.index(cell)
        x,y = cell.center
        Lx,Ly = cell.geometry[0:2]
        Ox,Oy = cell.geometry[4:6]
        if ix*Lx/nx+Ox < x < (ix+1)*Lx/nx+Ox and iy*Ly/ny+Oy < y < (iy+1)*Ly/ny+Oy:
            sub_grid.append(cell)
            sub_Z.append(Z[ind])
    return sub_grid,np.array(sub_Z)


def surr_model(sub_grid,sub_Z,axe):
    "return the array of coefficients of the polynomial model fit over sub_grid"
    X = coordinate(sub_grid)[axe]
    Z1,Z2,Z3 = sub_Z[:,0],sub_Z[:,1],sub_Z[:,2]
    # degré 2
    A = np.stack((Z1**2,Z2**2,Z3**2,Z1*Z2,Z1*Z3,Z2*Z3,Z1,Z2,Z3,np.ones(Z1.size)),-1)
    # resolution du système
    A_star = np.linalg.pinv(A)
    param = np.dot(A_star,X)
    return param


def coeffs(grid,Z,axe,nx,ny):
    "return the list of coefficients of all the nx time ny surrogate models of the grid"
    Coeffs = []
    for iy in range(ny):
        for ix in range(nx):
            sub_grid,sub_Z = split_grid(grid,Z,nx,ny,ix,iy)
            param = surr_model(sub_grid,sub_Z,axe)
            Coeffs.append(param)
    return Coeffs


def gradient(grid,Z,nx,ny,Coeffs):
    grad =  []
    Lx,Ly = grid[0].geometry[0:2]
    Ox,Oy = grid[0].geometry[4:6]
    ind = 0
    for x,y in coordinate(grid).T:
        z1,z2,z3 = Z[ind,:]
        ind += 1
        ix,iy = int((x-Ox)/(Lx/nx)), int((y-Oy)/(Ly/ny))
        # degré 2
        [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10] = Coeffs[iy*nx+ix]
        g1 = 2*a1*z1+a4*z2+a5*z3+a7
        g2 = 2*a2*z2+a4*z1+a6*z3+a8
        g3 = 2*a3*z3+a5*z1+a6*z2+a9
        grad.append(np.sqrt(g1**2+g2**2+g3**2))
    return np.array(grad)