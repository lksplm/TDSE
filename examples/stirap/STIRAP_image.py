# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:19:00 2018

@author: Lukas
"""
import numpy as np
import math
from time import time
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from scipy.integrate import trapz
from tqdm import tqdm
import pickle

import os
os.system("vcvarsamd64.bat")

import TDSEND as TDSES

datestring = time.strftime("%y%m%d_%H%M")


def gauss1d(x, mx, sigx, ampl=1.0):
    return ampl/(2*math.pi*sigx**2)*np.exp(-(((x-mx)**2/(sigx**2))))


def s(t, t0, td, a, b, std=1.):
            s = (np.tanh((t-t0+td/2.)/std)+np.tanh((-t+t0+td/2.)/std))*0.5
            return (1.-s)*a+s*b
        
class moving_well:
    def __init__(self, d, w, dmin, t_dur, t_0, t_1, std=1., pert=0.):
        self.d = d
        self.w = w
        self.dmin = dmin
        self.t_dur = t_dur #tau
        self.t_0 = t_0 #T
        self.t_1 = t_1
        self.std = std
        self.pert = pert
    
    def s(self, t):
        """
        if t<=self.t_0:
            xL = -self.d
        else:
            xL = -self.dmin - (self.d-self.dmin)*np.cos(np.pi*(t-self.t_0)/self.t_dur)**2
        """
        xL = np.where(t<=self.t_0, -self.d, -self.dmin - (self.d-self.dmin)*np.cos(np.pi*(t-self.t_0)/self.t_dur)**2)
        """
        if t>self.t_0:
            xR = self.d
        else:
            xR = self.dmin + (self.d-self.dmin)*np.cos(np.pi*t/self.t_dur)**2
        """
        xR = np.where(t>self.t_dur, self.d, self.dmin + (self.d-self.dmin)*np.cos(np.pi*t/self.t_dur)**2)
        return xL, xR
    
    def compute(self, x, t):
        def g(x, w, x0):
            return -1.*(np.exp(-2*((x-x0)/w)**2))

        xL, xR = self.s(t)
        
        vR = np.max(np.abs(g(x, self.w, 0)+g(x, self.w, xR)))
        vL = np.max(np.abs(g(x, self.w, 0)+g(x, self.w, xL)))
        
        pert = lambda x: (1.+self.pert*np.sin(x*2*np.pi/0.158)+0.001*x)
        pot = g(x, self.w, 0)+(2.-vR)*g(x, self.w, xR)*pert(xR)+(2.-vL)*g(x, self.w, xL)*pert(xL)
        return pot
    
    def plot(self, ax=None, save=False):
        temp = np.linspace(0, self.t_dur+self.t_0, 200)
        xL, xR = self.s(temp)
        plt.plot(temp, xL, 'r-', label='L')
        plt.plot(temp, np.zeros_like(temp), 'g-', label='M')
        plt.plot(temp, xR, 'b-',  label='R')
        plt.xlabel(r'$\omega t$')
        plt.ylabel(r'$d(t)/a$')
        plt.legend()
        if save:
            plt.savefig("potential.png", dpi=300)
        plt.show(block=True)
        
    def get_x(self, temp):
        a, b = self.s(temp)
        c = np.zeros_like(b)
        return a, b, c
        
class moving_well2:
    def __init__(self, d, w, dmin, t_dur, t_0, t_1, std=1.):
        self.d = d
        self.w = w
        self.dmin = dmin
        self.t_dur = t_dur
        self.t_0 = t_0
        self.t_1 = t_1
        self.std = std
    
    def compute(self, x, t):
        def g(x, w, x0):
            return -1.*(np.exp(-2*((x-x0)/w)**2))

        xR = s(t, self.t_0, self.t_dur, self.d, self.dmin, self.std)
        xL = s(t, self.t_1, self.t_dur, -self.d, -self.dmin, self.std)
        
        vR = np.max(np.abs(g(x, self.w, 0)+g(x, self.w, xR)))
        vL = np.max(np.abs(g(x, self.w, 0)+g(x, self.w, xL)))
        
        pert = lambda x: (1.+0.01*np.sin(x*2*np.pi/0.158)+0.001*x)
        pot = g(x, self.w, 0)+(2.-vR)*g(x, self.w, xR)*pert(xR)+(2.-vL)*g(x, self.w, xL)*pert(xL)
        return pot
    
    def plot(self, ax=None):
        temp = np.linspace(0, max(self.t_0, self.t_1)*2+10., 200)
        plt.plot(temp, s(temp, self.t_0, self.t_dur, self.d, self.dmin, self.std), 'r-')
        plt.plot(temp, s(temp, self.t_0, self.t_dur, -self.d, -self.dmin, self.std), 'r-.')
        plt.plot(temp, s(temp, self.t_1, self.t_dur, -self.d, -self.dmin, self.std), 'b-')
        plt.xlabel(r'$\omega t$')
        plt.ylabel(r'$d(t)/a$')
        plt.show(block=True)
        
    def get_x(self, temp):

        a = s(temp, self.t_0, self.t_dur, self.d, self.dmin, self.std)
        b = s(temp, self.t_1, self.t_dur, -self.d, -self.dmin, self.std)
        c = np.zeros_like(b)
        return a, b, c
            
if __name__ == "__main__":
    plt.close()
    #############################################
    # Set up simulation parameters
    #############################################
    dt = 0.2e-1
    tStart = 0
    tEnd = 1000.0
    dtStepsatOnce = 100
    imagSteps = 40
    
    a, b = -40, 40
    N = 1024#[512,512]#[256, 256]
    gridsize=[[a, b, N]]
    
    #movingPot = moving_well(d=15., w=4., dmin=4., t_dur=30., t_start=2.)
    t_r = 300
    t_delay = 50#0.25*t_r
    t_i = 500#0.3*t_r
    
    tEnd = t_i+t_delay
    
    waist = 2.
    #movingPot = moving_well2(d=15., w=waist, dmin=3., t_dur=t_i, t_0=(tEnd-t_delay)/2, t_1=(tEnd+t_delay)/2., std=60.)
    movingPot = moving_well(d=15., w=waist, dmin=3., t_dur=t_i, t_0=t_delay, t_1=0., std=60.)
    movingPot.plot(save=True)
    
    def s2(t, t0, td, a, b):
        return (np.tanh(t-t0+td/2.)+np.tanh(-t+t0+td/2.))*0.5
    
    Simulation = TDSES.TDSEND_solver(coords=gridsize, dt=1.0e-4, tol=1.0e-4, Vtd = [movingPot,])#, 
    Simulation.addCAP()
    Simulation.setWavefunction(gauss1d, mx=-15.0,  sigx=waist)
    
    ext= Simulation.getxExtent()
    
    #status = tqdm(total=imagSteps+200)
    t = 0
    
    def frame_generator():
        global t, tEnd
        f=0
        while t<tEnd:
            f += 1
            yield f
    
    x = np.linspace(a,b,N)
    wavefunc = []
    times = []
    
    for f in frame_generator():
        Simulation.propT_adaptive(10)
        #Simulation.propT(dtStepsatOnce)
        t = Simulation.getT()
        psi=Simulation.getPsiR0()
    
        wavefunc.append(psi)
        times.append(t)
    
    
    wavefunc = np.stack(wavefunc)
    print(wavefunc.shape)
    times = np.array(times)
    
    
    X, T = np.meshgrid(x, times)
    print(X.shape)
    print(T.shape)
    plt.figure()
    plt.pcolormesh(X, T, wavefunc)
    a, b, c = movingPot.get_x(times)
    plt.plot(a, times, 'w-.', alpha=0.2)
    plt.plot(b, times, 'w-.', alpha=0.2)
    plt.plot(c, times, 'w-.', alpha=0.2)
    plt.ylabel(r'$\omega t$')
    plt.xlabel(r'$d/a$')
    plt.xlim(-25,25)
    plt.show()
    
    idx0 = np.where(x>-movingPot.dmin/2)[0][0]
    idx1 = np.where(x>movingPot.dmin/2)[0][0]
    print(idx0,"; ",idx1)
    
    well = np.empty((wavefunc.shape[0],3))
    norm = wavefunc.sum(axis=1)
    well[:,0] = wavefunc[:,0:idx0].sum(axis=1)/norm
    well[:,1] = wavefunc[:,idx0:idx1].sum(axis=1)/norm
    well[:,2] = wavefunc[:,idx1:].sum(axis=1)/norm
    plt.figure()
    lbl = ['L', 'M', 'R']
    for i in range(3):
        plt.plot(times, well[:,i], label='well {}'.format(lbl[i]))
    plt.xlabel(r'$\omega t$')
    plt.ylabel(r'$P$')
    plt.ylim(0,1)
    plt.legend()
    plt.show()
