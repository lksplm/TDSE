# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:53:47 2018

@author: Lukas
"""
import numpy as np
import math
from time import time
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import pickle
import os
os.system("vcvarsamd64.bat")

import TDSEND as TDSES
from scipy.interpolate import interp2d
datestring = time.strftime("%y%m%d_%H%M")
from STIRAP_image import moving_well, gauss1d

path = "scan/"+datestring
try:
    os.mkdir(path)
except:
    pass

plt.close()
#############################################
# Set up simulation parameters
#############################################
dt = 0.2e-1
tStart = 0
tEnd = 1000.0
dtStepsatOnce = 100
imagSteps = 40

detail_plot=True

a, b = -40, 40
N = 1024#[512,512]#[256, 256]
gridsize=[[a, b, N]]
x = np.linspace(a,b,N)
waist = 2.

def frame_generator():
    global t, tEnd
    f=0
    while t<tEnd:
        f += 1
        yield f

t_i = 800
t_delay = 50
movingPot = moving_well(d=15., w=waist, dmin=3., t_dur=t_i, t_0=t_delay, t_1=0., std=60.)
#movingPot.plot()

Simulation = TDSES.TDSEND_solver(coords=gridsize, dt=1.0e-4, tol=1.0e-4, Vtd = [movingPot,])
Simulation.addCAP()
Simulation.setWavefunction(gauss1d, mx=-15.0,  sigx=waist)

#movingPot = moving_well(d=15., w=4., dmin=4., t_dur=30., t_start=2.)
durations = np.linspace(600, 1800, 50)
delays = np.linspace(40, 170, 25)

Dur, Delay = np.meshgrid(durations, delays, indexing='ij')
fid = np.empty_like(Dur)
for i, t_i in enumerate(durations):
    for j, t_delay in enumerate(delays):
        print("({:d}, {:d}) tau = {}, t_delay={}".format(i, j, t_i, t_delay))
        svpath = path + "/t{:.0f}_d{:.0f}_".format(t_i, t_delay)
        tEnd = t_i+t_delay
        #fid[i,j] = np.random.rand()
        #Reset
        movingPot = moving_well(d=15., w=waist, dmin=3., t_dur=t_i, t_0=t_delay, t_1=0., std=60.)
        Simulation._t = 0
        Simulation.setWavefunction(gauss1d, mx=-15.0,  sigx=waist)
        Simulation._VsTimeDep = [movingPot,]
                
        t = 0
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
        times = np.array(times)
        idx0 = np.where(x>-movingPot.dmin/2)[0][0]
        idx1 = np.where(x>movingPot.dmin/2)[0][0]
        
        well = np.empty((wavefunc.shape[0],3))
        norm = wavefunc.sum(axis=1)
        well[:,0] = wavefunc[:,0:idx0].sum(axis=1)/norm
        well[:,1] = wavefunc[:,idx0:idx1].sum(axis=1)/norm
        well[:,2] = wavefunc[:,idx1:].sum(axis=1)/norm
        
        fidelity = well[-1,2]
        fid[i,j] = fidelity
        if detail_plot:
            X, T = np.meshgrid(x, times)
            plt.figure()
            plt.pcolormesh(X, T, wavefunc)
            a, b, c = movingPot.get_x(times)
            plt.plot(a, times, 'w-.', alpha=0.2)
            plt.plot(b, times, 'w-.', alpha=0.2)
            plt.plot(c, times, 'w-.', alpha=0.2)
            plt.ylabel(r'$\omega t$')
            plt.xlabel(r'$d/a$')
            plt.title(r'$\tau = {:.1f}, t_{{delay}}={:.1f}$'.format(t_i, t_delay))
            plt.xlim(-25,25)
            plt.savefig(svpath+"trace.png", dpi=300)
            plt.close()
            

            plt.figure()
            lbl = ['L', 'M', 'R']
            for k in range(3):
                plt.plot(times, well[:,k], label='well {}'.format(lbl[k]))
            plt.xlabel(r'$\omega t$')
            plt.ylabel(r'$P$')
            plt.ylim(0,1)
            plt.legend()
            plt.title(r'$\tau = {:.1f}, t_{{delay}}={:.1f}$'.format(t_i, t_delay))
            plt.savefig(svpath+"population.png", dpi=300)
            plt.close()

np.save(svpath+"fidelity.npy", fid)
np.save(svpath+"delays.npy", delays)
np.save(svpath+"durations.npy", durations)
plt.figure()
plt.pcolormesh(Dur, Delay, fid)
plt.ylabel(r'Delay $T \omega$')
plt.xlabel(r'Duration $\tau \omega$')
plt.colorbar()
plt.savefig(svpath+"fidelity.png", dpi=300)
plt.show()


f = interp2d(Dur, Delay, fid, kind='cubic')
xnew = np.linspace(durations[0], durations[-1], 300)
ynew = np.linspace(delays[0], delays[-1], 300)
data1 = f(xnew,ynew)
Xn, Yn = np.meshgrid(xnew, ynew)

plt.figure()
plt.pcolormesh(Xn, Yn, data1)
plt.ylabel(r'Delay $T \omega$')
plt.xlabel(r'Duration $\tau \omega$')
plt.colorbar()
plt.savefig(svpath+"fidelity_interp.png", dpi=300)
plt.show()
