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

from STIRAP_image import moving_well, gauss1d

def s(t, t0, td, a, b, std=1.):
            s = (np.tanh((t-t0+td/2.)/std)+np.tanh((-t+t0+td/2.)/std))*0.5
            return (1.-s)*a+s*b
    
class moving_well2:
    def __init__(self, d, w, dmin, t_dur, t_0, t_1, std=1., pert=0.0):
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
    
    def plot(self):
        plt.figure()
        temp = np.linspace(0, max(self.t_0, self.t_1)*2+10., 200)
        plt.plot(temp, s(temp, self.t_0, self.t_dur, self.d, self.dmin, self.std), 'r-')
        plt.plot(temp, s(temp, self.t_0, self.t_dur, -self.d, -self.dmin, self.std), 'r-.')
        plt.plot(temp, s(temp, self.t_1, self.t_dur, -self.d, -self.dmin, self.std), 'b-')
        plt.xlabel(r'$\omega t$')
        plt.ylabel(r'd(t)/a$')
        plt.show()

plt.close()
#############################################
# Set up simulation parameters
#############################################
dt = 0.2e-1
tStart = 0
tEnd = 662
dtStepsatOnce = 100
imagSteps = 40

a, b = -40, 40
N = 1024#[512,512]#[256, 256]
gridsize=[[a, b, N]]

#movingPot = moving_well(d=15., w=4., dmin=4., t_dur=30., t_start=2.)
t_r = 300
t_delay = 100#0.25*t_r
t_i = 500#0.3*t_r

waist = 2.
#movingPot = moving_well2(d=15., w=waist, dmin=3., t_dur=t_i, t_0=(tEnd-t_delay)/2, t_1=(tEnd+t_delay)/2., std=60.)
movingPot = moving_well(d=15., w=waist, dmin=3., t_dur=600, t_0=62, t_1=0., std=60., pert=0.0)
movingPot.plot()

def s2(t, t0, td, a, b):
    return (np.tanh(t-t0+td/2.)+np.tanh(-t+t0+td/2.))*0.5

Simulation = TDSES.TDSEND_solver(coords=gridsize, dt=1.0e-4, tol=1.0e-4, Vtd = [movingPot,])#, 
Simulation.addCAP()
Simulation.setWavefunction(gauss1d, mx=-15.0,  sigx=waist)


#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,7))
fig, ax1 = plt.subplots(1, 1, figsize=(9,7))
title0 = fig.suptitle('N={:d}, dt={:.1e}'.format(N, dt))
title = ax1.set_title("x-y")

ext= Simulation.getxExtent()

#status = tqdm(total=imagSteps+200)
t = 0

def frame_generator():
    global t, tEnd
    f=0
    while t<tEnd:
        f += 1
        yield f


im1, = ax1.plot(np.linspace(a,b,N), Simulation.getPsiR0())

ax1.set_xlim(-30, 30)
ax1.set_ylim(0., 0.002)

ax5 = ax1.twinx()
im3, = ax5.plot(np.linspace(a,b,N), Simulation.getV0(), 'r', alpha=0.5)


wavefunc = []
times = []
# initialization function: plot the background of each frame
def init():
    return im1, title, im3, 

# animation function.  This is called sequentially
def animate(i):
    global N, t, wavefunc, times

    Simulation.propT_adaptive(10)
    #Simulation.propT(dtStepsatOnce)
    t = Simulation.getT()
    title.set_text("x, t = {:.3f}".format(t))
    #title2.set_text(
    #    r'$|\Psi_0(x)|$ = {:.6f} $|\Psi_0(p)|$ = {:.4f}'.format(Simulation.getNormPsiR0(), Simulation.getNormPsiP0()))
    
    psi=Simulation.getPsiR0()
    im1.set_ydata(psi)
    wavefunc.append(psi)
    times.append(t)
    #update n(k=0) plot
    im3.set_ydata(Simulation.getV0())
    #im1.autoscale()
    return im1, title, im3,


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frame_generator, interval=20, blit=False, repeat=False, save_count=1000)
#plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Lukas\\Documents\\ffmpeg\\bin'
FFwriter = animation.FFMpegWriter(fps=30, codec="h264", bitrate=10000)

ani.save('stirap.mp4', dpi=300, writer=FFwriter)
#plt.show(block=True)
#fig.savefig(datestring+'stirap.png', dpi=300)
#status.close()

wavefunc = np.stack(wavefunc)
print(wavefunc.shape)
times = np.array(times)
print(times.shape)

plt.figure()
plt.imshow(wavefunc)
plt.show()

