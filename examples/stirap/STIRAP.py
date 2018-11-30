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

class moving_well:
    def __init__(self, d, w, dmin, t_dur, t_start):
        self.d = d
        self.w = w
        self.dmin = dmin
        self.t_dur = t_dur
        self.t_start = t_start
    
    
    
    def compute(self, x, t):
        def compute_pot(x, w, xL, xM, xR):
            v = -1.*(np.exp(-2*((x-xL)/w)**2) + np.exp(-2*((x-xM)/w)**2) + np.exp(-2*((x-xR)/w)**2))
            return v
        
        def g(x, w, x0):
            return -1.*(np.exp(-2*((x-x0)/w)**2))
        #test
        pos_m = (self.d-self.dmin)*np.sin(2*np.pi*min(1., max(0, (t-self.t_start)/self.t_dur) ) ) 
        
        
        xs = self.d if pos_m >= 0. else -self.d
        vmax = np.max(np.abs(g(x, self.w, pos_m)+g(x, self.w, xs)))
        ampl = vmax/2

        pot = (2.-vmax)*g(x, self.w, pos_m)+g(x, self.w, -self.d)+g(x, self.w, self.d) #compute_pot(x, self.w, -self.d, pos_m, self.d)
        return pot

def s(t, t0, td, a, b, std=1.):
            s = (np.tanh((t-t0+td/2.)/std)+np.tanh((-t+t0+td/2.)/std))*0.5
            return (1.-s)*a+s*b
    
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
    
    def plot(self):
        plt.figure()
        temp = np.linspace(0, max(self.t_0, self.t_1)*2+10., 200)
        plt.plot(temp, s(temp, self.t_0, self.t_dur, self.d, self.dmin, self.std), 'r-')
        plt.plot(temp, s(temp, self.t_0, self.t_dur, -self.d, -self.dmin, self.std), 'r-.')
        plt.plot(temp, s(temp, self.t_1, self.t_dur, -self.d, -self.dmin, self.std), 'b-')
        plt.show()

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
t_delay = 100#0.25*t_r
t_i = 500#0.3*t_r

waist = 2.
movingPot = moving_well2(d=15., w=waist, dmin=3., t_dur=t_i, t_0=(tEnd-t_delay)/2, t_1=(tEnd+t_delay)/2., std=60.)
movingPot.plot()

def s2(t, t0, td, a, b):
    return (np.tanh(t-t0+td/2.)+np.tanh(-t+t0+td/2.))*0.5

Simulation = TDSES.TDSEND_solver(coords=gridsize, dt=1.0e-4, tol=1.0e-4, Vtd = [movingPot,])#, 
#Simulation.addPotential(compute_pot, w=2., xL=-15., xM=0., xR=15.)
#Simulation.addPotentialTd(movingPot)
#Simulation.positivePotential()
Simulation.addCAP()
#Simulation.addPotential(gauss1d, mx=0,  sigx=2., ampl=-10.)
Simulation.setWavefunction(gauss1d, mx=-15.0,  sigx=waist)

#plt.figure()
#plt.plot(np.linspace(a,b,N), Simulation._V0.real)
#plt.colorbar()
#plt.show(block=False)

#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,7))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))
title0 = fig.suptitle('N={:d}, dt={:.1e}'.format(N, dt))
title = ax1.set_title("x-y")
title2 = ax2.set_title("px-py")

ext= Simulation.getxExtent()
pext= Simulation.getpExtent()

#status = tqdm(total=imagSteps+200)
t = 0

def frame_generator():
    global t, tEnd
    f=0
    #print(t, tEnd)
    while t<tEnd:
        f += 1
        print(t, f)
        yield f


im1, = ax1.plot(np.linspace(a,b,N), Simulation.getPsiR0())
im2, = ax2.plot(np.linspace(*pext[0],N),Simulation.getPsiP0())
ax1.set_xlim(-30, 30)
ax2.set_xlim(-30, 30)
ax1.set_ylim(0., 0.002)
ax2.set_ylim(0., 10.)

ax5 = ax1.twinx()
im3, = ax5.plot(np.linspace(a,b,N), Simulation.getV0(), 'r', alpha=0.5)
# initialization function: plot the background of each frame
def init():
    return im1, im2, title, title2, im3, 

# animation function.  This is called sequentially
def animate(i):
    global N, t

    Simulation.propT_adaptive(10)
    #Simulation.propT(dtStepsatOnce)
    t = Simulation.getT()
    print(t)
    title.set_text("x, t = {:.3f}".format(t))
    title2.set_text(
        r'$|\Psi_0(x)|$ = {:.6f} $|\Psi_0(p)|$ = {:.4f}'.format(Simulation.getNormPsiR0(), Simulation.getNormPsiP0()))


    im1.set_ydata(Simulation.getPsiR0())
    psip=Simulation.getPsiP0()
    im2.set_ydata(psip)
    #update n(k=0) plot
    im3.set_ydata(Simulation.getV0())
    #im1.autoscale()
    #im2.autoscale()
    #status.update()
    return im1, im2, title, title2, im3,


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frame_generator, interval=20, blit=False, repeat=False, save_count=1000)
#plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Lukas\\Documents\\ffmpeg\\bin'
FFwriter = animation.FFMpegWriter(fps=30, codec="h264", bitrate=10000)

#ani.save('stirap.mp4'.format(N), dpi=300, writer=FFwriter)
plt.show()
fig.savefig(datestring+'stirap.png', dpi=300)
#status.close()
