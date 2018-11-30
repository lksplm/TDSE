import numpy as np
import math
from time import time
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import trapz
from tqdm import tqdm
import pickle

import TDSEND as TDSES

datestring = time.strftime("%y%m%d_%H%M")

def compute_ho2(x, y, wx, wy):
    v = wx**2*x**2 + wy**2*y**2
    return v

def gauss2d(x, y, mx, my, sigx, sigy, ampl=1.0):
    return ampl/(2*math.pi*sigx*sigy)*np.exp(-(((x-mx)**2/(2*sigx**2))+((y-my)**2/(2*sigy**2))))

def exp_pot(x, y, b, c):
    eps = 1.0e-4
    '''
    best parameters so far:
    eps = 1.0e-4
    b=0.2, c=0.0
    '''
    #return b/((x+eps)**2+(y+eps)**2)+c/((x+eps)**3+(y+eps)**3)
    return b / ((np.abs(x) + eps) ** 2 + (np.abs(y) + eps) ** 2)

def exp_pot_asym(x, y, b, c):
    eps = 1.0e-4
    return b / ((np.abs(x) + eps) ** 2 + (np.abs(y) + eps) ** 2) + x*y*c/( (np.abs(x)+eps)**3+(np.abs(y)+eps)**3)

def gauss3d(x, y, z, mx, my, mz, sigx, sigy, sigz):
    return 1/(2*math.pi*sigx*sigy*sigz)*np.exp(-(((x-mx)**2/(2*sigx**2))+((y-my)**2/(2*sigy**2))+((z-mz)**2/(2*sigz**2)) ))
#############################################
# Set up simulation parameters
#############################################
dt = 1.0e-3
tStart = 0
tEnd = 4.0
dtStepsatOnce = 10
imagSteps = 40

a, b = -40, 40
N = [1024, 1024]#[512,512]#[256, 256]
gridsize=[[a, b, N[0]], [a, b, N[1]]]

Simulation = TDSES.TDSEND_solver(coords=gridsize, dt=1.0e-3, tol=1.0e-5)
Simulation.addPotential(compute_ho2, wx=1., wy=1.)
Simulation.addPotential(exp_pot, b=0.2, c=0.0)
#Simulation.addPotential(gauss2d, mx=0., my=0., sigx=0.1, sigy=0.1, ampl=50.0)
#Simulation.positivePotential()
Simulation.addCAP()
Simulation.setWavefunction(gauss2d, mx=0.0, my=0.0, sigx=1.3, sigy=1.3)

plt.figure()
plt.imshow(Simulation._V0[:,:].real)
plt.colorbar()
plt.show(block=False)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,12))
title0 = fig.suptitle('N={:d}, dt={:.1e}'.format(N[0], dt))
title = ax1.set_title("x-y")
title2 = ax2.set_title("px-py")

ext= Simulation.getxExtent()
pext= Simulation.getpExtent()
im1 = ax1.imshow(Simulation.getPsiR0() [:,:].T , interpolation='none', origin='lower', aspect='equal',  extent=[*ext[0], *ext[1]])
im2= ax2.imshow(Simulation.getPsiP0()[:,:], interpolation='none', origin='lower', aspect='equal', extent=[*pext[0], *pext[1]])
#create plot for n(k=0)
ts = []
nks = []
plt1 = ax3.plot(ts, nks)
ax3.set_title("")
ax3.set_xlabel('t')
ax3.set_ylabel('n(k=0)')

k1d = Simulation.getPsiP0()[N[0]//2, N[1]//2:N[1]//2+20]
k1d = np.zeros_like(k1d)
k1d_t = np.array(k1d).reshape((-1,1))
im3=ax4.imshow(k1d_t, interpolation='none', origin='lower', aspect='equal', extent=[0, 0.1, pext[1][0]//2,  pext[1][1]])

ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)

status = tqdm(total=imagSteps+200)
t = 0

def frame_generator():
    global t, tEnd
    f=0
    print(t, tEnd)
    while t<tEnd:
        f += 1
        print(t, f)
        yield f

# initialization function: plot the background of each frame
def init():
    return im1, im2, im3, plt1, title, title2,

# animation function.  This is called sequentially
def animate(i):
    global N, t, status, imagSteps, k1d_t

    Simulation.propT_adaptive(dtStepsatOnce)

    t = Simulation.getT()
    title.set_text("x-y, t = {:.3f} ms ".format(t))
    title2.set_text(
        r'$|\Psi_0(x)|$ = {:.6f} $|\Psi_0(p)|$ = {:.4f}'.format(Simulation.getNormPsiR0(), Simulation.getNormPsiP0()))


    im1.set_array(Simulation.getPsiR0()[:,:].T)
    psip=Simulation.getPsiP0()[:, :]
    #psip = psip**2
    im2.set_array(psip)
    #update n(k=0) plot
    ts.append(t)
    nks.append(psip[N[0]//2, N[0]//2])
    plt1[0].set_data(ts, nks)
    ax3.relim()
    ax3.autoscale_view()

    k1d = psip[N[0]//2, N[1]//2:N[1]//2+20].reshape((-1,1))
    k1d_t = np.append(k1d_t, k1d, axis=1)
    ax4.imshow(k1d_t, interpolation='none', origin='lower',  aspect='auto', extent=[0, t, 0, pext[1][1]/2])

    im1.autoscale()
    im2.autoscale()
    status.update()
    return im1, im2, im3, title, title2,


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frame_generator, interval=20, blit=False, repeat=False, save_count=1000)
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
FFwriter = animation.FFMpegWriter(fps=30, codec="h264", bitrate=10000)

ani.save('movies/' + datestring + 'mexican_hat_{}.mp4'.format(N), dpi=300, writer=FFwriter)
#plt.show()
try:
    fig.savefig(datestring + 'mexican_hat_{}.png'.format(N), dpi=300)
except:
    pass

status.close()
