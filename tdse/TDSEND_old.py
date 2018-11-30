
import numpy as np
import scipy
import math
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.fftpack import fftfreq
from scipy.integrate import trapz


from jinja2 import Template
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.elementwise as elementwise
import pycuda.reduction as reduction
import pycuda.scan as scan
import pycuda.tools as tools
from pycuda.tools import context_dependent_memoize, dtype_to_ctype
from pycuda.compiler import SourceModule
from pytools import memoize
from time import time
import skcuda.fft as cu_fft
import skcuda.misc as sk
from joblib import Memory

cachedir = '.cache'#mkdtemp()
memory = Memory(cachedir=cachedir, verbose=1)

@memory.cache
def cap(r, r_cutoff, delta_x = 0.02, k_min=0.05):
    def yf(s):
        c = 2.622057
        a = c ** 3 - 16
        b = c ** 3 - 17
        return (a * s - b * (s ** 3) + 4. / ((1. - s) ** 2) - 4. / (1. + s) ** 2) / (c ** 2)
    return np.where(np.abs(r)<r_cutoff, 0.0, 0.5j*(k_min**2)*yf((np.abs(r)-r_cutoff)/delta_x))


class TDSEND_solver(object):
    def __init__(self, coords, dt, tol=1.0e-4, psi0=None, V0=None, m=1., g=0.):
        """
        Initializes the TDSE Solver
        :param coords: the boundary coordinates of the grid, supplied as a list [(xmin,xmax,Nx),(ymin,xmax,Ny),...]
        :param dt: initial timestep, can be changed by adaptive time stepping
        :param tol: tolreance of the adaptive timestepping
        :param psi0: initial wavefunction, if not supplied it is calc. with imaginary time stepping
        :param V: potential
        """

        #######################################################################
        # Set member varibales
        #######################################################################
        self._dt = dt                                           # time step size

        self._D = len(coords)
        assert self._D <= 3
        self._x = [np.linspace(*coords[i]) for i in range(self._D)]
        self._grid = np.meshgrid(*self._x)
        self._dx = [self._x[i][1]-self._x[i][0] for i in range(self._D)]
        self._N = np.asarray(self._grid[0].shape, dtype=np.int32)
        self._dp = [2 * np.pi / (self._N[i] * self._dx[i]) for i in range(self._D)]  # momentum space spacing
        self._p = [np.float32(fftfreq(self._N[i], self._dx[i]) * 2 * np.pi) for i in range(self._D)]

        self._Psi_r0 = np.empty(self._N, dtype=np.complex64)    # spatial wave function
        self._Psi_p0 = np.zeros_like(self._Psi_r0, dtype=np.complex64) # wave function in momentum space.
        if psi0 is None:
            self._Psi_r0 =np.complex64(np.random.rand(*self._N) + 1j * np.random.rand(*self._N))
        else:
            self._Psi_r0 = np.asarray(psi0 , dtype=np.complex64)

        if V0 is None:
            self._V0 = np.zeros(self._N, dtype=np.complex64)
        else:
            self._V0 = np.asarray(V0, dtype=np.complex64)  # Potential

        self._m = m                                              # particle mass
        self._tol = tol
        self._t = 0                                             # time propagated
        self._g = g

        self._norm_Psi_r0 =0.0
        self._norm_Psi_p0 = 0.0
        #######################################################################
        # Normalize wave function input
        # -----------------------------
        #
        # If input wave function is not normalized, this is done here. Also a
        # warning is printed.
        #######################################################################
        #integral =  trapz(trapz(trapz(self._Psi_r0 * self._Psi_r0.conjugate())))*np.prod(self._dx)
        #self._Psi_r0 = self._Psi_r0/np.abs(np.sqrt(integral))
        #integral =  trapz(trapz(trapz(self._Psi_r0 * self._Psi_r0.conjugate())))*np.prod(self._dx)
        #print("Norm nach Normierung {}".format(integral))

        ################################################################################################################
        # Copy data to CUDA device
        ################################################################################################################
        cuda.init()
        print("%d device(s) found."%cuda.Device.count())
        for ordinal in range (cuda.Device.count()):
            dev = cuda.Device(ordinal)
            print("Device # %d: %s"%(ordinal, dev.name()) )
            print("  Compute Capability: %d.%d"%dev.compute_capability() )
            print("  Total Memory: %s MB"%(dev.total_memory()//(1024*1024)) )


        self.gpu_Psi0 = gpuarray.to_gpu_async(self._Psi_r0)
        self.gpu_Psi0_coarse = gpuarray.to_gpu_async(self._Psi_r0)
        self.gpu_Psi0_fine = gpuarray.to_gpu_async(self._Psi_r0)
        self.gpu_V0 = gpuarray.to_gpu_async(self._V0)
        self.gpu_xGrid = [gpuarray.to_gpu_async(self._x[i]) for i in range(self._D)]
        self.gpu_pGrid = [gpuarray.to_gpu_async(self._p[i]) for i in range(self._D)]

        self._buildKernels()
        # compute the grid size for CUDA kernel calls
        self.block = (32, 32, 1)
        s = self.gpu_Psi0.shape
        if len(s)==1:
            n, m, o = *s, 1, 1
        elif len(s)==2:
            n, m, o = *s, 1
        else:
            n, m, o = s
        gridx = int(n // self.block[0] + 1 * (n % self.block[0] != 0))
        gridy = int(m // self.block[1] + 1 * (m % self.block[1] != 0))
        gridz = int(o // self.block[2] + 1 * (o % self.block[2] != 0))
        self.grid = (gridx, gridy, gridz)
        print("Block size")
        print(self.block)
        print("Grid size")
        print(self.grid)

        #get initial psi_r norm
        self._norm_Psi_r0 = self.norm_kernel(self.gpu_Psi0).get()*np.prod(self._dx)
        self.plan = cu_fft.Plan(self.gpu_Psi0.shape, np.complex64, np.complex64)

        #Compute initial wavefunction in momentum space
        gpu_Psi_p0 = gpuarray.zeros_like(self.gpu_Psi0)

        cu_fft.fft(self.gpu_Psi0, gpu_Psi_p0, self.plan)
        self._Psi_p0 = gpu_Psi_p0.get()
        # get initial psi_p norm
        self._norm_Psi_p0 = self.norm_kernel(self.gpu_Psi0).get() * np.prod(self._dx)
        del gpu_Psi_p0


    ###########################################################################
    # Definition of class interfaces
    ###########################################################################

    def propT(self, N=1):
        """
        propT uses the split-step technique to propagate the time. This
        algorithm is unitary (as the Crank-Nicolson algorithm).

        This function is the heart of the split-step implementation.

        N: int number of time steps to be evaluated at once (default = 1).
        """
        # Copy data on CUDA device
        self.gpu_Psi0.set(self._Psi_r0)

        # Propagate half step first
        self._x_halfstep()
        # FFT
        cu_fft.fft(self.gpu_Psi0, self.gpu_Psi0, self.plan)
        # Propagate p step
        self._p_step()
        # FFT inverse
        cu_fft.ifft(self.gpu_Psi0, self.gpu_Psi0, self.plan, scale=True)

        for i in range(0,N-1):
            self._t = self._t + self._dt
            # Propagate full step
            self._x_step()
            # FFT
            cu_fft.fft(self.gpu_Psi0, self.gpu_Psi0, self.plan)
            # Propagate p step
            self._p_step()
            # FFT inverse
            cu_fft.ifft(self.gpu_Psi0, self.gpu_Psi0, self.plan, scale=True)

        # Propagate half step last
        self._x_halfstep()
        self._t = self._t + self._dt

        # retrieve data from CUDA device to RAM
        self._Psi_r0 = self.gpu_Psi0.get()
        self._norm_Psi_r0 = self.norm_kernel(self.gpu_Psi0).get() * np.prod(self._dx)

        # FFT
        cu_fft.fft(self.gpu_Psi0, self.gpu_Psi0, self.plan)
        self._Psi_p0 = self.gpu_Psi0.get()
        self._norm_Psi_p0 = self.norm_kernel(self.gpu_Psi0).get() * np.prod(self._dx)

    def imag_propT(self, N=1):

        self.gpu_Psi0.set(self._Psi_r0)

        for i in range(0,N):
            # Propagate half step first
            self._x_halfstep(imag=True)
            # FFT
            cu_fft.fft(self.gpu_Psi0, self.gpu_Psi0, self.plan)
            # Propagate p step
            self._p_step(imag=True)

            # FFT inverse
            cu_fft.ifft(self.gpu_Psi0, self.gpu_Psi0, self.plan, scale=True)

            # Propagate half step last
            self._x_halfstep(imag=True)
            #self._t = self._t + self._dt

            #renormailze
            norm = self.norm_kernel(self.gpu_Psi0_fine).get()*np.prod(self._dx)
            self.scale_kernel(self.gpu_Psi0_fine, norm)

        # retrieve data from CUDA device to RAM
        self._norm_Psi_r0 = self.norm_kernel(self.gpu_Psi0).get() * np.prod(self._dx)
        self._Psi_r0 = self.gpu_Psi0.get()
        # FFT
        cu_fft.fft(self.gpu_Psi0, self.gpu_Psi0, self.plan)
        self._Psi_p0 = self.gpu_Psi0.get()
        self._norm_Psi_p0 = self.norm_kernel(self.gpu_Psi0).get() * np.prod(self._dx)

    def _propT_adaptive(self, imag=False, method='max'):

        while True:
            old_dt = self._dt
            #coarse_step
            self.gpu_Psi0_coarse.set(self._Psi_r0)
            # Propagate half step first
            self._x_halfstep(psi=self.gpu_Psi0_coarse, dt=self._dt, imag=imag)
            # FFT
            cu_fft.fft(self.gpu_Psi0_coarse, self.gpu_Psi0_coarse, self.plan)
            # Propagate p step
            self._p_step(psi=self.gpu_Psi0_coarse, dt=self._dt, imag=imag)
            # FFT inverse
            cu_fft.ifft(self.gpu_Psi0_coarse, self.gpu_Psi0_coarse, self.plan, scale=True)
            # Propagate half step
            self._x_halfstep(psi=self.gpu_Psi0_coarse, dt=self._dt, imag=imag)


            #fine step
            self.gpu_Psi0_fine.set(self._Psi_r0)
            # Propagate half step first
            self._x_halfstep(self.gpu_Psi0_fine, dt=0.5*self._dt, imag=imag)
            # FFT
            cu_fft.fft(self.gpu_Psi0_fine, self.gpu_Psi0_fine, self.plan)
            # Propagate p step
            self._p_step(self.gpu_Psi0_fine, dt=0.5*self._dt, imag=imag)
            # FFT inverse
            cu_fft.ifft(self.gpu_Psi0_fine, self.gpu_Psi0_fine, self.plan, scale=True)
            # Propagate x step
            self._x_step(self.gpu_Psi0_fine, dt=0.5*self._dt, imag=imag)
            # FFT
            cu_fft.fft(self.gpu_Psi0_fine, self.gpu_Psi0_fine, self.plan)
            # Propagate p step
            self._p_step(self.gpu_Psi0_fine, dt=0.5*self._dt, imag=imag)
            # FFT inverse
            cu_fft.ifft(self.gpu_Psi0_fine, self.gpu_Psi0_fine, self.plan, scale=True)
            # Propagate half step
            self._x_halfstep(self.gpu_Psi0_fine, dt=0.5*self._dt, imag=imag)

            #calculate error estimates
            err = self._get_error(self.gpu_Psi0_coarse, self.gpu_Psi0_fine, method=method)

            #calculate new stepsize
            eps=1.0e-6
            dt_new = self._dt*0.9*min(max((self._tol/(err+eps))**(1./3.), 0.3), 3.0)
            #print("error {:3e}".format(err))
            #print("new dt {:.3e}".format(dt_new))
            #print("tol {:3e}".format(self._tol))
            self._dt = dt_new

            if err<self._tol:
                #print("accept")
                if imag:
                    # renormalize wavefunction
                    norm = self.norm_kernel(self.gpu_Psi0_fine).get()*np.prod(self._dx)
                    self.scale_kernel(self.gpu_Psi0_fine, np.sqrt(norm))
                else:
                    self._t = self._t + old_dt

                # retrieve data from CUDA device to RAM
                self._Psi_r0 = self.gpu_Psi0_fine.get() #TODO: Chin formula
                self._norm_Psi_r0 = self.norm_kernel(self.gpu_Psi0_fine).get() * np.prod(self._dx)
                # FFT
                cu_fft.fft(self.gpu_Psi0_fine, self.gpu_Psi0_fine, self.plan)
                self._Psi_p0 = self.gpu_Psi0_fine.get()
                self._norm_Psi_p0 = self.norm_kernel(self.gpu_Psi0_fine).get()*np.prod(self._dx)
                break

    def _get_error(self, a, b, method='max'):
        if method=='mean':
            error = self.meanabs_kernel(self.gpu_Psi0_coarse, self.gpu_Psi0_fine).get()
            error = float(error)/np.prod(self._N)
            return error
        elif method=='max':
            error = self.maxabs_kernel(self.gpu_Psi0_coarse, self.gpu_Psi0_fine).get()
            error = float(error)
            return error
        else:
            raise ValueError('Error method not known, use \'mean\' or \'max\'')



    def propT_adaptive(self, N, imag=False):
        for j in range(N):
            self._propT_adaptive(imag=imag)

    def _x_halfstep(self, psi=None, dt=None, imag=False):
        if dt is None:
            self._x_step(dt=0.5*self._dt, psi=psi, imag=imag)
        else:
            self._x_step(dt=0.5*dt, psi=psi, imag=imag)


    def _x_step(self, psi=None, dt=None, imag=False):
        if dt is None:
            dt = self._dt
        if psi is None:
            psi = self.gpu_Psi0
        if imag:
            self.imag_propT_R(*self._N[1:],
                    np.float32(dt),
                    *self.gpu_xGrid,
                    psi, self.gpu_V0, block=self.block, grid=self.grid)
        else:
            self.propT_R(*self._N[1:],
                    np.float32(dt),
                    *self.gpu_xGrid,
                    psi, self.gpu_V0, block=self.block, grid=self.grid)

    def _p_step(self, psi=None, dt=None, imag=False):
        if dt is None:
            dt = self._dt
        if psi is None:
            psi = self.gpu_Psi0
        if imag:
            self.imag_propT_P(*self._N[1:], np.float32(dt), np.float32(self._m), np.float32(self._g), *self.gpu_pGrid, psi, block=self.block, grid=self.grid)
        else:
            self.propT_P(*self._N[1:], np.float32(dt), np.float32(self._m), np.float32(self._g), *self.gpu_pGrid, psi, block=self.block, grid=self.grid)

    def _buildKernels(self):
        """
        This method makes the required CUDA variables and kernels.
        :rtype : void
        """
        ################################################################################
        # This section is the heart of the program: the CUDA compute kernel.
        #
        # First some basic functions are implemented and after that the half- and full-
        # steps in coordinate- and momentum space.
        ################################################################################

        tpl = Template("""
        #include <pycuda-complex.hpp>
        
        __global__ void step_x(
                  {% if N >= 2 %}int ySize,{% endif %}
                  {% if N == 3 %}int zSize,{% endif %}
                  float dt,
                  float *x,
                  {% if N >= 2 %} float *y, {% endif %}
                  {% if N == 3 %} float *z, {% endif %}
                  pycuda::complex<float> *Psi_r0,
                  pycuda::complex<float> *V0)
        {
            int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
            {% if N >= 2 %}int yIdx = threadIdx.y + blockIdx.y * blockDim.y; {% endif %}
            {% if N == 3 %}int zIdx = threadIdx.z + blockIdx.z * blockDim.z; {% endif %}
            
            {% if N == 3 %}
            //int idx = xIdx + ySize*(yIdx + zSize*zIdx);
            int idx = zIdx + ySize*(yIdx + zSize*xIdx);
            {% elif N == 2 %}
            int idx = xIdx + ySize*yIdx;
            {% else %}
            int idx = xIdx;
            {% endif %}
            
            pycuda::complex<float> j(0.0, 1.0);
            Psi_r0[idx] = Psi_r0[idx]*exp(-dt*j*V0[idx]);
        }
        
        __global__ void imag_step_x(
                  {% if N >= 2 %}int ySize,{% endif %}
                  {% if N == 3 %}int zSize,{% endif %}
                  float dt,
                  float *x,
                  {% if N >= 2 %} float *y, {% endif %}
                  {% if N == 3 %} float *z, {% endif %}
                  pycuda::complex<float> *Psi_r0,
                  pycuda::complex<float> *V0)
        {
            int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
            {% if N >= 2 %}int yIdx = threadIdx.y + blockIdx.y * blockDim.y; {% endif %}
            {% if N == 3 %}int zIdx = threadIdx.z + blockIdx.z * blockDim.z; {% endif %}
            
            {% if N == 3 %}
            //int idx = xIdx + ySize*(yIdx + zSize*zIdx);
            int idx = zIdx + ySize*(yIdx + zSize*xIdx);
            {% elif N == 2 %}
            int idx = xIdx + ySize*yIdx;
            {% else %}
            int idx = xIdx;
            {% endif %}
            
            Psi_r0[idx] = Psi_r0[idx]*exp(-dt*V0[idx]);
            Psi_r0[idx].imag(0.0f);
        }

        __global__ void step_p(
                    {% if N >= 2 %}int ySize,{% endif %}
                    {% if N == 3 %}int zSize,{% endif %}
                    float dt,
                    float m,
                    float g,
                    float *px,
                    {% if N >= 2 %} float *py, {% endif %}
                    {% if N == 3 %} float *pz, {% endif %}
                    pycuda::complex<float> *Psi_p0)
        {   
            pycuda::complex<float> j(0.0, 1.0);
            int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
            {% if N >= 2 %}int yIdx = threadIdx.y + blockIdx.y * blockDim.y; {% endif %}
            {% if N == 3 %}int zIdx = threadIdx.z + blockIdx.z * blockDim.z; {% endif %}
            {% if N == 3 %}
            //int idx = xIdx + ySize*(yIdx + zSize*zIdx);
            int idx = zIdx + ySize*(yIdx + zSize*xIdx);
            Psi_p0[idx] = Psi_p0[idx]*exp(-j*dt*(((float)0.5/m*(px[xIdx]*px[xIdx]+py[yIdx]*py[yIdx]+pz[zIdx]*pz[zIdx]))+g));
            {% elif N == 2 %}
            int idx = xIdx + ySize*yIdx;
            Psi_p0[idx] = Psi_p0[idx]*exp(-j*dt*(((float)0.5/m*(px[xIdx]*px[xIdx]+py[yIdx]*py[yIdx]))+g));
            {% else %}
            int idx = xIdx;
            Psi_p0[idx] = Psi_p0[idx]*exp(-j*dt*(((float)0.5/m*(px[xIdx]*px[xIdx]))+g));
            {% endif %}
            
        }


        __global__ void imag_step_p(
                    {% if N >= 2 %}int ySize,{% endif %}
                    {% if N == 3 %}int zSize,{% endif %}
                    float dt,
                    float m,
                    float g,
                    float *px,
                    {% if N >= 2 %} float *py, {% endif %}
                    {% if N == 3 %} float *pz, {% endif %}
                    pycuda::complex<float> *Psi_p0)
        {
            int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
            {% if N >= 2 %}int yIdx = threadIdx.y + blockIdx.y * blockDim.y; {% endif %}
            {% if N == 3 %}int zIdx = threadIdx.z + blockIdx.z * blockDim.z; {% endif %}
            {% if N == 3 %}
            //int idx = xIdx + ySize*(yIdx + zSize*zIdx);
            int idx = zIdx + ySize*(yIdx + zSize*xIdx);
            Psi_p0[idx] = Psi_p0[idx]*exp(-dt*(((float)0.5/m*(px[xIdx]*px[xIdx]+py[yIdx]*py[yIdx]+pz[zIdx]*pz[zIdx]))+g));
            {% elif N == 2 %}
            int idx = xIdx + ySize*yIdx;
            Psi_p0[idx] = Psi_p0[idx]*exp(-dt*(((float)0.5/m*(px[xIdx]*px[xIdx]+py[yIdx]*py[yIdx]))+g));
            {% else %}
            int idx = xIdx;
            Psi_p0[idx] = Psi_p0[idx]*exp(-dt*(((float)0.5/m*(px[xIdx]*px[xIdx]))+g));
            {% endif %}
            
            Psi_p0[idx].imag(0.0f);
        }
        """)
        rendered_tpl = tpl.render(N=self._D)
        #print(rendered_tpl)
        mod = SourceModule(rendered_tpl)

        self.propT_R = mod.get_function('step_x')
        self.propT_P = mod.get_function('step_p')
        self.imag_propT_R = mod.get_function('imag_step_x')
        self.imag_propT_P = mod.get_function('imag_step_p')

        self.maxabs_kernel = reduction.ReductionKernel(np.float32, neutral="0.0",
                                                    reduce_expr="fmaxf(a,b)", map_expr="abs(x[i]-y[i])",
                                                    arguments="pycuda::complex<float> *x, pycuda::complex<float> *y",
                                                    preamble="#include <pycuda-complex.hpp>")

        self.meanabs_kernel = reduction.ReductionKernel(np.float32, neutral="0.0",
                           reduce_expr="a+b", map_expr="abs(x[i]-y[i])",
                           arguments="pycuda::complex<float> *x, pycuda::complex<float> *y", preamble="#include <pycuda-complex.hpp>")

        self.norm_kernel = reduction.ReductionKernel(np.float64, neutral="0.0",
                                                    reduce_expr="a+b", map_expr="abs(x[i])",
                                                    arguments="pycuda::complex<float> *x",
                                                    preamble="#include <pycuda-complex.hpp>")


        self.scale_kernel = elementwise.ElementwiseKernel(
            "pycuda::complex<float> *x, float r",
            "x[i] /= r",
            "divide_norm", preamble="#include <pycuda-complex.hpp>")

    def addCAP(self, cutoff=0.8, delta_x = 0.02, k_min=0.05):
        #damp = np.sum([cap(self._grid[i], self._x[i][-1]*cutoff, delta_x, k_min) for i in range(self._D)])
        #damp.imag[damp.imag > 0.0] = 0.0  # hack to get lines away...
        for i in range(self._D):
            self._V0 += np.asarray(cap(self._grid[i], self._x[i][-1]*cutoff, delta_x, k_min), dtype=np.complex64)
        self.gpu_V0 = gpuarray.to_gpu_async(self._V0)

    def setPotential(self, func, **kwargs):
        self._V0 = np.asarray(func(*self._grid, **kwargs), dtype=np.complex64)
        self.gpu_V0 = gpuarray.to_gpu_async(self._V0)

    def addPotential(self, func, **kwargs):
        self._V0 += np.asarray(func(*self._grid, **kwargs), dtype=np.complex64)
        self.gpu_V0 = gpuarray.to_gpu_async(self._V0)

    def positivePotential(self):
        zero = np.min(self._V0)
        print(zero)
        self._V0 -= zero

    def setWavefunction(self, func, **kwargs):
        self._Psi_r0 = np.asarray(func(*self._grid, **kwargs), dtype=np.complex64)

    #########
    # getter
    #########
    def getPsiR0(self):
        return self._Psi_r0.real**2+self._Psi_r0.imag**2

    def getPsiGPU(self):
        psi = self.gpu_Psi0.get()
        return psi.real


    def getPsiP0(self):
        Psi0_temp = np.fft.fftshift(self._Psi_p0)
        return Psi0_temp.real**2+Psi0_temp.imag**2


    def getRawPsiR0(self):
        return self._Psi_r0

    def getRawPsiP0(self):
        Psi0_temp = np.fft.fftshift(self._Psi_p0)
        return Psi0_temp

    def getRawPsiP1(self):
        Psi1_temp = np.fft.fftshift(self._Psi_p1)
        return Psi1_temp

    def getNormPsiR0(self):
        #return  (trapz(trapz(trapz(self._Psi_r0 * self._Psi_r0.conjugate())))*np.prod(self._dx)).real
        return self._norm_Psi_r0

    def getNormPsiP0(self):
        #return  (trapz(trapz(trapz(self._Psi_p0 * self._Psi_p0.conjugate())))*np.prod(self._dp)).real
        return self._norm_Psi_p0

    def getDx(self):
        """
        Returns spatial grid X spacing.
        Return type: float
        """
        return self._dx


    def getDp(self):
        """
        Returns the momentum grid X spacing.
        Return type: float
        """
        return self._dp


    def getXGrid(self):
        """
        Returns the X space grid.
        Return type: array (float)
        """
        return self._x


    def getPGrid(self):
        """
        Returns the momentum X grid.
        Return type: array (float)
        """
        return self._p

    def getxExtent(self):
        return [(self._x[i][0], self._x[i][-1]) for i in range(self._D)]

    def getpExtent(self):
        return [(np.min(self._p[i]), np.max(self._p[i])) for i in range(self._D)]

    def getT(self):
        """
        Returns the total propagation time: t_TotalPropagation = tNow - tStart
        Return type: float
        """
        return self._t


    def getDt(self):
        """
        Returns the time step size used for time propagation.
        Return type: float
        """
        return self._dt

    # and setter
    def setDt(self, dt):
        """
        Set the time step size that is used for the propagation.
        Type to set: float
        """
        self._dt = dt

    def setV0(self, v):
        self._V0 = np.asarray(v, dtype=np.complex64)
        self.gpu_V0.set(self._V0)

