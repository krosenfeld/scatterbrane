'''
Compare the ensemble average image to the average image.
'''

import numpy as np
from scipy.ndimage import imread
import time
import matplotlib.pyplot as plt
from palettable.cubehelix import jim_special_16
from palettable.colorbrewer.qualitative import Dark2_5
cmap = jim_special_16.mpl_colormap
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['axes.color_cycle'] = Dark2_5.mpl_colors

from scatterbrane import Brane,utilities

# set up logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# we will use a ring source structure
def source_model(nx,dx,radius,width):
    sigma = width / (2 * np.sqrt(np.log(4))) / dx
    x = np.dot(np.transpose([np.ones(nx)]),[np.arange(nx)-nx/2])
    r = np.sqrt(x**2 + x.T**2)
    m = np.exp(-0.5/sigma**2*(r-radius/dx)**2)
    return m/m.sum()
nx = 256
dx = 0.75
m = source_model(nx,dx,28.,10.)

# initialize the scattering class
b = Brane(m,dx,wavelength=1.3e-3,nphi=(2**12,2**12),screen_res=6,r_inner=50)

# points along the uv axes where to record the visibility
u = np.linspace(0.,10e9,num=50)

# generate a bunch of scattering instances
tic = time.time()
u_vis = []
v_vis = []
num_sims = 100
avg = np.zeros_like(b.isrc)
for i in range(num_sims):
    # create a new instance of the random phases
    b.generatePhases()
    # calculate the scattered image
    b.scatter()
    # keep track of the average
    avg += b.iss
    # calculate the visibility function along the u and v axis.
    u_vis.append(np.array([utilities.FTElementFast(b.iss,b.dx,[u_,0]) for u_ in u]))
    v_vis.append(np.array([utilities.FTElementFast(b.iss,b.dx,[0,u_]) for u_ in u]))
avg /= num_sims
logger.info('took {0:g}s'.format(time.time()-tic))

# make figures
write_figs = False
fig_file = '../_static/ensemble_average/'
extent=b.dx*b.nx//2*np.array([1,-1,-1,1])
ensemble = utilities.ensembleSmooth(b.isrc,b.dx,b)
u_kernel = utilities.getUVKernel(u,np.zeros_like(u),b)
v_kernel = utilities.getUVKernel(np.zeros_like(u),u,b)

# show ensemble average
plt.figure()
plt.subplot(121)
plt.imshow(ensemble,cmap=cmap,extent=extent)
plt.xlim([70,-70]); plt.ylim([-70,70])
plt.xlabel('$\Delta\\alpha$ [$\mu$as]'); plt.ylabel('$\Delta\delta$ [$\mu$as]')
plt.subplot(122)
plt.imshow(avg,cmap=cmap,extent=extent)
plt.xlim([70,-70]); plt.ylim([-70,70]); plt.gca().set_yticklabels(10*[''])
if write_figs: plt.savefig(fig_file+'/avg_image.png',bbox_inches='tight')

# show how the averages converge to the ensemble average
u_src = np.array([utilities.FTElementFast(b.isrc,b.dx,[u_,0]) for u_ in u])
v_src = np.array([utilities.FTElementFast(b.isrc,b.dx,[0,u_]) for u_ in u])
u_ens = np.array([utilities.FTElementFast(ensemble,b.dx,[u_,0]) for u_ in u])
v_ens = np.array([utilities.FTElementFast(ensemble,b.dx,[0,u_]) for u_ in u])
u_vis = np.array(u_vis)
v_vis = np.array(v_vis)

nshow = 5

# amplitudes

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.gca().fill_between(u/1e9,0,u_kernel,color=3*(0.6,),alpha=0.3)
plt.plot(u/1e9,np.abs(u_vis.T[:,:nshow]),color='#377EB8')
plt.plot(u/1e9,np.abs(u_ens),'-.k',lw=2)
plt.plot(u/1e9,np.abs(u_src),'--r',lw=2)
plt.xlabel('u [G$\lambda$]'); plt.ylabel('Amplitude')
ylim = plt.gca().get_ylim()
plt.subplot(122)
plt.gca().fill_between(u/1e9,0,v_kernel,color=3*(0.6,),alpha=0.3)
plt.plot(u/1e9,np.abs(v_vis.T[:,:nshow]),color='#377EB8')
plt.plot(u/1e9,np.abs(v_ens),'-.k',lw=2)
plt.plot(u/1e9,np.abs(v_src),'--r',lw=2)
plt.xlabel('v [G$\lambda$]'); plt.gca().set_yticklabels(10*['']); plt.ylim(ylim)
if write_figs: plt.savefig(fig_file+'/vis_profiles.png',bbox_inches='tight')

# show effect of deblurring
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(u/1e9,np.abs(u_vis.T[:,:nshow]/u_kernel[:,np.newaxis]),color='#377EB8')
plt.gca().fill_between(u/1e9,0,u_kernel,color=3*(0.6,),alpha=0.3)
plt.plot(u/1e9,np.abs(u_ens/u_kernel),'k',lw=2)
plt.plot(u/1e9,np.abs(u_ens),'-.k',lw=2)
plt.plot(u/1e9,np.abs(u_src),'--r',lw=2)
plt.xlabel('u [G$\lambda$]'); plt.ylabel('Amplitude')
ylim = plt.gca().get_ylim()
plt.subplot(122)
plt.plot(u/1e9,np.abs(v_vis.T[:,:nshow]/v_kernel[:,np.newaxis]),color='#377EB8')
plt.gca().fill_between(u/1e9,0,v_kernel,color=3*(0.6,),alpha=0.3)
plt.plot(u/1e9,np.abs(v_ens/v_kernel),'k',lw=2)
plt.plot(u/1e9,np.abs(v_ens),'-.k',lw=2)
plt.plot(u/1e9,np.abs(v_src),'--r',lw=2)
plt.xlabel('v [G$\lambda$]'); plt.gca().set_yticklabels(10*['']); plt.ylim(ylim)
if write_figs: plt.savefig(fig_file+'/vis_profiles_deblur.png',bbox_inches='tight')

# show mse for various averaging options
plt.figure(figsize=(12,6))
plt.subplot(121)
ns = [1,2,5,25]
ylim = [0,0.2]
for n in ns: 
    foo = u_vis.reshape(num_sims/n,n,-1).mean(axis=1)/u_kernel - u_src
    plt.plot(u/1e9,np.sqrt(np.mean(foo*foo.conj(),axis=0)),lw=2,label='N={0:d}'.format(n))
plt.xlabel('u [G$\lambda$]'); plt.ylabel('RMS refractive noise')
plt.ylim(ylim)
plt.subplot(122)
for n in ns: 
    foo = v_vis.reshape(num_sims/n,n,-1).mean(axis=1)/v_kernel - v_src
    plt.plot(u/1e9,np.sqrt(np.mean(foo*foo.conj(),axis=0)),lw=2,label='N={0:d}'.format(n))
plt.xlabel('v [G$\lambda$]'); plt.gca().set_yticklabels(10*[''])
plt.legend(frameon=False)
plt.ylim(ylim)
if write_figs: plt.savefig(fig_file+'/vis_rms_deblur.png',bbox_inches='tight')


plt.show()


