'''
Generate a time series incorporating the motion of the screen across the source.
This script may take a long time to run. I suggest you read through it first and
adjust the num_samples variable to check out its performance.
'''
import numpy as np
from scipy.ndimage import imread
import time
import matplotlib.pyplot as plt
from palettable.cubehelix import jim_special_16
cmap = jim_special_16.mpl_colormap
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['patch.edgecolor'] = 'white'
plt.rcParams['lines.linewidth'] = 2

from scatterbrane import Brane,utilities

# set up logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# import our source image and covert it to gray scale
src_file = 'source_images/nh_01_stern_05_pluto_hazenew2.square.jpg'
rgb = imread(src_file)[::-1]
I = (np.array([0.2989,0.5870,0.1140])[np.newaxis,np.newaxis,:]*rgb).sum(axis=-1)
I *= np.pi/I.sum()

# make up some scale for our image.
write_figs = False
wavelength=1e-3
FOV = 90.
dx = FOV/I.shape[0]

# initialize the scattering screen @ 0.87mm
b = Brane(I,dx,wavelength=0.87e-3,nphi=(2**12,2**14),anisotropy=1,pa=None,r_inner=50,live_dangerously=True)

# estimate the time resolution of our simulation assuming some screen velocity.
screen_velocity = 200.  #km/s
fs = screen_velocity/(b.screen_dx*b.ips) # Hz
num_samples = b.nphi[1]/b.ips - b.nx   # try num_samples = 100 for testing porpoises.
logger.info('Number of samples: {0:g}'.format(num_samples))
logger.info('Sampling interval: {0:g}s'.format(1./fs))
logger.info('Time coverage: {0:g} days'.format(num_samples/fs/(3600.*24.)))

# generate the screen (this takes a while)
logger.info('generating screen...')
tic = time.time()
b.generatePhases()
logger.info('took {0:g}s'.format(time.time()-tic))

# generate time series (this takes a while)
logger.info('generating time series...')
fluxes = []
frames = []
tic = time.time()
for i in range(num_samples):
    # update source image to include a sinusoidal flux modulation 
    b.setModel(I*(1. - 0.4*np.sin(2*np.pi*i/(2*num_samples))), dx) # comment out to speedup 
    b.scatter(move_pix=i*b.ips)
    fluxes.append(b.iss.sum())
    frames.append(b.iss)
logger.info('took {0:g}s'.format(time.time()-tic))
# 1962.92s

# make figures
fig_file = '../_static/time_variability/'
extent=b.dx*b.nx//2*np.array([1,-1,-1,1])

plt.figure()
plt.subplot(121)
isrc_smooth = utilities.smoothImage(b.isrc,b.dx,2.*b.dx)
plt.imshow(isrc_smooth,extent=extent,cmap=cmap)
plt.xlabel('$\Delta\\alpha$ [$\mu$as]'); plt.ylabel('$\Delta\delta$ [$\mu$as]')
plt.subplot(122)
iss_smooth = utilities.smoothImage(b.iss,b.dx,2.*b.dx)
plt.imshow(iss_smooth,extent=extent,cmap=cmap)
plt.gca().set_yticklabels(10*['']); plt.gca().set_xticklabels(10*[''])
if write_figs: plt.savefig(fig_file+'/iss.png',bbox_inches='tight')

plt.figure()
t = 1./fs*np.arange(len(fluxes))/3600.
plt.plot(t,fluxes,color='#377EB8')
plt.xlabel('time [hr]')
plt.ylabel('flux [Jy]')
plt.xlim([0,t.max()])
plt.grid()
if write_figs: plt.savefig(fig_file+'/flux.png',bbox_inches='tight')

# and a movie
import matplotlib.animation as animation
i = 0
def updatefig(*args):
    global i
    i = (i + 1) % num_samples
    im.set_array(utilities.smoothImage(frames[i],b.dx,2*b.dx))
    return im

plt.show()

fig = plt.figure(figsize=(8,6))
im = plt.imshow(utilities.smoothImage(frames[0],b.dx,2*b.dx), cmap=cmap, animated=True, 
    extent=extent, interpolation=None)
plt.xlabel('$\Delta\\alpha$ [$\mu$as]')
plt.ylabel('$\Delta\delta$ [$\mu$as]')
ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=False, frames=int(1000))
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Katherine Rosenfeld'), bitrate=1800)
if write_figs: 
    logger.info('writing movie!')
    ani.save('mov.mp4',writer=writer)
    plt.close()
else:
    plt.show()
