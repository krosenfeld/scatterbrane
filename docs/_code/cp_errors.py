'''
Estimate rms of closure phase error due to scattering.
'''

import numpy as np
from scipy.ndimage import imread
import time
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Dark2_5
plt.rcParams['image.cmap'] = 'gray_r'
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['patch.edgecolor'] = 'white'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.color_cycle'] = Dark2_5.mpl_colors

from scatterbrane import Brane,Target,utilities

# set up logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# import our source image and covert it to gray scale
src_file = 'source_images/640px-Bethmaennchen2.square.jpg'
rgb = imread(src_file)[::-1]
I = 255 - (np.array([0.2989,0.5870,0.1140])[np.newaxis,np.newaxis,:]*rgb).sum(axis=-1)
I *= np.pi/I.sum()

# and smooth image
I = utilities.smoothImage(I,1.,8.)

# make up some scale for our image. Let's have it span 120 uas.
write_figs = False
wavelength=1e-3
FOV = 120.
dx = FOV/I.shape[0]

# initialize the scattering screen @ 1mm
b = Brane(I,dx,wavelength=1.e-3,nphi=2**13)

# initialize Target object for calculating visibilities and uv samples
sgra = Target(wavelength=1.e-3)

# calculate uv samples for our closure phase triangle and on the source image
site_triangle = ["SMT","ALMA","LMT"]
(uv,hr) = sgra.generateTriangleTracks(sgra.sites(site_triangle),times=(0.,24.,int(24.*60./2.)),return_hour=True)
CP_src = sgra.calculateCP(b.isrc,b.dx,uv)

# calculate CP on these triangles for a number of screens
def one_sim(i):
    global sgra,b,CP
    b.generatePhases()
    b.scatter()
    fluxes.append(b.iss.sum())
    CP.append(sgra.calculateCP(b.iss,b.dx,uv))

# initialize arrays, we will save total flux as well
CP = []
fluxes = []
# estimate time for one simulation
tic = time.time()
one_sim(0)
# run enough to take about 10 minutes
num_sim = int(10*60. / (time.time()-tic))
logger.info('running {0:g} simulations'.format(num_sim))
tic = time.time()
for i in xrange(1,num_sim):
    one_sim(i)
CP = np.asarray(CP)
print 'took {0:g}s'.format(time.time()-tic) # 140s on my machine

# calculate rms of cp
wrapdiff = lambda angle1,angle2: np.angle(np.exp(1j*angle1)/np.exp(1j*angle2))
rms_CP = np.sqrt(np.mean((wrapdiff(CP,CP_src[np.newaxis,:]))**2,axis=0))

# generate figures
fig_file = '../_static/cp_errors/'

# source structure
plt.figure()
extent=b.dx*b.nx//2*np.array([1,-1,-1,1])
plt.subplot(121)
plt.imshow(b.isrc,extent=extent)
plt.xlabel('l [$\mu$as]'); plt.ylabel('m [$\mu$as]');
plt.subplot(122)
plt.imshow(utilities.smoothImage(b.isrc,b.dx,10.),extent=extent)
plt.gca().set_yticklabels(10*['']); plt.gca().set_xticklabels(10*[''])
if write_figs: plt.savefig(fig_file+'src.png',bbox_inches='tight')

plt.figure()
plt.scatter(uv[:,:,0]/1e9,uv[:,:,1]/1e9,lw=0)
plt.scatter([0],[0],s=150,marker='+',lw=2,c='k')
plt.gca().set_aspect('equal')
plt.xlim([9,-9]); plt.ylim([-9,9])
plt.xlabel('u [G$\lambda$]'); plt.ylabel('v [G$\lambda$]');
if write_figs: plt.savefig(fig_file+'uv_track.png',bbox_inches='tight')

plt.figure()
extent=b.dx*b.nx//2*np.array([1,-1,-1,1])
plt.subplot(121)
plt.imshow(b.iss,extent=extent)
plt.xlabel('l [$\mu$as]'); plt.ylabel('m [$\mu$as]');
plt.subplot(122)
iss_smooth = utilities.smoothImage(b.iss,b.dx,2.)
plt.imshow(iss_smooth,extent=extent)
plt.contour(np.linspace(extent[0],extent[1],num=b.nx),\
            np.linspace(extent[2],extent[3],num=b.nx),\
            iss_smooth/iss_smooth.max(),np.linspace(0.2,0.9,num=4),\
            colors=[(55/256.,126/256.,184/256.)],lw=1)
plt.gca().set_yticklabels(10*['']); plt.gca().set_xticklabels(10*[''])
if write_figs: plt.savefig(fig_file+'iss.png',bbox_inches='tight')

plt.figure()
gst = sgra._gst(hr)
ii = np.argsort(gst)
plt.plot(gst[ii],np.degrees(rms_CP[ii]))
plt.xlabel('time [hr]'); plt.ylabel('rms closure phase error [deg]')
plt.grid()
plt.ylim([0,np.degrees(rms_CP).max()*1.1])
if write_figs: plt.savefig(fig_file+'cp.png',bbox_inches='tight')

plt.show()
