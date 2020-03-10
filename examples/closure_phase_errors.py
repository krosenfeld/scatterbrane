from scatterbrane import Brane,Target,utilities
import numpy as np
import time
from imageio import imread
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray_r'
plt.rcParams['image.origin'] = 'lower'

# set up logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# import our source image and covert it to gray scale
src_file = 'bethmaennchen.jpg'
rgb = imread(src_file)[::-1]
I = 255 - (np.array([0.2989,0.5870,0.1140])[np.newaxis,np.newaxis,:]*rgb).sum(axis=-1)
I = I*np.pi/I.sum()

# make I square
tmp = np.min(I.shape)
I = I[:tmp,:tmp]

# and smooth image
I = utilities.smoothImage(I,1.,8.)

# make up some scale for our image. Let's have it span 120 uas.
wavelength=1e-3
FOV = 120.
dx = FOV/I.shape[0]

# initialize the scattering screen @ 1mm
b = Brane(I,dx,wavelength=1.e-3,nphi=2**13)

# initialize Target object for calculating visibilities and uv samples
sgra = Target(wavelength=1.e-3)
# calculate uv samples for our closure phase triangle and on the source image
site_triangle = ["SMA","ALMA","LMT"]
(uv,hr) = sgra.generateTriangleTracks(sgra.sites(site_triangle),times=(0.,24.,int(24.*60./2.)),return_hour=True)

# closure phases for the source image
CP_src = sgra.calculateCP(b.isrc,b.dx,uv)

def one_sim(i):
    ''' helper function to run one simulation '''
    global sgra,b,CP
    b.generatePhases()
    b.scatter()
    CP.append(sgra.calculateCP(b.iss,b.dx,uv))

# initialize list
CP = []

# estimate time for one simulation
tic = time.time()
one_sim(0)

# run enough to take about 10 minutes
num_sim = int(10*60. / (time.time()-tic))
logger.info('running {0:g} simulations'.format(num_sim))
tic = time.time()
for i in range(1,num_sim):
    one_sim(i)
CP = np.asarray(CP)
logger.info('took {0:g}s'.format(time.time()-tic))

# calculate rms error of closure phase array (CP)
wrapdiff = lambda angle1,angle2: np.angle(np.exp(1j*angle1)/np.exp(1j*angle2))
rms_CP = np.sqrt(np.mean((wrapdiff(CP,CP_src[np.newaxis,:]))**2,axis=0))
