"""
.. module:: brane 
    :platform: Unix
    :synopsis: Simulate effect of anisotropic scattering.

.. moduleauthor:: Katherine Rosenfeld <krosenf@gmail.com>
.. moduleauthor:: Michael Johnson

Default settings are appropriate for Sgr A*

Resources:
Bower et al. (2004, 2006)

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom,rotate
from numpy import (pi,sqrt,log,sin,cos,exp,ceil,arange,
                  min,abs,ones,radians,dot,transpose,
                  zeros_like,clip,empty,empty,empty_like,reshape)
from numpy.fft import fft2,fftfreq
from astropy.constants import au,pc
from astropy import units
import logging
from scatterbrane import utilities

__all__ = ["Brane"]

class Brane(object):
  """
  Scattering simulation object.

  :param model: ``(n, n)``
    Numpy array of the source image.
  :param dx: scalar 
    Resolution element of model in uas.
  :param nphi: (optional) ``(2, )`` or scalar
    Number of pixels in a screen.  This may be a tuple specifying the dimensions of a rectangular screen.
  :param screen_res: (optional) scalar
    Size of a screen pixel in units of :math:`R_0`.
  :param wavelength: (optional) scalar
    Observing wavelength in meters.
  :param dpc: (optional) scalar
    Observer-Source distance in parsecs.
  :param rpc: (optional) scalar
    Observer-Scatterer distance in parsecs.
  :param r0: (optional) scalar
    Phase coherence length along major axis as preset string or in km.
  :param r_outer: (optional) scalar
    Outer scale of turbulence in units of :math:`R_0`.
  :param r_inner: (optional) scalar
    Inner scale of turbulence in units of :math:`R_0`.
  :param alpha: (optional) string or scalar
    Preset string or float to set power-law index for tubulence scaling (e.g., Kolmogorov has :math:`\\alpha= 5/3`)
  :param anisotropy: (optional) scalar
    Anisotropy for screen is the EW / NS elongation.
  :param pa: (optional) scalar
    Orientation of kernel's major axis, East of North, in degrees.
  :param live_dangerously: (optional) bool
    Skip the parameter checks?
  :param think_positive: (optional) bool
    Should we enforce that the source image has no negative pixel values?

  :returns: An instance of a scattering simulation.

  :Example:

  .. code-block:: python

    s = Brane(m,dx,nphi=2**12,screen_res=5.,wavelength=3.0e-3,dpc=8.4e3,rpc=5.8e3)

  where ``s`` is the class instance, ``m`` is the image array, ``nphi`` is the number of screen pixels,
  ``wavelength`` is the observing wavelength.

  .. note:: :math:`R_0` is the phase coherence length and Sgr A* defaults are from Bower et al. (2006).
  """

  def __init__(self,model,dx,nphi=2**12,screen_res=2,\
               wavelength=1.3e-3,dpc=8400,rpc=5800,r0 = 'sgra',\
               r_outer=10000000,r_inner=12,alpha='kolmogorov',\
               anisotropy=2.045,pa=78,match_screen_res=False,live_dangerously=False,think_positive=False):

    # set initial members
    self.logger = logging.getLogger(self.__class__.__name__)
    self.live_dangerously = live_dangerously
    self.think_positive = think_positive

    self.wavelength = wavelength*1e-3          # observing wavelength in km
    self.dpc = float(dpc)                      # earth-source distance in pc
    self.rpc = float(rpc)                      # R: source-scatterer distance in pc
    self.d   = self.dpc - self.rpc             # D: earth-scatterer distance in pc
    self.m   = self.d/self.rpc                 # magnification factor (D/R)
    if r0 == 'sgra':
        # major axis (EW) phase coherence length in km
        self.r0  = 3136.67*(1.3e-6/self.wavelength)
    else:
        try:
            self.r0 = float(r0)
        except:
            raise ValueError('Bad value for r0')
    self.anisotropy = anisotropy               # anisotropy for screen = (EW / NS elongation)
    self.pa = pa                               # orientation of major axis, E of N (or CCW of +y)

    # Fresnel scale in km
    self.rf = sqrt(self.dpc*pc.to(units.km).value / (2*pi / self.wavelength) * self.m / (1+self.m)**2)

    # compute pixel scale for image
    if match_screen_res:
      self.ips = 1
      self.screen_dx = screen_res * self.r0
    else:
      self.screen_dx = screen_res * self.r0                     # size of a screen pixel in km   
      self.ips = int(ceil(1e-6*dx*self.d*au.to(units.km).value/self.screen_dx)) # image pixel / screen pixel

    # image arrays
    self.dx = 1e6 * self.ips * (self.screen_dx / (au.to(units.km).value * self.d))  # image pixel scale in uas
    self.nx = int(ceil(model.shape[-1] * dx / self.dx))          # number of image pixels
    self.model = model                                           # source model
    self.model_dx = dx                                           # source model resolution
    self.iss  = np.array([],dtype=np.float64)                    # scattered image
    self.isrc = np.array([],dtype=np.float64)                    # source image at same scale as scattered image

    # screen parameters
    if type(nphi) == int:
        self.nphi = (nphi,nphi)                                  # size of screen array
    else:
        self.nphi = nphi
    self.nphi = np.asarray(self.nphi)
    self.r_inner = r_inner                                       # inner turbulence scale in r0
    self.r_outer = r_outer                                       # outer turbulence scale in r0
    #self.qmax = 1.*screen_res/r_inner                            # 1 / inner turbulence scale in pix
    #self.qmin = 1.*screen_res/r_outer                            # 1 / outer tubulence scale in pix
    if alpha == 'kolmogorov':
        self.alpha = 5./3
    else:
        try:
            self.alpha = float(alpha)
        except:
            raise ValueError('Bad value for alpha')

    # use logger to report
    self.chatter()

    # includes sanity check
    self.setModel(self.model,self.model_dx,think_positive=self.think_positive)

  def _checkSanity(self):
    '''
    Check that screen parameters are consistent.
    '''

    # sanity check: is screen large enough?
    assert np.ceil(self.nx*self.ips)+2 < np.min(self.nphi), \
          "screen is not large enough: {0} > {1}".\
          format(int(np.ceil(self.ips*self.nx)+2),np.min(self.nphi))

    # sanity check: is image square?
    assert self.model.shape[-1] == self.model.shape[-2], \
        'source image must be square'

    # sanity check: integer number of screen pixels / image pixel?
    assert self.ips % 1 == 0, 'image/screen pixels should be an integer'

    # is inner turbulence scale larger than r0?
    #assert 1./self.qmax > self.r0/self.screen_dx, 'turbulence inner scale < r0'
    assert self.r_inner > 1., 'turbulence inner scale < r0'

    # check image smoothness
    V = fft2(self.isrc)
    freq = fftfreq(self.nx,d=self.dx*radians(1.)/(3600*1e6))
    u = dot(transpose([np.ones(self.nx)]),[freq])
    v = dot(transpose([freq]),[ones(self.nx)])
    try:
        if max(abs(V[sqrt(u*u+v*v) > (1.+self.m)*self.r_inner*self.r0/self.wavelength])) / self.isrc.sum() > 0.01:
         self.logger.warning('image is not smooth enough:  {0:g} > 0.01'.format(max(abs(V[sqrt(u*u+v*v) > (1.+self.m)*self.r_inner*self.r0/self.wavelength])) / self.isrc.sum()))
    except ValueError:
        self.logger.warning('r_inner is too large to test smoothness')

    # is screen pixel smaller than inner turbulence scale?
    #assert 1./self.qmax >= 1, 'screen pixel > turbulence inner scale'
    assert self.r_inner*self.r0/self.screen_dx >= 1, 'screen pixel > turbulence inner scale'

    if (self.rf*self.rf/self.r0/(self.ips*self.screen_dx) < 3):
      self.logger.warning('WARNING: image resolution is approaching Refractive scale')

  def chatter(self):
    '''
    Print information about the current scattering simulation where many parameters are cast as integers.
    '''
        
    fmt = "{0:32s} :: "
    self.logger.info( (fmt + "{1:g}").format('Observing wavelength [mm]',1e6*self.wavelength) )
    self.logger.info( (fmt + "{1:d}").format('Phase coherence length [km]',int(self.r0))           )
    self.logger.info( (fmt + "{1:d}").format('Fresnel scale [km]',int(self.rf))                    )
    self.logger.info( (fmt + "{1:d}").format('Refractive scale [km]',int(self.rf**2/self.r0))     )
    #self.logger.info( (fmt + "{1:d}").format('Inner turbulence scale [km]',int(self.screen_dx/self.qmax)))
    self.logger.info( (fmt + "{1:d}").format('Inner turbulence scale [km]',int(self.r_inner*self.r0)))
    self.logger.info( (fmt + "{1:d}").format('Screen resolution [km]',int(self.screen_dx)))
    self.logger.info( (fmt + "{1:d} {2:d}").format('Linear filling factor [%,%]',*map(int,100.*self.nx*self.ips/self.nphi)) )
    self.logger.info( (fmt + "{1:g}").format('Image resolution [uas]',self.dx))
    self.logger.info( (fmt + "{1:d}").format('Image size',int(self.nx)))

  def _generateEmptyScreen(self):
    '''
    Create an empty screen.
    '''
    if not self.live_dangerously: self._checkSanity()
    self.phi = np.zeros(self.nphi)

  def setModel(self,model,dx,think_positive=False):
    '''
    Set new model for the source.

    :param model: ``(n, n)``
      Numpy image array.
    :param dx: scalar
      Pixel size in microarcseconds.
    :param think_positive: (optional) bool
        Should we enforce that the source image has no negative pixel values?
    '''
    self.nx = int(ceil(model.shape[-1] * dx / self.dx))          # number of image pixels
    self.model = model                                           # source model
    self.model_dx = dx                                           # source model resolution

    # load source image that has size and resolution compatible with the screen.
    self.isrc = np.empty(2*(self.nx,))
    self.think_positive = think_positive

    M = self.model.shape[1]       # size of original image array
    f_img = RectBivariateSpline(self.model_dx/self.dx*(np.arange(M) - 0.5*(M-1)),\
                                self.model_dx/self.dx*(np.arange(M) - 0.5*(M-1)),\
                                self.model)

    xx_,yy_ = np.meshgrid((np.arange(self.nx) - 0.5*(self.nx-1)),\
                          (np.arange(self.nx) - 0.5*(self.nx-1)),indexing='xy')

      
    m  = f_img.ev(yy_.flatten(),xx_.flatten()).reshape(2*(self.nx,))
    self.isrc  = m * (self.dx/self.model_dx)**2     # rescale for change in pixel size

    if self.think_positive:
      self.isrc[self.isrc < 0] = 0

    if not self.live_dangerously: self._checkSanity()

  def generatePhases(self,seed=None,save_phi=None):
    '''
    Generate screen phases.

    :param seed: (optional) scalar
      Seed for random number generator
    :param save_phi: (optional) string 
      To save the screen, set this to the filename.
    '''

    # check setup
    if not self.live_dangerously: self._checkSanity()

    # seed the generator
    if seed != None:
      np.random.seed(seed=seed)

    # include anisotropy
    qx2 = dot(transpose([np.ones(self.nphi[0])]),[np.fft.rfftfreq(self.nphi[1])**2])
    qy2 = dot(transpose([np.fft.fftfreq(self.nphi[0])**2*self.anisotropy**2]),[np.ones(self.nphi[1]//2+1)])
    rr = qx2+qy2
    rr[0,0] = 0.02  # arbitrary normalization

    # generating phases with given power spectrum
    size = rr.shape 
    qmax2 = (self.r_inner*self.r0/self.screen_dx)**-2
    qmin2 = (self.r_outer*self.r0/self.screen_dx)**-2
    phi_t = (1/sqrt(2) * sqrt(exp(-1./qmax2*rr) * (rr + qmin2)**(-0.5*(self.alpha+2.)))) \
            * (np.random.normal(size=size) + 1j * np.random.normal(size=size))

    # calculate phi
    self.phi = np.fft.irfft2(phi_t)

    # normalize structure function 
    nrm = self.screen_dx/(self.r0*sqrt(self._getPhi(1,0)))
    self.phi *= nrm

    # save screen
    if save_phi != None:
      np.save(save_phi,self.phi)

  def _checkDPhi(self,nLag=5):
    '''
    Report the phase structure function for various lags.

    :param nLag: (optional) int
      Number of lags to report starting with 0.
    '''
    self.logger.info( "\nEstimates of the phase structure function at various lags:")
    for i in np.arange(nLag):
      self.logger.info( "lag ",i, self._getPhi(i,0), self._getPhi(0,i), self._getPhi(i,i))

  def _getPhi(self,lag_x,lag_y):
    '''
    Empirical estimate for phase structure function

    :param lag_x: int 
      Screen pixels to lag in x direction.
    :param lag_y: int
      Screen pixels to lag in y direction.
    '''
    assert (lag_x < self.nphi[0]) and (lag_x < self.nphi[1]), "lag choice larger than screen array"
    # Y,X
    if (lag_x == 0 and lag_y == 0):
      return 0.
    if (lag_x == 0):
      return 1.*((self.phi[:-1*lag_y,:] - self.phi[lag_y:,:])**2).sum()/(self.nphi[0]*(self.nphi[1]-lag_y))
    if (lag_y == 0):
      return 1.*((self.phi[:,:-1*lag_x] - self.phi[:,lag_x:])**2).sum()/((self.nphi[0]-lag_x)*self.nphi[1])
    else:
      return (1.*((self.phi[:-1*lag_y,:-1*lag_x] - self.phi[lag_y:,lag_x:])**2).sum()/((self.nphi[1]-lag_y)*(self.nphi[0]-lag_x)))

  def readScreen(self,filename):
    '''
    Read in screen phases from a file.

    :param filename: string 
      File containing the screen phases.
    '''
    self.phi =  np.fromfile(filename,dtype=np.float64).reshape(self.nphi)
  
  def _calculate_dphi(self,move_pix=0):
    '''
    Calculate the screen gradient.

    :param move_pix: (optional) int 
      Number of pixels to roll the screen (for time evolution).

    :returns: ``(nx, nx)``, ``(nx, nx)`` 
      numpy arrays containing the dx,dy components of the gradient vector.

    .. note:: Includes factors of the Fresnel scale and the result is in units of the source image.
    '''
    ips = self.ips                  # number of screen pixels per image pixel
                                    # -- when this != 1, some sinusoidal signal
                                    # over time with period of image_resolution
    nx  = self.nx                   # number of image pixels
    rf  = self.rf / self.screen_dx  # Fresnel scale in screen pixels
    assert move_pix < (self.nphi[1] - self.nx*self.ips), 'screen is not large enough'
    dphi_x = (0.5 * rf * rf / ips ) * \
            (self.phi[0:ips*nx:ips,2+move_pix:ips*nx+2+move_pix:ips] -
             self.phi[0:ips*nx:ips,0+move_pix:ips*nx+move_pix  :ips])
    dphi_y = (0.5 * rf * rf  / ips ) * \
            (self.phi[2:ips*nx+2:ips,0+move_pix:ips*nx+move_pix:ips] -
             self.phi[0:ips*nx  :ips,0+move_pix:ips*nx+move_pix:ips])

    self.logger.debug('{0:d},{1:d}'.format(*dphi_x.shape))
    return dphi_x,dphi_y
  
  def scatter(self,move_pix=0,scale=1):
    '''
    Generate the scattered image which is stored in the ``iss`` member.

    :param move_pix: (optional) int 
      Number of pixels to roll the screen (for time evolution).
    :param scale: (optional) scalar
      Scale factor for gradient.  To simulate the scattering effect at another 
      wavelength this is (lambda_new/lambda_old)**2
    '''

    M = self.model.shape[-1]       # size of original image array
    N = self.nx                    # size of output image array

    #if not self.live_dangerously: self._checkSanity()

    # calculate phase gradient
    dphi_x,dphi_y = self._calculate_dphi(move_pix=move_pix)

    if scale != 1:
        dphi_x *= scale
        dphi_y *= scale

    xx_,yy = np.meshgrid((np.arange(N) - 0.5*(N-1)),\
                         (np.arange(N) - 0.5*(N-1)),indexing='xy')

    # check whether we care about PA of scattering kernel
    if self.pa != None:
      f_model = RectBivariateSpline(self.model_dx/self.dx*(np.arange(M) - 0.5*(M-1)),\
                                    self.model_dx/self.dx*(np.arange(M) - 0.5*(M-1)),\
                                    self.model)

      # apply rotation
      theta = -(90 * pi / 180) + np.radians(self.pa)     # rotate CW 90 deg, then CCW by PA
      xx_ += dphi_x
      yy  += dphi_y
      xx = cos(theta)*xx_ - sin(theta)*yy
      yy = sin(theta)*xx_ + cos(theta)*yy
      self.iss  = f_model.ev(yy.flatten(),xx.flatten()).reshape((self.nx,self.nx))

      # rotate back and clip for positive values for I
      if self.think_positive:
          self.iss  = clip(rotate(self.iss,-1*theta/np.pi*180,reshape=False),a_min=0,a_max=1e30) * (self.dx/self.model_dx)**2
      else:
          self.iss  = rotate(self.iss,-1*theta/np.pi*180,reshape=False) * (self.dx/self.model_dx)**2

    # otherwise do a faster lookup rather than the expensive interpolation.
    else:
      yyi = np.rint((yy+dphi_y+self.nx/2)).astype(np.int) % self.nx
      xxi = np.rint((xx_+dphi_x+self.nx/2)).astype(np.int) % self.nx
      if self.think_positive:
        self.iss = clip(self.isrc[yyi,xxi],a_min=0,a_max=1e30)
      else:
        self.iss = self.isrc[yyi,xxi]


  def _load_src(self,stokes=(0,),think_positive=True):
    '''
    Load the source image from model (changes size and resolution to match the screen).

    :param stokes: (optional) tuple 
      Stokes parameters to consider. 
    :param think_positive: (optional) bool
        Should we enforce that the source image has no negative pixel values?
    '''
    M = self.model.shape[1]       # size of original image array
    N = self.nx                   # size of output image array

    if len(self.model.shape) > 2:
      self.isrc = np.empty((self.model.shape[-1],N,N))
    else:
      self.isrc = np.empty((1,N,N))
      self.model = np.reshape(self.model,(1,M,M))

    for s in stokes:
      f_img = RectBivariateSpline(self.model_dx/self.dx*(np.arange(M) - 0.5*(M-1)),\
                                 self.model_dx/self.dx*(np.arange(M) - 0.5*(M-1)),\
                                 self.model[s,:,:])

      xx_,yy_ = np.meshgrid((np.arange(N) - 0.5*(N-1)),\
                        (np.arange(N) - 0.5*(N-1)),indexing='xy')

      
      m  = f_img.ev(yy_.flatten(),xx_.flatten()).reshape((self.nx,self.nx))
      res  = m * (self.dx/self.model_dx)**2     # rescale for change in pixel size

      if s == 0 and think_positive:
        res[res < 0] = 0

      self.isrc[s,:,:]  = res 

    self.model = np.squeeze(self.model)
    self.isrc = np.squeeze(self.isrc)

  def saveSettings(self,filename):
    '''
    Save screen settings to a file.

    :param filename: string 
      settings filename
    '''
    f = open(filename,"w")
    f.write("wavelength \t {0}\n".format(self.wavelength)) 
    f.write("dpc \t {0}\n".format(self.dpc))
    f.write("rpc \t {0}\n".format(self.rpc))
    f.write("d \t {0}\n".format(self.d))
    f.write("m \t {0}\n".format(self.m))
    f.write("r0 \t {0}\n".format(self.r0)) 
    f.write("anisotropy \t {0}\n".format(self.anisotropy)) 
    f.write("pa \t {0}\n".format(self.pa))
    f.write("nphix \t {0}\n".format(self.nphi[0]))        # size of phase screen
    f.write("nphiy \t {0}\n".format(self.nphi[1]))        # size of phase screen
    f.write("screen_dx \t {0}\n".format(self.screen_dx))
    f.write("rf \t {0}\n".format(self.rf))
    f.write("ips \t {0}\n".format(self.ips))
    f.write("dx \t {0}\n".format(self.dx))
    f.write("nx \t {0}\n".format(self.nx))
    f.write("qmax \t {0}\n".format(self.r_inner))        # inner turbulence scale in r0 
    f.write("qmin \t {0}\n".format(self.r_outer))        # outer turbulence scale in r0 
    #f.write("qmax \t {0}\n".format(self.qmax))        # 1./inner turbulence scale in screen pixels
    #f.write("qmin \t {0}\n".format(self.qmin))        # 1./inner turbulence scale in screen pixels
    f.close()
