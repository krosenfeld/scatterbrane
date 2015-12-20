"""
.. module:: tracks
    :platform: Unix
    :synopsis: Lightweight module for generating visibilities and closure phases

.. moduleauthor:: Katheirne Rosenfeld <krosenf@gmail.com>
.. moduleauthor:: Michael Johnson

"""

from __future__ import print_function

import itertools
from numpy import (pi,array,cos,sin,cross,dot,asarray,linspace,sum,transpose,dot,exp,angle,
                    newaxis,prod,flipud,arange,zeros,complex64,abs,radians)
from numpy.fft import fftshift,fftfreq
from numpy.linalg import norm
import logging

__all__ = ["Target"]

class Target(object):
  '''
  Class for constructing visibility samples.

  :param wavelength: (optional) 
    Scalar wavelength in meters.
  :param position: (optional) ``(2, )`` 
    A tuple with right ascension and declination of the source in (hours, degrees) or the preset string "sgra".
  :param obspos: (optional) 
    A dictionary of observatories and their geocentric positions.
  :param elevation_cuts: (optional) ``(2, )`` 
    The upper and lower bound on the observing zenith angle. 

  .. note:: The zenith angle cut corresponds to whether the site can observe the source.  As a result, for functions such as :func:`generateTracks`, the number of uv samples returned will be less than or equal to the number of samples set by the ``times`` argument.  To disable the zenith angle cut you can initialize :class:`Target` with ``zenith_cuts`` = ``[180.,0.]``.
  '''

  def __init__(self,wavelength=1.3e-3,position='sgra',obspos='eht',zenith_cuts=[75.,0.]):

    if position=='sgra':
      ra,dec = (17+45./60+40.0409/3600,-1*(29+28.118/3600))
    else:
      ra,dec = position

    # observing settings
    self.dec    = dec
    self.ra     = ra 
    self.wavelength = wavelength
    self.cos_zenith_cuts = cos(radians(zenith_cuts))

    # derived quantities
    self.target_vec = array([cos(self.dec * pi / 180), 0., sin(self.dec * pi / 180)])
    self.projU = cross([0.,0.,1.],self.target_vec)
    self.projU /= norm(self.projU)
    self.projV = -1.*cross(self.projU,self.target_vec)
 
    # dictionary of observatories and their positions
    l = self.wavelength # in meters
    if obspos == 'eht':
        self.obs = dict([("SMA",array([-5464523.400, -2493147.080, 2150611.750])/l),\
                     ("SMT",array([-1828796.200, -5054406.800, 3427865.200])/l),\
                     ("CARMA",array([-2397431.300, -4482018.900, 3843524.500])/l),\
                     ("LMT",array([-768713.9637, -5988541.7982, 2063275.9472])/l),\
                     ("ALMA",array([2225037.1851, -5441199.1620, -2479303.4629])/l),\
                     ("PV",array([5088967.9000, -301681.6000, 3825015.8000])/l),\
                     ("PDBI",array([4523998.40, 468045.240, 4460309.760])/l),\
                     ("SPT",array([0.0, 0.0, -6359587.3])/l),\
                     ("HAYSTACK",array([1492460, -4457299, 4296835])/l),\
              ]) # in wavelengths
    else:
        self.obs = obspos
        for k in self.obs.iterkeys():
            self.obs[k] /= l

  def sites(self,s):
    '''
    Look up geocentric positions of sites by their string keyword.

    :param s: List of site keys.
    
    :returns: Array of site geocentric positions.
    '''
    return array([self.obs[s_] for s_ in s])

  def _RR(self,v,theta):
    '''
    Rotates the vector v by angle theta.

    :param v: A vector (x,y,z).
    :param theta: Angle for CCW rotation in radians.

    :returns: Rotated vector (x',y',z')
    '''
    return array([cos(theta)*v[0] - sin(theta)*v[1],\
                  sin(theta)*v[0] + cos(theta)*v[1],\
                  v[2]])

  def _cosZenith(self,v,theta):
    ''' 
    :param v: ``(3, )`` vector.
    :param theta: Scalar angle in radians.

    :returns: cos(zenith angle) = sin(elevation)
    '''
    return dot(self._RR(v,theta),self.target_vec) / (norm(v) * norm(self.target_vec))


  def _RRelevcut(self,v,theta):
    '''
    Does vector v pass the zenith angle cut?

    :param v: ``(3, )`` vector
    :param theta: Scalar angle in radians.
    '''
    if abs(self._cosZenith(v,theta)-0.5*sum(self.cos_zenith_cuts)) < 0.5*sum([-1,1]*self.cos_zenith_cuts):
        return True
    else:
        return False
    
  def generateUV(self,site_pair,theta):
    '''
    Generate uv baseline.

    :param site_pair: ``(2, 3)`` 
      Geocentric (x,y,z) positions of two sites.
    :param theta: scalar 
      Angle in radians.

    :returns: ``(2, )`` 
      A uv point. 
    '''
    return array([dot(self._RR(site_pair[0],theta)-self._RR(site_pair[1],theta),self.projU),\
                  dot(self._RR(site_pair[0],theta)-self._RR(site_pair[1],theta),self.projV)])

  def hour(self,theta):
    ''' 
    Convert to UT time.

    :param theta: ``(n, )`` vector of angles in radians.

    :returns: ``(n, )`` vector of times in hours. 
    '''
    return (theta / (2. * pi) * 24. + self.ra) % 24

  def theta(self,hour):
    ''' 
    Convert from UT to angle.

    :param hour: ``(n, )`` 
      Vector of times in hours.

    :returns: ``(n, )`` 
      Vector of angles in radians. 
    '''
    return ((hour - self.ra) / 24. * 2. * pi) % (2 * pi)

  def _gst(self,hour):
    return ((hour-12.) % 24.) - 12.

  def generateTracks(self,sites,times=(0.,24.,int((24.-0.)*60./10.)),return_hour=False):
    '''
    Generates uv-samples where the sample rate is to be equally spaced in time. 
    For instance, if you want samples spaced by 5 minutes ranging from 1 UT to 4 UT 
    (non-inclusive) you would set ``times`` = ``(1, 4, int((4.-1.)*60./5.))``.  As 
    another example, you would generate 128 samples separated by 10 seconds if 
    ``times`` = ``(0., 128*10./3600, 128)``.

    :param sites: ``(num_sites, 3)`` 
      Numpy array of sites' geocentric positions.
    :param times: (optional) ``(start, stop, num_samples)`` 
      Tuple setting the spacing of the uv-samples in time.  The ``start`` and ``stop`` times should be in hours and in the range [0,24].  ``stop`` is non-inclusive.
    :param return_hour: (optional) bool
      Return hour of uv samples as second item to unpack.

    :returns: ``(num_measurements, 2)`` 
      Numpy array with uv samples where ``num_measurements`` :math:`\\leq` ``num_samples``. 

    '''

    times = linspace(times[0],times[1],num=times[2],endpoint=False)
    uv = []
    hr = []
    for h in times:
        theta = self.theta(h)
        for baseline in itertools.combinations(sites,2):
            if self._RRelevcut(baseline[0],theta) and self._RRelevcut(baseline[1],theta):
                if return_hour:
                    hr.append(h)
                    hr.append(h)
                uv.append(self.generateUV((baseline[0],baseline[1]),theta))
                uv.append(self.generateUV((baseline[1],baseline[0]),theta))

    if return_hour:
        return (array(uv),array(hr))
    else:
        return asarray(uv)

  def calculateCA(self,img,dx,uvQuadrangle):
    ''' 
    Calculate closure amplitudes for numpy arrays of 4 baselines: 

    .. math::

        A_{abcd} = \\frac{|V_{ab}||V_{cd}|}{|V_{ac}||V_{bd}|}

    :param img: ``(n, n)`` 
      Numpy image array. 
    :param dx: scalar 
      Pixel size of image in microarcseconds. 
    :param uvQuadrangle: ``(num_measurements, 4, 2)`` 
      The uv samples for which to calculate the closure amplitude. The baseline order along axis=1 should be (ab,cd,ac,bd).  

    :returns: ``(num_measurements, )`` 
      Closure amplitudes.
    '''

    # case we are only given one time sample
    if len(uvQuadrangle.shape) == 2: uvQuadrangle = uvQuadrangle[newxais,:]

    ca = []
    for quad in uvQuadrangle:
        v = array([ abs(self.FTElementFast(img,dx,t)) for t in quad ])
        ca.append( v[0]*v[1]/(v[2]*v[3]) )
    return asarray(ca)

  def calculateCP(self,img,dx,uvTriangle):
    ''' 
    Calculate closure phases for numpy arrays of 3 baselines that should form a closed triangle:

    .. math::

        \\phi_{abc} = \\mathrm{arg}(V_{ab}V_{bc}V_{ca})

    :param img: ``(n, n)`` 
      Numpy image array. 
    :param dx: scalar 
      Pixel size of image in microarcseconds. 
    :param uvTriangle: ``(num_measurements, 3, 2)`` 
      The uv samples for which to calculate the closure phase.  The baseline order along axis=1 should be (ab,bc,ca).

    :returns: ``(num_measurements, )`` 
      Closure phases in radians.
    '''

    # case where we are only given one time sample
    if len(uvTriangle.shape) == 2: uvTriangle = uvTriangle[newaxis,:]
    
    cp = []
    for triang in uvTriangle:
        v = [self.FTElementFast(img,dx,t) for t in triang]
        cp.append(angle(prod(v)))

    return asarray(cp)

  def calculateBS(self,img,dx,uvTriangle):
    ''' 
    Calculate bispectrum for numpy arrays of 3 baselines that should form a closed triangle: 

    .. math::

        B_{abc} = V_{ab}V_{bc}V_{ca}

    :param img: ``(n, n)`` 
      Image array. 
    :param dx: scalar 
      Pixel size of image in microarcseconds. 
    :param uvTriangle: ``(num_measurements, 3, 2)`` 
      uv samples for which to calculate the bispectra.  The baseline order along axis=1 should be (ab,bc,ca).

    :returns: ``(num_measurements, )`` 
      Complex bispectra.
    '''

    # case where we are only given one time sample
    if len(uvTriangle.shape) == 2: uvTriangle = uvTriangle[newaxis,:]
    
    bs = []
    for triang in uvTriangle:
        v = [self.FTElementFast(img,dx,t) for t in triang]
        bs.append(prod(v))

    return asarray(bs)

  def generateQuadrilateralTracks(self,sites,times=(0.,24.,int((24.-0.)*60./10.)),return_hour=False):
    '''
    Generates a quadruplet of uv points for a quadrilateral (a,b,c,d) ensuring that all sites can see the source.  See the docstring for :func:`generateTracks` for notes about the sampling rate.

    **Suggested usage:** Feed the generated `uv` samples to :func:`calculateCA` which will return the closure amplitude quantity.

    :param sites: ``(4, 3)`` 
      Array of geocentric positions for 4 sites (a,b,c,d).
    :param times: (optional) ``(start, stop, num_samples)`` 
      Tuple setting the spacing of the uv-samples in time.  The ``start`` and ``stop`` times should be in hours and in the range [0,24].  ``stop`` is non-inclusive.
    :param return_hour: (optional) 
      Return hour of uv samples as second item to unpack.

    :returns: ``(num_measurements, 4, 2)`` 
      Array where the order of the baselines along the first dimension (axis=1) is (ab, cd, ac, bd).
    '''

    assert len(sites) == 4, "sites should form a quadrilateral"
    times = linspace(times[0],times[1],num=times[2],endpoint=False)
    uv = []
    hr = []
    for h in times:
        theta = self.theta(h)
        # zenith cut
        if self._RRelevcut(sites[0],theta) and \
           self._RRelevcut(sites[1],theta) and \
           self._RRelevcut(sites[2],theta) and \
           self._RRelevcut(sites[3],theta):
            quad = (self.generateUV((sites[0],sites[1]),theta),self.generateUV((sites[2],sites[3]),theta),\
                    self.generateUV((sites[0],sites[2]),theta),self.generateUV((sites[1],sites[3]),theta))
            uv.append(quad)
            if return_hour: hr.append(h)

    if return_hour:
        return (array(uv),array(hr))
    else:
        return asarray(uv)

  def generateTriangleTracks(self,sites,times=(0.,24.,int((24.-0.)*60./10.)),return_hour=False):
    '''
    Generates a triplet of uv points for a closed triangle of sites (a,b,c) ensuring that all sites can see the source.  
    See the docstring for :func:`generateTracks` for notes about the sampling rate.

    **Suggested usage:** Feed the generated UV samples to :func:`calculateBS` or :func:`calculateCP` to calculate closure phase or bispectrum quanities.

    :param sites: ``(3, 3)`` 
      Array of geocentric positions for 3 sites (a,b,c).
    :param times: (optional) ``(start, stop, num_samples)`` 
      Tuple setting the spacing of the uv-samples in time.  The ``start`` and ``stop`` times should be in hours and in the range [0,24].  ``stop`` is non-inclusive.
    :param return_hour: (optional) 
      Return hour of uv samples as second item to unpack.

    :returns: ``(num_measurements, 3, 2)`` 
      Array of uv points where the order of the baselines along the first dimension (axis=1) is (ab,bc,ca).
    '''
    assert len(sites) == 3, "sites should form a triangle"

    times = linspace(times[0],times[1],num=times[2],endpoint=False)
    uv = []
    hr = []
    for h in times:
        theta = self.theta(h)
        # zenith cut
        if self._RRelevcut(sites[0],theta) and \
           self._RRelevcut(sites[1],theta) and \
           self._RRelevcut(sites[2],theta):
            triang = array([ self.generateUV((sites[0],sites[1]),theta),\
                            self.generateUV((sites[1],sites[2]),theta),\
                            self.generateUV((sites[2],sites[0]),theta) ])
            uv.append(triang)
            if return_hour: hr.append(h)

    if return_hour:
        return (array(uv),array(hr))
    else:
        return asarray(uv)

  def calculateVisibilities(self,img,dx,uv):
    '''
    Calculate complex visibilities.

    :param img: Image array.
    :param dx: Pixel size in microarcseconds.
    :param uv: ``(num_measurements, 2)`` An array of `uv` samples with dimension.

    :returns: ``(num_measurements, )`` Scalar complex visibilities.
    '''

    # case of only 1 uv point
    if len(uv.shape) == 1: uv = uv[newaxis,:] 

    num_samples = uv.shape[0]
    V = zeros(num_samples,dtype=complex64)
    for i in range(num_samples):
        V[i] = self.FTElementFast(img,dx,uv[i,:])
    return V

  def FTElementFast(self,img,dx,baseline):
    '''
    Return complex visibility.

    :param img: Image array.
    :param dx: pixel size in microarcseconds.
    :param baseline: (u,v) point in wavelengths.

    :returns: complex visibility

    '''
    nx = img.shape[-1]
    du = 3600.*1e6/(nx*dx*radians(1.))
    ind = arange(nx)
    return sum(img * dot(\
             transpose([exp(-2*pi*1j/du/nx*baseline[1]*flipud(ind))]),\
             [exp(-2*pi*1j/du/nx*baseline[0]*ind)]))
