.. module:: brane

.. _brane:

The Brane
=========

Important Notes
---------------

This section highlights some important aspects of the module.

The first thing to do is load the :class:`Brane` class and logging module

.. code-block:: python

    from scatterbrane import Brane
    import logging
    logging.basicConfig(level=logging.INFO) # for example

The :class:`Brane` object holds the parameters for the scattering screen and functions 
for generating images.  To initialize a simulation you contruct an instance of the screen
and set parameters

.. code-block:: python

    import numpy as np
    b = Brane(np.zeros((64,64)),1,nphi=4096,screen_res=5,wavelength=3e-6,r_inner=100)

In this example I have set the source to be empty
with a pixel size of 1 microarcsecond.  Upon initialization, :class:`Brane` creates a version of the 
source image whose pixel size is chosen to match the screen.  


.. warning::

    The resolution of the scattered (`iss`) and source images (`isrc`) may not match the original
    resolution element you provided Brane.  You should access the property through `dx`.

Construction of a :class:`Brane` instance will check some properties:

- Is the image resolution approaching the refractive scale? (warning only)
- Screen covers the source image.
- Source image is square.
- Inner turbulence scale is larger than the phase coherence length.
- Resolution element of the screen is larger than the inner turbulence scale.
- A screen pixel is smaller than inner turbulence scale.
- Image is smooth enough to satisfy the approximation for equation 12 in Johnson & Gwinn (2015): 

.. math::

    V_{\mathrm{src}}\left(\frac{(1+M)r_{\mathrm{inner}}}{\lambda}\right) \ll V_{\mathrm{src}}(0) 


We can now proceed to generate the phase screen and scatter the source image:

.. code-block:: python
   
    b.generateScreen()
    b.scatter()

By default, for anisotropic scattering with some position angle, :func:`scatter` rotates the source 
structure.  This may introduce artifacts and reduce performance.  See `the examples <examples>`_ for 
some ways around this. 

Class Methods
-------------

.. autoclass:: scatterbrane.Brane
    :members:
