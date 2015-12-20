ScatterBrane
============

A python package for simulating realistically scattered images. 

scatterbrane allows for anisotropic scattering kernels as well as time variability studies.  For a 
lightweight version try `SlimScat <https://github.com/krosenfeld/slimscat>`_.

The package is composed of two classes.  A :class:`Brane` is used to simulate the scattering while
a :class:`Target` can be used to calculate complex visibilities and closure phases.  We have also 
included some helpful functions in the ``utiilties`` module.

.. note::

    This is a pre-release of scatterbrane.  If you have a problem please submit a pull request on the git repository.

Installation
------------
Download the latest version from the `GitHub repository <https://github.com/krosenfeld/simscat>`_
, change to the main directory and run:

.. code-block:: bash

    python setup.py install

You can test your installation using the provided :doc:`examples <user/examples>`.

Quick Guide
-----------

.. toctree::
    :maxdepth: 3

    user/brane
    user/tracks
    user/utilities
    user/examples

References
----------
If you make use of this code, please cite
`Johnson, M. D., & Gwinn, C. R. 2015, ApJ, 805, 180  <http://adsabs.harvard.edu/abs/2015ApJ...805..180J>`_.

If you are interested in learning more about the physics
of scattering in this regime see these papers: 

- `The physics of Pulsar scintillation <http://adsabs.harvard.edu/abs/1992RSPTA.341..151N>`_
- `The shape of a scatter-broadened image. <http://adsabs.harvard.edu/abs/1989MNRAS.238..963N>`_
- `Radio propagation through the turbulent interstellar plasma <http://adsabs.harvard.edu/abs/1990ARA%26A..28..561R>`_

This documentation is styled after `dfm's projects <https://github.com/dfm>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

