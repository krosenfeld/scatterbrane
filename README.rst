scatterbrane
============

A python module for adding realistic scattering to astronomical images.  This version
represents a very early release so please submit a pull request if you have trouble or
need help for your application.

For installation instructions and help getting started please see `the documentation <http://krosenfeld.github.io/scatterbrane/>`_.

Installation
------------

This project now uses ``pyproject.toml`` and `uv <https://docs.astral.sh/uv/>`_ for dependency management.

Install the package and its runtime dependencies with:

.. code-block:: bash

   uv sync

Install the release and development tooling with:

.. code-block:: bash

   uv sync --group dev

Release workflow
----------------

Version management is handled with ``bump-my-version``. To create a new tagged release:

.. code-block:: bash

   uv run bump-my-version bump patch

This updates the package version, creates a commit, and creates a ``vX.Y.Z`` git tag. Pushing that tag to GitHub triggers the publish workflow for PyPI.

Reference
---------

`Johnson, M. D., & Gwinn, C. R. 2015, ApJ, 805, 180  <http://adsabs.harvard.edu/abs/2015ApJ...805..180J>`_
