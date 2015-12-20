__version__ = "0.0.1"

try:
    __SCATTERBRANE_SETUP__
except NameError:
    __SCATTERBRANE_SETUP__ = False

if not __SCATTERBRANE_SETUP__:
    __all__ = ["Brane"]

    import utilities
    from .brane import Brane
    from .tracks import Target
