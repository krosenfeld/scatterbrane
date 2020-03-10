import os
from setuptools import setup

if __name__ == "__main__":
    import scatterbrane
    setup(
        name="scatterbrane",
        version = "0.1.0",
        packages = ["scatterbrane"],
        author = "Katherine Rosenfeld",
        author_email = "krosenf@gmail.com",
        description = ("A python module to simulate the effect of anisotropic scattering "
                    "on astrophysical images."),
        license = "MIT",
        keywords = "scattering astronomy EHT"
    )
