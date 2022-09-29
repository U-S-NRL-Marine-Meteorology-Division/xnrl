# # Distribution Statement A. Approved for public release. Distribution unlimited.
# # 
# # Author:
# # Naval Research Laboratory, Marine Meteorology Division
# # 
# # This program is free software:
# # you can redistribute it and/or modify it under the terms
# # of the NRLMMD License included with this program.
# # If you did not receive the license, see
# # https://github.com/U-S-NRL-Marine-Meteorology-Division/
# # for more information.
# # 
# # This program is distributed WITHOUT ANY WARRANTY;
# # without even the implied warranty of MERCHANTABILITY
# # or FITNESS FOR A PARTICULAR PURPOSE.
# # See the included license for more details.

'''Python setup script for xnrl package'''

from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
version = (this_directory / "VERSION").read_text()

setup(
    name="xnrl",
    version=version,
    description="xNRL - Read NRL files into xarray Datasets nested within pandas DataFrames",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=["xnrl"],
    include_package_data=True,
    install_requires=[ "xarray", "numpy", "dask[delayed]", "metpy"],    
    keywords=["flatfile", "nrl", "xarray", "dataset", "binary", "hdf5", "grib"],
    entry_points={
        "console_scripts": [
            "xdump = xnrl.cli:xdump",
            "xoutf = xnrl.cli:xoutf",
            "xplot = xnrl.cli:xplot",
        ]
    },
    zip_safe=False,
)
