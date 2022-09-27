> Distribution Statement A. Approved for public release. Distribution unlimited.
> 
> Author:
> Naval Research Laboratory, Marine Meteorology Division
> 
> This program is free software:
> you can redistribute it and/or modify it under the terms
> of the NRLMMD License included with this program.
> If you did not receive the license, see
> https://github.com/U-S-NRL-Marine-Meteorology-Division/
> for more information.
> 
> This program is distributed WITHOUT ANY WARRANTY;
> without even the implied warranty of MERCHANTABILITY
> or FITNESS FOR A PARTICULAR PURPOSE.
> See the included license for more details.


<div
  align="center"
>

# <p style="text-align:center"><img src="https://github.com/U-S-NRL-Marine-Meteorology-Division/xnrl/blob/main/images/logo.png?raw=true" width=250>

<!-- Badges -->

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- (Badges) -->
</div>

**xNRL** helps you read NRL NWP output into xarray Datasets nested within Pandas DataFrames.

# Install
xNRL requires numpy, pandas, and xarray.

## Clone and install with pip
```bash
git clone https://github.com/U-S-NRL-Marine-Meteorology-Division/xnrl.git
cd xnrl
pip install -e .
```
> The `-e` makes the code editable. 
> 
> You can update the code with 
> ```bash
> cd xnrl
> git pull origin main
> ```

## Install with Conda

Copy the [`environment_xnrl.yml`](./environment_xnrl.yml) file and create the Conda environment. 

```bash
conda env create -f environment_xnrl.yml
```
> Note: This only installs the current main branch and does not let you edit the code. You can update xnrl (and all packages in the environment) with 
> ```
> conda env update -f environment_xnrl.yml
>```

# Examples
```python
import xnrl

# Load COAMPS flatfiles into an xarray Dataset
fp = '<path>/*pre*00120*'
ds = xnrl.open_dataset(fp, model='COAMPS')
```
