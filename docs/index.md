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
