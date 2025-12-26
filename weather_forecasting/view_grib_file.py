import xarray as xr
import pandas as pd

# open datasets with cfgrib engine
ds_ecmwf = xr.open_dataset("test.grib", engine="cfgrib", filter_by_keys={"centre": "ecmf"})
ds_ncep  = xr.open_dataset("test.grib", engine="cfgrib", filter_by_keys={"centre": "kwbc"})

# 'tp' is total precipitation, shape (time, step, lat, lon)
tp_ecmwf = ds_ecmwf["tp"]
tp_ncep  = ds_ncep["tp"]

print(tp_ecmwf.shape)  # (31, 3, 361, 720)

# pick one time and one forecast step
time_idx = 3    # first day
step_idx = 2   # first forecast step (0 hours)

# select the slice
tp_ecmwf_slice = tp_ecmwf.isel(time=time_idx, step=step_idx)
tp_ncep_slice  = tp_ncep.isel(time=time_idx, step=step_idx)

# Convert to pandas DataFrame for inspection
df_ecmwf = tp_ecmwf_slice.to_dataframe().reset_index()
df_ncep  = tp_ncep_slice.to_dataframe().reset_index()

print(df_ecmwf.head(100))
print(df_ncep.head(100))
