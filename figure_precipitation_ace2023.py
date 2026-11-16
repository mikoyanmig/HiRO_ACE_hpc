import glob
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation

output_dir = "outputs/01_HIRO_ACE_2023"
zarr_path = sorted(glob.glob(f"{output_dir}/ace2s_2023jan_dec*/*.zarr"))[0]
print("reading", zarr_path, flush=True)
ds = xr.open_zarr(zarr_path)

# 6-hourly -> daily mean, convert kg/m2/s to mm/day
pr_daily = ds["PRATEsfc"].resample(time="1D").mean() * 86400
n = pr_daily.sizes["time"]
print("days:", n, flush=True)

fig, ax = plt.subplots(figsize=(10, 4), dpi=100, subplot_kw={"projection": ccrs.PlateCarree()})
first = pr_daily.isel(time=0)
mesh = ax.pcolormesh(first.lon, first.lat, first.values, cmap="turbo", vmin=0, vmax=30,
                     transform=ccrs.PlateCarree())
ax.coastlines()
fig.colorbar(mesh, ax=ax, label="mm/day", shrink=0.7)

def update(i):
    frame = pr_daily.isel(time=i)
    mesh.set_array(frame.values.ravel())
    ax.set_title(f"ACE2S precip {str(frame.time.values)[:10]}")
    return mesh,

anim = FuncAnimation(fig, update, frames=n, interval=150)
anim.save("precip_daily_2023.gif", writer="pillow", fps=8)
print("saved precip_daily_2023.gif")