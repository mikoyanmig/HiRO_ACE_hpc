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
ds = xr.open_zarr(zarr_path)

u = ds["UGRD10m"]
v = ds["VGRD10m"]

# typical wind speed: speed first, then daily mean
speed = np.sqrt(u**2 + v**2).resample(time="1D").mean()
# alternative (speed of the average wind):
# speed = np.sqrt(u.resample(time="1D").mean()**2 + v.resample(time="1D").mean()**2)
n = speed.sizes["time"]

fig, ax = plt.subplots(figsize=(10, 4), dpi=100, subplot_kw={"projection": ccrs.PlateCarree()})
first = speed.isel(time=0)
mesh = ax.pcolormesh(first.lon, first.lat, first.values, cmap="viridis", vmin=0, vmax=15,
                     transform=ccrs.PlateCarree())
ax.coastlines()
fig.colorbar(mesh, ax=ax, label="m/s", shrink=0.7)

def update(i):
    frame = speed.isel(time=i)
    mesh.set_array(frame.values.ravel())
    ax.set_title(f"ACE2S 10 m wind speed {str(frame.time.values)[:10]}")
    return mesh,

anim = FuncAnimation(fig, update, frames=n, interval=150)
anim.save("windspeed_daily_2023.gif", writer="pillow", fps=8)
print("saved windspeed_daily_2023.gif")