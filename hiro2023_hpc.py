import sys, os, glob, shutil, logging, time, datetime
import dacite, yaml, torch
import xarray as xr

torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)
t_start = time.perf_counter()
print("start:", datetime.datetime.now().isoformat(timespec="seconds"), flush=True)
print("GPU available:", torch.cuda.is_available(), flush=True)

output_dir = "outputs/01_HIRO_ACE_2023"
HIRO_CKPT  = "./hiro_ace/HiRO.ckpt"

START = sys.argv[1] if len(sys.argv) > 1 else "2023-01-02T00:00:00"
STOP  = sys.argv[2] if len(sys.argv) > 2 else "2023-01-03T18:00:00"
LAT0, LAT1, LON0, LON1 = 4.0, 21.0, 93.0, 109.0

ace_dir  = sorted(glob.glob(f"{output_dir}/ace2s_2023jan_dec*"))[0]
ace_zarr = sorted(glob.glob(f"{ace_dir}/*.zarr"))[0]
ace_ds   = xr.open_zarr(ace_zarr)
t_first  = str(ace_ds.time.values[0])[:19].replace(" ", "T")
t_last   = str(ace_ds.time.values[-1])[:19].replace(" ", "T")
print("ACE2S output :", ace_zarr)
print("ACE2S times  :", t_first, "to", t_last)
print("variables    :", list(ace_ds.data_vars))
if not (t_first <= START and STOP <= t_last):
    print(f"WARNING: {START} to {STOP} is not inside the ACE2S output; HiRO will probably fail", flush=True)

tag = f"{START[:10]}_{STOP[:10]}"
hiro_dir = f"{output_dir}/hiro_{tag}"
shutil.rmtree(hiro_dir, ignore_errors=True)

hiro_yaml = f"""
experiment_dir: {hiro_dir}
patch:
  divide_generation: true
  composite_prediction: true
  coarse_horizontal_overlap: 1
model:
  checkpoint_path: {HIRO_CKPT}
{rename_block}data:
  coarse:
  - data_path: {ace_dir}
    engine: zarr
    file_pattern: {os.path.basename(ace_zarr)}
  batch_size: 4
  num_data_workers: 0
  strict_ensemble: false
outputs:
  - name: 3km
    save_vars: [PRATEsfc]
    n_ens: 1
    max_samples_per_gpu: 4
    time_range:
      start_time: '{START}'
      stop_time: '{STOP}'
    lat_extent:
      start: {LAT0}
      stop: {LAT1}
    lon_extent:
      start: {LON0}
      stop: {LON1}
logging:
  log_to_screen: true
  log_to_wandb: false
  log_to_file: true
"""

from fme.downscaling.inference.inference import (
    InferenceConfig as HiroInferenceConfig,
    prepare_directory,
)

def run_hiro(yaml_text: str) -> None:
    config = yaml.safe_load(yaml_text)
    generation_config = dacite.from_dict(HiroInferenceConfig, config,
                                         config=dacite.Config(strict=True))
    prepare_directory(generation_config.experiment_dir, config)
    generation_config.configure_logging(log_filename="out.log")
    logging.info("Starting downscaling generation...")
    generation_config.build().run_all()

t_run = time.perf_counter()
run_hiro(hiro_yaml)
t_end = time.perf_counter()

msg = f"{START} -> {STOP} | setup {t_run - t_start:.0f} s | run {t_end - t_run:.0f} s | total {(t_end - t_start)/60:.1f} min"
print(msg, flush=True)
with open(f"{output_dir}/hiro_timing.txt", "a") as f:
    f.write(msg + "\n")