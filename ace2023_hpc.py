import time, datetime, os, logging
import dacite, yaml, torch

t_start = time.perf_counter()
print("start:", datetime.datetime.now().isoformat(timespec="seconds"), flush=True)

torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)
print("GPU available:", torch.cuda.is_available(), flush=True)

from fme.ace.inference.inference import InferenceConfig, run_inference_from_config

HIRO_PATH         = "./hiro_ace"
ACE2S_CKPT        = f"{HIRO_PATH}/ACE2S.ckpt"
ACE2S_IC          = f"{HIRO_PATH}/initial_conditions/ic_2023.nc"
ACE2S_IC_IDX      = 0          # 0 = 1 Jan 2023
ACE2S_FORCING_DIR = f"{HIRO_PATH}/forcing_data"
N_STEPS           = 1460       # 365 days x 4; try 1459 if forcing runs out

output_dir = "outputs/01_HIRO_ACE_2023"
os.makedirs(output_dir, exist_ok=True)

ace2s_yaml = f"""
experiment_dir: {output_dir}/ace2s_2023jan_dec
n_forward_steps: {N_STEPS}
forward_steps_in_memory: 40
checkpoint_path: {ACE2S_CKPT}
logging:
  log_to_screen: true
  log_to_wandb: false
initial_condition:
  path: {ACE2S_IC}
  engine: netcdf4
  start_indices:
    first: {ACE2S_IC_IDX}
    n_initial_conditions: 1
forcing_loader:
  dataset:
    data_path: {ACE2S_FORCING_DIR}
    file_pattern: forcing_2023.nc
    engine: netcdf4
  num_data_workers: 0
data_writer:
  save_prediction_files: false
  save_monthly_files: false
  files:
    - label: ace2s
      format:
        name: zarr
      separate_ensemble_members: true
"""

ace2s_cfg = dacite.from_dict(InferenceConfig, yaml.safe_load(ace2s_yaml),
                             config=dacite.Config(strict=True))

t_run = time.perf_counter()
_ = run_inference_from_config(ace2s_cfg)
t_end = time.perf_counter()

msg = (f"setup {t_run - t_start:.1f} s | run {t_end - t_run:.1f} s | "
       f"total {(t_end - t_start)/60:.1f} min | {(t_end - t_run)/N_STEPS:.3f} s/step")
print(msg, flush=True)
with open(f"{output_dir}/timing.txt", "a") as f:
    f.write(f"steps={N_STEPS}: {msg}\n")