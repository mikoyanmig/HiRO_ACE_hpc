from huggingface_hub import hf_hub_download

repo_id = "allenai/HiRO-ACE"
local_dir = "hiro_ace"

for filename in [
    "ACE2S.ckpt",
    "initial_conditions/ic_2023.nc",
    "forcing_data/forcing_2023.nc",
]:
    hf_hub_download(repo_id, filename, local_dir=local_dir)
print("download done")