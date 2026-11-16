from huggingface_hub import hf_hub_download

repo_id = "allenai/HiRO-ACE"
local_dir = "hiro_ace"

for filename in [
    "HiRO.ckpt"
]:
    hf_hub_download(repo_id, filename, local_dir=local_dir)
print("download done")