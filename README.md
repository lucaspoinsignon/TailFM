```bash
# ~/.condarc
channels:
  - https://<nexus-host>/repository/<conda-proxy>/conda-forge
default_channels: []
channel_priority: strict

ls /domino/datasets/local/ /mnt 2>/dev/null; df -h | grep -v overlay

conda create -p /domino/datasets/local/Quail/envs/tails python=3.12 -y
conda activate /domino/datasets/local/Quail/envs/tails
python --version

pip install --index-url https://<nexus-host>/repository/<pypi-proxy>/simple \
    torch numpy scipy matplotlib pandas

```
