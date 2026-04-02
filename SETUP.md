# Cluster Experiment Runner — Setup Guide

This guide walks you through setting up CER on your local machine and the remote SLURM cluster.

## Prerequisites

- Python 3.11+ on your local machine
- SSH key-based access to the cluster (no password prompts)
- Git + GitHub repo with your experiment code
- SLURM cluster with Singularity installed
- A Weights & Biases account (for experiment tracking)

---

## 1. Local Installation

Clone the repo and install in editable mode:

```bash
cd ~/Programming/cluster_experiment_runner
pip install -e ".[dev]"
```

Verify it works:

```bash
cer --help
```

---

## 2. SSH Setup

CER uses your system `ssh` binary directly, so it respects your `~/.ssh/config`.

### 2a. Your SSH config

You should already have this in `~/.ssh/config`:

```
Host lumi.csc.fi
    HostName lumi.csc.fi
    User mrogalaj
    ForwardAgent yes
    IdentityFile ~/.ssh/id_lumi
```

Verify the connection works:

```bash
ssh lumi.csc.fi echo "Connection OK"
```

> **Note:** `ForwardAgent yes` is already set, which means your GitHub SSH key is forwarded to LUMI — this is needed for cloning repos on the cluster.

### 2b. (Recommended) Enable SSH multiplexing

CER makes multiple SSH calls (submit, status checks, etc.). Multiplexing keeps a persistent connection open so these are fast. Add to your existing config:

```
Host lumi.csc.fi
    HostName lumi.csc.fi
    User mrogalaj
    ForwardAgent yes
    IdentityFile ~/.ssh/id_lumi
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
```

Create the socket directory:

```bash
mkdir -p ~/.ssh/sockets
```

---

## 3. Cluster Setup

SSH into LUMI and set up the base directory structure:

```bash
ssh lumi.csc.fi
```

### 3a. Create the experiments directory

On LUMI, project files go in `/scratch`. Replace `project_xxx` with your actual LUMI project ID:

```bash
mkdir -p /scratch/project_xxx/mrogalaj/experiments/logs
mkdir -p /scratch/project_xxx/mrogalaj/containers
```

This is where CER will store the bare repo, worktrees, and SLURM logs.

### 3b. Build the Singularity container

Copy the definition file to LUMI:

```bash
# From your local machine
scp cluster/singularity/experiment.def lumi.csc.fi:/scratch/project_xxx/mrogalaj/containers/
```

On LUMI, build the container:

```bash
ssh lumi.csc.fi
cd /scratch/project_xxx/mrogalaj/containers
singularity build --fakeroot experiment.sif experiment.def
```

This uses `nvcr.io/nvidia/pytorch:24.07-py3` as the base image, which includes CUDA, cuDNN, and PyTorch. It also installs `wandb`, `pyyaml`, `scipy`, `matplotlib`, and `tqdm`.

> **Note on LUMI:** LUMI uses Singularity/Apptainer natively. If `--fakeroot` isn't available, try `singularity build experiment.sif experiment.def` or use `cotainr` (LUMI's recommended container tool). Check [LUMI container docs](https://docs.lumi-supercomputer.eu/software/containers/) for specifics.

If you need additional base packages, edit `cluster/singularity/experiment.def` and rebuild. Project-specific pip dependencies from `requirements.txt` are installed at runtime automatically.

### 3c. Set up W&B on the cluster

You need your W&B API key accessible inside containers on LUMI. The simplest way: get your key from https://wandb.ai/authorize and add it to your shell config on LUMI:

```bash
ssh lumi.csc.fi
echo 'export WANDB_API_KEY="paste-your-key-here"' >> ~/.bashrc
```

Singularity containers on LUMI mount your home directory (`/users/mrogalaj/`) automatically, so the container will pick up this env var.

Alternatively, install wandb on the login node and use `wandb login` (which saves the key to `~/.netrc`):

```bash
ssh lumi.csc.fi
pip install --user wandb
wandb login
```

### 3d. Verify GitHub SSH access from the cluster

Since your SSH config has `ForwardAgent yes`, your local GitHub SSH key is forwarded to LUMI. Verify it works:

```bash
ssh lumi.csc.fi "ssh -T git@github.com"
```

You should see "Hi <username>! You've successfully authenticated...". If not, check that your local SSH agent has the GitHub key loaded:

```bash
ssh-add -l   # should show your GitHub key
```

---

## 4. Configure CER

Back on your local machine, create your config file:

```bash
cp cer.yaml.example cer.yaml
```

Edit `cer.yaml` with your actual values:

```yaml
cluster:
  host: "lumi.csc.fi"                                   # SSH host from step 2
  base_dir: "/scratch/project_xxx/mrogalaj/experiments"  # Directory from step 3a
  repo_url: "git@github.com:<you>/<repo>.git"            # Your experiment repo
  repo_branch: "main"

container:
  image: "/scratch/project_xxx/mrogalaj/containers/experiment.sif"  # Built in step 3b
  bind_mounts:
    - "/scratch/project_xxx/mrogalaj/data:/data"         # Mount your datasets

slurm:
  partition: "gpu"           # Your cluster's GPU partition name
  gres: "gpu:1"             # GPU request (gpu:2 for multi-GPU)
  cpus_per_task: 8
  mem: "32G"
  time: "24:00:00"           # Max wall time
  extra_flags: []            # e.g. ["--exclusive", "--nodelist=node01"]

experiment:
  entrypoint: "python train.py"    # Command run inside the container
  wandb_project: "my-project"      # Your W&B project name
  wandb_entity: ""                 # W&B team name (leave empty for personal)

local:
  db_path: "~/.local/share/cer/experiments.db"   # Local tracking database
```

### Config search order

1. `./cer.yaml` (project directory)
2. `~/.config/cer/cer.yaml` (user global)
3. Environment variables override any file value: `CER_CLUSTER_HOST`, `CER_SLURM_PARTITION`, etc.

### Required fields

These must be set — CER will error if they're missing:

| Field | Example |
|-------|---------|
| `cluster.host` | `"lumi.csc.fi"` |
| `cluster.base_dir` | `"/scratch/project_xxx/mrogalaj/experiments"` |
| `cluster.repo_url` | `"git@github.com:user/ml-experiments.git"` |
| `container.image` | `"/scratch/project_xxx/mrogalaj/containers/experiment.sif"` |

Everything else has sensible defaults.

---

## 5. Prepare Your Experiment Code

Your training script should:

1. **Accept CLI arguments** for hyperparameters (CER passes `--config key=value` as `--key value` to the entrypoint):

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()
```

2. **Use W&B for logging** (CER sets `WANDB_PROJECT`, `WANDB_RUN_NAME`, and `WANDB_TAGS` automatically):

```python
import wandb
import os

wandb.init(
    config=vars(args),
    tags=[os.environ.get("CER_COMMIT", "unknown")],
)

# Your training loop
for epoch in range(100):
    loss = train_one_epoch(args)
    wandb.log({"loss": loss, "epoch": epoch})
```

3. **Include a `requirements.txt`** in the repo root for any project-specific dependencies (installed at runtime inside the container).

---

## 6. Usage

### Submit an experiment

```bash
# 1. Make changes to your code
# 2. Commit and push
git add -A && git commit -m "try higher learning rate"
git push

# 3. Submit the experiment
cer submit $(git rev-parse HEAD) --config lr=0.01 --config batch_size=64
```

Output:
```
Submitting experiment for commit a1b2c3d4...
Submitted! Job ID: 12345
```

### Check status

```bash
cer status 12345
```

Output:
```
Job ID:    12345
Commit:    a1b2c3d4 (a1b2c3d4e5f6...)
Status:    RUNNING
Submitted: 2026-04-02T14:30:00
Config:    {"lr": "0.01", "batch_size": "64"}
W&B:       https://wandb.ai/team/project/runs/abc123
```

### List all experiments

```bash
cer list
```

### Cancel an experiment

```bash
cer cancel 12345
```

---

## 7. Troubleshooting

### "No cer.yaml found"

Create a config file — see step 4.

### "SSH command timed out"

- Check your SSH connection: `ssh lumi.csc.fi echo ok`
- Check SSH multiplexing is enabled (step 2c)
- Increase timeout if your cluster is slow

### "Commit not found after 30 retries"

CER polls for 5 minutes waiting for the commit to appear on GitHub. Make sure you pushed:

```bash
git push origin main
```

### sbatch fails with "Invalid partition"

Check your cluster's available partitions:

```bash
ssh lumi.csc.fi sinfo -s
```

Update `slurm.partition` in `cer.yaml` to match.

### Singularity "image not found"

Verify the `.sif` file exists at the path in your config:

```bash
ssh lumi.csc.fi ls -la /scratch/project_xxx/mrogalaj/containers/experiment.sif
```

### W&B run not showing up

- Check W&B is logged in on the cluster: `ssh lumi.csc.fi wandb status`
- Check the SLURM error log: `ssh lumi.csc.fi cat /scratch/project_xxx/mrogalaj/experiments/logs/<job_id>.err`

---

## 8. Directory Layout on the Cluster

After running experiments, the cluster will look like:

```
/scratch/project_xxx/mrogalaj/experiments/
├── repo/                          # Bare git clone (shared, created once)
├── worktrees/
│   ├── a1b2c3d4e5f6.../          # Checked-out code for each experiment
│   └── f7e8d9c0b1a2.../
└── logs/
    ├── 12345.out                  # SLURM stdout
    ├── 12345.err                  # SLURM stderr
    ├── 12346.out
    └── 12346.err
```

To clean up old worktrees manually:

```bash
ssh lumi.csc.fi "cd /scratch/project_xxx/mrogalaj/experiments/repo && git worktree list"
ssh lumi.csc.fi "cd /scratch/project_xxx/mrogalaj/experiments/repo && git worktree remove /scratch/project_xxx/mrogalaj/experiments/worktrees/<commit>"
```
