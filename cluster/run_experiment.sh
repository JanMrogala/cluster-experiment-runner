#!/bin/bash
# Standalone cluster-side script for manual use / debugging.
# Normally the CER tool sends this logic inline via SSH.
#
# Usage: ./run_experiment.sh <commit_hash>
#
# Required env vars:
#   CER_BASE_DIR        - Root directory (e.g. /raid/user/experiments)
#   CER_REPO_URL        - Git repo SSH URL
#   CER_BRANCH          - Branch to fetch (default: main)
#   CER_CONTAINER_IMAGE - Path to .sif file
#   CER_BIND_MOUNTS     - Comma-separated bind mounts (optional)
#   CER_PARTITION        - SLURM partition (default: gpu)
#   CER_GRES            - SLURM gres (default: gpu:1)
#   CER_CPUS            - CPUs per task (default: 8)
#   CER_MEM             - Memory (default: 32G)
#   CER_TIME            - Time limit (default: 24:00:00)
#   CER_ENTRYPOINT      - Command to run (default: python train.py)
#   CER_WANDB_PROJECT   - W&B project name
#   CER_CONFIG_JSON     - JSON config overrides (optional)

set -euo pipefail

COMMIT="${1:?Usage: $0 <commit_hash>}"
COMMIT_SHORT="${COMMIT:0:8}"

BASE_DIR="${CER_BASE_DIR:?CER_BASE_DIR required}"
REPO_URL="${CER_REPO_URL:?CER_REPO_URL required}"
BRANCH="${CER_BRANCH:-main}"
CONTAINER_IMAGE="${CER_CONTAINER_IMAGE:?CER_CONTAINER_IMAGE required}"
BIND_MOUNTS="${CER_BIND_MOUNTS:-}"
PARTITION="${CER_PARTITION:-gpu}"
GRES="${CER_GRES:-gpu:1}"
CPUS="${CER_CPUS:-8}"
MEM="${CER_MEM:-32G}"
TIME="${CER_TIME:-24:00:00}"
ENTRYPOINT="${CER_ENTRYPOINT:-python train.py}"
WANDB_PROJECT="${CER_WANDB_PROJECT:-}"
WANDB_ENTITY="${CER_WANDB_ENTITY:-}"
CONFIG_JSON="${CER_CONFIG_JSON:-{}}"

REPO_DIR="$BASE_DIR/repo"
WORKTREE_DIR="$BASE_DIR/worktrees/$COMMIT"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$BASE_DIR" "$LOG_DIR"

# Clone bare repo if not present
if [ ! -d "$REPO_DIR/HEAD" ]; then
    echo "Cloning bare repo..."
    git clone --bare "$REPO_URL" "$REPO_DIR"
fi

# Fetch with retry until commit is available
MAX_RETRIES=30
RETRY_DELAY=10
echo "Fetching commit $COMMIT_SHORT..."
for i in $(seq 1 $MAX_RETRIES); do
    git -C "$REPO_DIR" fetch origin "$BRANCH" --quiet 2>/dev/null || true
    if git -C "$REPO_DIR" cat-file -t "$COMMIT" >/dev/null 2>&1; then
        echo "Commit found."
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "ERROR: Commit $COMMIT not found after $MAX_RETRIES retries" >&2
        exit 1
    fi
    echo "Commit not yet available, retrying in ${RETRY_DELAY}s... ($i/$MAX_RETRIES)"
    sleep $RETRY_DELAY
done

# Create worktree
if [ -d "$WORKTREE_DIR" ]; then
    git -C "$REPO_DIR" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
fi
git -C "$REPO_DIR" worktree add --detach "$WORKTREE_DIR" "$COMMIT"

# Build bind flags
BIND_FLAGS=""
if [ -n "$BIND_MOUNTS" ]; then
    IFS=',' read -ra MOUNTS <<< "$BIND_MOUNTS"
    for m in "${MOUNTS[@]}"; do
        BIND_FLAGS="$BIND_FLAGS --bind $m"
    done
fi

# Write sbatch script
cat > "$WORKTREE_DIR/_run.sbatch" << EOF
#!/bin/bash
#SBATCH --job-name=cer-$COMMIT_SHORT
#SBATCH --partition=$PARTITION
#SBATCH --gres=$GRES
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOG_DIR/%j.out
#SBATCH --error=$LOG_DIR/%j.err

export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_ENTITY="$WANDB_ENTITY"
export WANDB_RUN_NAME="cer-$COMMIT_SHORT"
export WANDB_TAGS="$COMMIT"
export CER_COMMIT="$COMMIT"
export CER_CONFIG='$CONFIG_JSON'

cd "$WORKTREE_DIR"

singularity exec \\
    --nv \\
    $BIND_FLAGS \\
    $CONTAINER_IMAGE \\
    bash -c 'pip install -q -r requirements.txt 2>/dev/null; $ENTRYPOINT'
EOF

sbatch "$WORKTREE_DIR/_run.sbatch"
