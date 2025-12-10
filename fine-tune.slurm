#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=rdt2_finetune
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200sxm:2
#SBATCH --mem=200G
#SBATCH --account=cai_cv
#SBATCH --partition=gpu_top
#SBATCH --output=/cluster/home/%u/.logs/slurm/RDT2_%j_%N.out
#SBATCH --error=/cluster/home/%u/.logs/slurm/RDT2_%j_%N.err

# Exit immediately if any command fails
set -e

# --- 1. Environment & Path Setup ---
export BASE_SCRATCH="/scratch"
export WORK_DIR="$SLURM_SUBMIT_DIR"
export PERSISTENT_DATA_DIR="$WORK_DIR/data_shards"
export SCRATCH_DATA_ROOT="$BASE_SCRATCH/rdt2_data"
export JOB_VENV="$BASE_SCRATCH/venvs/rdt2_node_venv"
export OUTPUT_DIR="$BASE_SCRATCH/outputs/vqvla-sft-${SLURM_JOB_ID}"
export FINAL_OUTPUT_DIR="$WORK_DIR/results/${SLURM_JOB_ID}"
export HF_HOME="$BASE_SCRATCH/hf_cache"
export UV_CACHE_DIR="$BASE_SCRATCH/uv_cache_shared"
export TORCH_HOME="$BASE_SCRATCH/torch_cache"
export WANDB_DIR="$BASE_SCRATCH/wandb"

# Optimizations
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MAX_JOBS=8
export WANDB_PROJECT="rdt2_finetune"
export WANDB_ENTITY="zhaw-physical-ai"
export WANDB_WATCH="false"

# --- 2. Safety & Cleanup ---
cleanup_and_sync() {
    set +e
    echo "--- END OF JOB DETECTED: SYNCING RESULTS ---"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Syncing from: $OUTPUT_DIR"
        mkdir -p "$FINAL_OUTPUT_DIR"
        rsync -av --update "$OUTPUT_DIR/" "$FINAL_OUTPUT_DIR/"
    else
        echo "Output directory skipping sync (not found)."
    fi
    echo "--- SYNC COMPLETE ---"
}
trap cleanup_and_sync EXIT SIGTERM ERR

# --- 3. Module Loading ---
module load python/3.10
module load uv/0.6.12
module load cuda/12.4

# --- 4. Environment Creation (Optimized) ---
# OPTIMIZATION: Only create venv if it doesn't exist. Saves 3-5 mins.
if [ ! -d "$JOB_VENV" ]; then
    echo "--- CREATING NEW VENV ---"
    uv venv "$JOB_VENV" --python 3.10
    UV_PIP="uv pip install -p $JOB_VENV/bin/python"

    $UV_PIP ninja packaging setuptools wheel hf_transfer huggingface_hub[cli]
    $UV_PIP torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    # $UV_PIP flash-attn --no-build-isolation  # (Commented out due to build issues on cluster)
    $UV_PIP accelerate

    grep -v "vllm" requirements.txt > requirements_train_no_vllm.txt
    $UV_PIP -r requirements_train_no_vllm.txt

    $UV_PIP --upgrade --force-reinstall nvidia-nccl-cu12==2.27.5
    $UV_PIP transformers==4.51.3
else
    echo "--- USING EXISTING VENV ($JOB_VENV) ---"
fi

echo "Skipping ur5e.txt (Not needed for training)"

# --- 5. DATA MANAGEMENT ---
echo "--- DATA PREPARATION ---"

# 5.1 Download (Only if missing)
if [ ! -d "$PERSISTENT_DATA_DIR" ]; then
    mkdir -p "$PERSISTENT_DATA_DIR"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    "$JOB_VENV/bin/huggingface-cli" download robotics-diffusion-transformer/BimanualUR5eExample \
        --repo-type dataset --local-dir "$PERSISTENT_DATA_DIR" --local-dir-use-symlinks False

    wget -nc -O "$PERSISTENT_DATA_DIR/normalizer.pt" "http://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt"
fi

# 5.2 Fast Copy to Scratch
mkdir -p "$SCRATCH_DATA_ROOT"
rsync -a --info=progress2 "$PERSISTENT_DATA_DIR/" "$SCRATCH_DATA_ROOT/"

# 5.3 Manual Instruction File Creation
# REMOVED: 50 lines of Python code.
# ADDED: Simple file creation. (This assumes the key is always 'language_instruction')
INSTRUCTION_FILE="$SCRATCH_DATA_ROOT/instruction.json"
if [ ! -f "$INSTRUCTION_FILE" ]; then
    echo "Creating instruction.json..."
    # Note: 'language_instruction' is the standard key for this dataset.
    # If your dataset changes, verify this key.
    echo '{"language_instruction": "Pick up the object and place it correctly."}' > "$INSTRUCTION_FILE"
fi

# 5.4 Generate Config
REAL_SHARDS_DIR=$(find "$SCRATCH_DATA_ROOT" -name "*.tar" -print -quit | xargs dirname)
GENERATED_CONFIG="$SCRATCH_DATA_ROOT/slurm_config.yaml"

cat <<EOF > "$GENERATED_CONFIG"
name: bimanual/ur_example
type: single
shards_dir: ${REAL_SHARDS_DIR}
kwargs:
  instruction_path: ${INSTRUCTION_FILE}
  normalizer_path: ${SCRATCH_DATA_ROOT}/normalizer.pt
EOF

# --- 6. Execution ---
TASK="bimanual-ur5e-example"
export TOKENIZER_ID="Qwen/Qwen2.5-VL-7B-Instruct"
export VAE_ID="robotics-diffusion-transformer/RVQActionTokenizer"
export MODEL_ID="robotics-diffusion-transformer/RDT2-VQ"

if [ ! -d "$OUTPUT_DIR" ]; then mkdir -p "$OUTPUT_DIR"; fi

# 1. Auto-detect Number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[ -z "$NUM_GPUS" ] && NUM_GPUS=1
echo "Detected $NUM_GPUS GPUs."

# 2. Configuration Targets
# Constraint: You can only fit 32 per GPU.
PHYSICAL_BATCH_PER_GPU=16
# Goal: Match the stability of the original config (Batch=64).
TARGET_GLOBAL_BATCH_SIZE=64
BASE_LR=1e-5

# 3. Calculate Gradient Accumulation & LR
# Total physical throughput per step = 32 * Num_GPUs
CURRENT_GLOBAL_BATCH=$((PHYSICAL_BATCH_PER_GPU * NUM_GPUS))

# Calculate Grad Accumulation: Ensure we hit at least target 64
if [ $CURRENT_GLOBAL_BATCH -lt $TARGET_GLOBAL_BATCH_SIZE ]; then
    # e.g., 1 GPU: 32 < 64 -> Need 2 steps
    GRAD_ACCUM_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / CURRENT_GLOBAL_BATCH))
else
    # e.g., 2 GPUs: 64 >= 64 -> Need 1 step
    GRAD_ACCUM_STEPS=1
fi

# Calculate Final Effective Batch Size
EFFECTIVE_BATCH_SIZE=$((CURRENT_GLOBAL_BATCH * GRAD_ACCUM_STEPS))

# Scale LR linearly: New_LR = Base_LR * (Effective_Batch / 64)
# Python used for accurate floating point math
LEARNING_RATE=$(python3 -c "print(f'{${BASE_LR} * (${EFFECTIVE_BATCH_SIZE} / 64.0)}')")

echo "--- HYPERPARAMETER CONFIG ---"
echo "Constraint (Per GPU): $PHYSICAL_BATCH_PER_GPU"
echo "Target Global Batch:  $TARGET_GLOBAL_BATCH_SIZE"
echo "Active GPUs:          $NUM_GPUS"
echo "-----------------------------"
echo "Gradient Accum Steps: $GRAD_ACCUM_STEPS"
echo "Effective Batch Size: $EFFECTIVE_BATCH_SIZE"
echo "Learning Rate:        $LEARNING_RATE"
echo "-----------------------------"

echo "Starting Accelerate Launch..."

"$JOB_VENV/bin/python" -u -m accelerate.commands.launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision="bf16" \
    main.py \
    --attn_implementation="sdpa" \
    --gradient_checkpointing \
    --tokenizer_name=$TOKENIZER_ID \
    --vae_name=$VAE_ID \
    --pretrained_model_name_or_path=$MODEL_ID \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=$PHYSICAL_BATCH_PER_GPU \
    --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
    --eval_batch_size=32 \
    --max_train_steps=10000 \
    --eval_strategy="no" \
    --logging_steps=25 \
    --checkpoints_total_limit=20 \
    --checkpointing_step=1000 \
    --lr_scheduler="cosine" \
    --learning_rate=$LEARNING_RATE \
    --dataloader_num_workers=16 \
    --log_level="info" \
    --report_to="wandb" \
    --lr_warmup_steps=500 \
    --dataset="$GENERATED_CONFIG" \
    --image_corruption \
    --use_default_collate_fn_for_eval

# Old arguments we removed for ZHAW DGX:
#    --gradient_checkpointing \
#    --deepspeed="scripts/zero1.json" \
