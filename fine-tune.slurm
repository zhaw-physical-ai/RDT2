#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=rdt2_train
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

# --- USER CONFIGURATION ---
# Options: "full", "lora", "rdt"
TRAIN_MODE="full"

# --- 1. Environment & Path Setup ---
export BASE_SCRATCH="/scratch"
export WORK_DIR="$SLURM_SUBMIT_DIR"
export PERSISTENT_DATA_DIR="$WORK_DIR/data_shards"
export SCRATCH_DATA_ROOT="$BASE_SCRATCH/rdt2_data"
export JOB_VENV="$BASE_SCRATCH/venvs/rdt2_node_venv"

# Compiler flags (Required for RDT Custom Kernels)
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"

# Dynamic Output Directory based on mode
export OUTPUT_DIR="$BASE_SCRATCH/outputs/rdt2-${TRAIN_MODE}-${SLURM_JOB_ID}"
export FINAL_OUTPUT_DIR="$WORK_DIR/results/${TRAIN_MODE}/${SLURM_JOB_ID}"
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

# Dynamic WandB Name
export WANDB_NAME="${TRAIN_MODE}_${SLURM_JOB_ID}"

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
INSTRUCTION_FILE="$SCRATCH_DATA_ROOT/instruction.json"
if [ ! -f "$INSTRUCTION_FILE" ]; then
    echo "Creating instruction.json..."
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

# --- 6. Execution Setup ---
export TOKENIZER_ID="Qwen/Qwen2.5-VL-7B-Instruct"
export VAE_ID="robotics-diffusion-transformer/RVQActionTokenizer"
export MODEL_ID="robotics-diffusion-transformer/RDT2-VQ"

if [ ! -d "$OUTPUT_DIR" ]; then mkdir -p "$OUTPUT_DIR"; fi

# 1. Auto-detect Number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[ -z "$NUM_GPUS" ] && NUM_GPUS=1
echo "Detected $NUM_GPUS GPUs."


# 3. Determine Settings based on Mode
if [ "$TRAIN_MODE" == "lora" ]; then
    PHYSICAL_BATCH_PER_GPU=64
    TARGET_GLOBAL_BATCH_SIZE=96
    BASE_LR=1e-4
    BASE_BATCH_SCALING=96 # Normalize against the template's target
elif [ "$TRAIN_MODE" == "rdt" ]; then
    PHYSICAL_BATCH_PER_GPU=64
    TARGET_GLOBAL_BATCH_SIZE=64
    BASE_LR=1e-4
    BASE_BATCH_SCALING=64
else
    # Full Fine-tune (Default)
    PHYSICAL_BATCH_PER_GPU=16
    TARGET_GLOBAL_BATCH_SIZE=64
    BASE_LR=1e-5
    BASE_BATCH_SCALING=64
fi

# 4. Calculate Gradient Accumulation
CURRENT_GLOBAL_BATCH=$((PHYSICAL_BATCH_PER_GPU * NUM_GPUS))

if [ $CURRENT_GLOBAL_BATCH -lt $TARGET_GLOBAL_BATCH_SIZE ]; then
    GRAD_ACCUM_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / CURRENT_GLOBAL_BATCH))
else
    GRAD_ACCUM_STEPS=1
fi

EFFECTIVE_BATCH_SIZE=$((CURRENT_GLOBAL_BATCH * GRAD_ACCUM_STEPS))

# 5. Scale LR (Only for 'full' mode usually, but applying logic to all for consistency)
# For LoRA/RDT we respect the explicit 1e-4 base more, but scaling is still good practice.
LEARNING_RATE=$(python3 -c "print(f'{${BASE_LR} * (${EFFECTIVE_BATCH_SIZE} / ${BASE_BATCH_SCALING}.0)}')")

echo "--- HYPERPARAMETER CONFIG ---"
echo "Mode:                 $TRAIN_MODE"
echo "Constraint (Per GPU): $PHYSICAL_BATCH_PER_GPU"
echo "Target Global Batch:  $TARGET_GLOBAL_BATCH_SIZE"
echo "Gradient Accum Steps: $GRAD_ACCUM_STEPS"
echo "Effective Batch Size: $EFFECTIVE_BATCH_SIZE"
echo "Learning Rate:        $LEARNING_RATE"
echo "WandB Name:           $WANDB_NAME"
echo "-----------------------------"

echo "Starting Accelerate Launch..."

# --- 7. Execution Logic ---

if [ "$TRAIN_MODE" == "rdt" ]; then
    # === RDT TRAINING MODE ===
    export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

    # Assumes configs/rdt/post_train.yaml exists in your repo
    RDT_CONFIG_PATH="./configs/rdt/post_train.yaml"

    "$JOB_VENV/bin/python" -u -m accelerate.commands.launch \
        --num_processes=$NUM_GPUS \
        --mixed_precision="bf16" \
        rdt/main.py \
        --attn_implementation="sdpa" \
        --config_path="$RDT_CONFIG_PATH" \
        --pretrained_vision_language_model_name_or_path=$MODEL_ID \
        --output_dir=$OUTPUT_DIR \
        --webdataset_config="$GENERATED_CONFIG" \
        --train_batch_size=$PHYSICAL_BATCH_PER_GPU \
        --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
        --sample_batch_size=32 \
        --max_train_steps=1000000 \
        --checkpointing_period=5000 \
        --sample_period=1000 \
        --checkpoints_total_limit=40 \
        --lr_scheduler="constant" \
        --learning_rate=$LEARNING_RATE \
        --dataloader_num_workers=16 \
        --image_aug \
        --report_to="wandb"

elif [ "$TRAIN_MODE" == "lora" ]; then
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
        --use_lora \
        --use_default_collate_fn_for_eval

else
    # === FULL FINE-TUNING MODE (Default) ===
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
fi

# Old arguments we removed for ZHAW DGX:
#    --deepspeed="scripts/zero1.json" \