import os
import sys
from pathlib import Path

# Set Python path to current folder
current_dir = str(Path(__file__).parent.parent.parent.parent.parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
os.environ["PYTHONPATH"] = current_dir + ":" + os.environ.get("PYTHONPATH", "")

import subprocess
import torch
import json
from huggingface_hub import snapshot_download
from torch.profiler import ProfilerActivity, profile, record_function
from fastvideo.utils import logger
# Import the training pipeline
from fastvideo.training.wan_training_pipeline import main
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.utils import FlexibleArgumentParser
from fastvideo.training.wan_training_pipeline import WanTrainingPipeline

MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_PATH = "data/crush-smol_processed_t2v/training_dataset/worker_1/worker_0/"
VALIDATION_DATASET_FILE = "examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json"
OUTPUT_DIR = Path("checkpoints/wan_t2v_finetune")
PROFILER_TRACE_ROOT = Path("/mnt/fast-disks/hao_lab/ohm/profiler_traces/wan_t2v_finetune")
WANDB_SUMMARY_FILE = OUTPUT_DIR / "tracker/wandb/latest-run/files/wandb-summary.json"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"
GRAD_ACCUM = "1"
MASTER_PORT = "29604"

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = MASTER_PORT


def run_worker():
    """Worker function that will be run on each GPU"""
    # Create and populate args
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    
    # Set the arguments as they are in finetune_t2v.sh
    args = parser.parse_args([
        "--model_path", MODEL_PATH,
        "--inference_mode", "False",
        "--pretrained_model_name_or_path", MODEL_PATH,
        "--data_path", DATA_PATH,
        "--dataloader_num_workers", "1",
        "--train_batch_size", "4",
        "--train_sp_batch_size", "1",
        "--gradient_accumulation_steps", GRAD_ACCUM,
        "--num_latent_t", "20",
        "--num_height", "720",
        "--num_width", "1280",
        "--num_frames", "77",
        "--enable_gradient_checkpointing_type", "full",
        "--max_train_steps", "20",
        "--learning_rate", "5e-5",
        "--mixed_precision", "bf16",
        "--weight_only_checkpointing_steps", "250",
        "--training_state_checkpointing_steps", "250",
        "--weight_decay", "1e-4",
        "--max_grad_norm", "1.0",
        "--num_euler_timesteps", "50",
        "--multi_phased_distill_schedule", "4000-1",
        "--not_apply_cfg_solver",
        "--training_cfg_rate", "0.1",
        "--ema_start_step", "0",
        "--dit_precision", "fp32",
        "--output_dir", str(OUTPUT_DIR),
        "--tracker_project_name", "wan_t2v_finetune",
        "--checkpoints_total_limit", "3",
        "--validation_dataset_file", VALIDATION_DATASET_FILE,
        "--validation_steps", "200",
        "--validation_sampling_steps", "50",
        "--validation_guidance_scale", "6.0",
        #"--enable_torch_compile",
        #"--log_validation",
        "--num_gpus", NUM_GPUS_PER_NODE,
        "--sp_size", NUM_GPUS_PER_NODE,
        "--tp_size", "1",
        "--hsdp_replicate_dim", NUM_GPUS_PER_NODE,
        "--hsdp_shard_dim", "1"
    ])
    # Call the main training function
    pipeline = WanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args

    # Profile the training run with torch.profiler to estimate FLOPs.
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with record_function("training_step_profiler_run"):
            pipeline.train()

    logger.info("Training pipeline done")

    # Print FLOPs table
    try:
        table = prof.key_averages().table(sort_by="flops", row_limit=20)
        logger.info("Torch profiler FLOPs summary:\n%s", table)
    except Exception as exc:  # Defensive: profiler APIs can differ by torch version
        logger.warning("Failed to summarize torch.profiler results: %s", exc)

    # Numeric FLOPs extraction: sum across all SP ranks via all_reduce
    import torch.distributed as dist
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    max_train_steps = args.max_train_steps
    world_size = int(NUM_GPUS_PER_NODE)
    try:
        rank_flops = sum(
            getattr(e, "flops", 0) or 0 for e in prof.key_averages()
        )
    except Exception:
        rank_flops = 0.0
    flops_t = torch.tensor(rank_flops, dtype=torch.float64)
    try:
        if dist.is_initialized():
            dist.all_reduce(flops_t, op=dist.ReduceOp.SUM)
    except Exception as exc:
        logger.warning("dist.all_reduce failed: %s", exc)
    total_profiler_flops = flops_t.item()
    profiler_flops_per_step = total_profiler_flops / max_train_steps
    logger.info(
        "Profiler FLOPs: total=%.4e  per_step=%.4e",
        total_profiler_flops, profiler_flops_per_step,
    )

    # Compare with formula FLOPs (rank 0 only, reads wandb summary)
    if local_rank == 0 and WANDB_SUMMARY_FILE.exists():
        try:
            with WANDB_SUMMARY_FILE.open() as _f:
                _s = json.load(_f)
            _batch = _s.get("batch_size", 1)
            _seq   = _s.get("dit_seq_len", 0)
            _ctx   = _s.get("context_len", 512)
            _h     = _s.get("hidden_dim", 1536)
            _nl    = _s.get("num_layers", 30)
            _ffn   = _s.get("ffn_dim", 8960)
            _grad  = args.gradient_accumulation_steps
            _fpl = (
                8 * _h * _h * _seq
                + 4 * _h * _h * _seq + 4 * _h * _h * _ctx
                + 4 * _h * _ffn * _seq
                + 4 * _seq * _seq * _h
                + 4 * _seq * _ctx * _h
            )
            formula_flops_per_step = _batch * _fpl * _nl * 4 * _grad
            ratio = (
                profiler_flops_per_step / formula_flops_per_step
                if formula_flops_per_step else 0.0
            )
            logger.info(
                "\n=== Profiler vs Formula Sanity Check ===\n"
                "  formula  FLOPs/step: %.4e\n"
                "  profiler FLOPs/step: %.4e\n"
                "  ratio (profiler/formula): %.4f\n"
                "  (ratio < 1 expected: FlashAttn custom kernels excluded from profiler)",
                formula_flops_per_step, profiler_flops_per_step, ratio,
            )
        except Exception as exc:
            logger.warning("Profiler vs formula comparison failed: %s", exc)

def test_distributed_training(profile=False):
    """Test the distributed training setup"""
    os.environ["WANDB_MODE"] = "online"

    data_dir = Path("data/crush-smol_processed_t2v")
    
    if not data_dir.exists():
        print(f"Downloading test dataset to {data_dir}...")
        snapshot_download(
            repo_id="wlsaidhi/crush-smol_processed_t2v",
            local_dir=str(data_dir),
            repo_type="dataset",
            local_dir_use_symlinks=False
        )
    
    # Get the current file path
    current_file = Path(__file__).resolve()
    
    # Run torchrun command
    cmd = [
        "torchrun",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE,
        "--master_port", MASTER_PORT,
        str(current_file)
    ]
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print stdout and stderr for debugging
    if process.stdout:
        print("STDOUT:", process.stdout)
    if process.stderr:
        print("STDERR:", process.stderr)
    
    # Check if the process failed
    if process.returncode != 0:
        print(f"Process failed with return code: {process.returncode}")
        raise subprocess.CalledProcessError(process.returncode, cmd, process.stdout, process.stderr)

    summary_file = WANDB_SUMMARY_FILE

    with summary_file.open() as f:
        wandb_summary = json.load(f)
    
    # Calculate and print MFU metrics
    device_name = torch.cuda.get_device_name()
    try:
        # Get actual values from training run (logged from training_batch.raw_latent_shape)
        batch_size = wandb_summary.get("batch_size")
        seq_len = wandb_summary.get("dit_seq_len")
        context_len = wandb_summary.get("context_len")
        avg_step_time = wandb_summary.get("avg_step_time")
        hidden_dim = wandb_summary.get("hidden_dim")
        num_layers = wandb_summary.get("num_layers")
        ffn_dim = wandb_summary.get("ffn_dim")


        

        # FLOPs per layer (forward pass)
        # - QKV + out proj: 8 * hidden_dim^2 * seq_len
        # - Cross-attn proj: 4 * hidden_dim^2 * seq_len + 4 * hidden_dim^2 * context_len
        # - MLP: 4 * hidden_dim * ffn_dim * seq_len
        # - Self-attn matmuls: 4 * seq_len^2 * hidden_dim
        # - Cross-attn matmuls: 4 * seq_len * context_len * hidden_dim
        qkv_out_flops = 8 * hidden_dim * hidden_dim * seq_len
        cross_attn_proj_flops = (
            (4 * hidden_dim * hidden_dim * seq_len) +
            (4 * hidden_dim * hidden_dim * context_len) 
        )
        mlp_flops = 4 * hidden_dim * ffn_dim * seq_len
        self_attn_flops = 4 * seq_len * seq_len * hidden_dim
        cross_attn_flops = 4 * seq_len * context_len * hidden_dim
        flops_per_layer = (
            qkv_out_flops + cross_attn_proj_flops + mlp_flops + self_attn_flops + cross_attn_flops
        )

        # With full activation checkpointing: 1 forward + 3 backward (1 recompute + 2 gradient)
        achieved_flops = batch_size * flops_per_layer * num_layers * 4

        
        # Account for gradient accumulation (from config)
        grad_accum = int(GRAD_ACCUM)
        achieved_flops *= grad_accum  

        # Peak FLOPs based on device
        if "H100" in device_name:
            peak_flops_per_gpu = 989e12
        elif "A100" in device_name:
            peak_flops_per_gpu = 312e12
        elif "A40" in device_name:
            peak_flops_per_gpu = 312e12
        elif "L40S" in device_name:
            peak_flops_per_gpu = 362e12
        else:
            raise ValueError(f"Device {device_name} not supported")
        
        # Total peak (2 GPUs)
        world_size = int(NUM_GPUS_PER_NODE)
        total_peak_flops = peak_flops_per_gpu * world_size
        
        # Calculate MFU
        achieved_flops_per_sec = achieved_flops / avg_step_time if avg_step_time > 0 else 0
        mfu = (achieved_flops_per_sec / total_peak_flops * 100) if total_peak_flops > 0 else 0

        print(f"Per-Step MFU: {mfu:.4f}%")
    except Exception as e:
        print(f"Could not calculate MFU: {e}")
    

if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK") is not None:
        # We're being run by torchrun
        run_worker()
    else:
        # We're being run directly
        test_distributed_training()
