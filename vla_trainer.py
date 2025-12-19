# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""


import time
import io
import traceback
from typing import TYPE_CHECKING, List, Optional, Union

# isort: on

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from PIL import Image

from packaging import version
from torch.utils.data import DataLoader, Dataset

from transformers import __version__
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_pt_utils import (
    EvalLoopContainer,
    find_batch_size,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    denumpify_detensorize,
    has_length,
)
from transformers.utils import (
    XLA_FSDPV2_MIN_VERSION,
    is_datasets_available,
    is_in_notebook,
    is_torch_xla_available,
)
from transformers.integrations.tpu import tpu_spmd_dataloader

from transformers.trainer import logger, Trainer

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False



if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets


class VLATrainer(Trainer):

    def __init__(
        self,
        num_eval_datasets: int = 2,
        num_eval_batches: int = 4,
        use_default_collate_fn_for_eval: bool = False,
        processor = None,
        vae = None,
        normalizer = None,
        *args,
        **kwargs
    ):
        """Initializes VLATrainer.

        Args:
            num_eval_datasets (int, optional): Number of evaluation datasets to use. Defaults to 3.
            num_eval_batches (int, optional): Number of batches to evaluate for each dataset. Defaults to 10.
        """
        super().__init__(*args, **kwargs)

        self.num_eval_datasets = num_eval_datasets
        self.num_eval_batches = num_eval_batches
        self.use_default_collate_fn_for_eval = use_default_collate_fn_for_eval

        self.processor = processor
        self.vae = vae
        self.normalizer = normalizer

        print(f"DEBUG: VLATrainer Initialized. Processor: {processor is not None}, VAE: {vae is not None}, Normalizer: {normalizer is not None}")

        # Try to determine VAE codebook size for safety checks
        self.vae_codebook_size = 1024  # Default fallback
        try:
            if hasattr(self.vae, 'n_codes'):
                self.vae_codebook_size = self.vae.n_codes
            elif hasattr(self.vae, 'num_embeddings'):
                self.vae_codebook_size = self.vae.num_embeddings
            # Check for MultiVQVAE specific attributes
            elif hasattr(self.vae, 'vocab_size'):
                self.vae_codebook_size = self.vae.vocab_size
            print(f"DEBUG: Detected VAE codebook size limit: {self.vae_codebook_size}")
        except:
            print("DEBUG: Could not determine VAE codebook size, defaulting to 1024")

        # initialize the index of the evaluation dataset
        # we only evaluate `num_eval_datasets` datasets in a round-robin manner
        self.eval_dataset_index = 0
        self.eval_dataset_names = None
        if isinstance(self.eval_dataset, dict):
            self.eval_dataset_names = sorted(self.eval_dataset.keys())

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )


        # TODO: was
        # collate_fn = (
        #     None if self.use_default_collate_fn_for_eval
        #     else lambda examples: examples
        # )

        if self.use_default_collate_fn_for_eval:
            # If the flag is set, it implies we want standard torch collation, BUT
            # standard torch collation crashes on PIL images.
            # We just force the standard one
            print("DEBUG WARNING: use_default_collate_fn_for_eval is True, but falling back to self.data_collator to avoid PIL errors.")
            collate_fn = self.data_collator
        else:
            collate_fn = self.data_collator

        # NOTE: we use customized collate_fn for evaluation
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": collate_fn,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": False # TODO: set to False, does not work with DDP
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        """
        print(f"DEBUG: Starting evaluation loop: {description}")  # DEBUG
        args = self.args

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        containers = {
            "action_valid_rate": EvalLoopContainer(args.eval_do_concat_batches, padding_index=-100),
            "action_mse_error": EvalLoopContainer(args.eval_do_concat_batches, padding_index=-100),
            "action_mse_error_pos": EvalLoopContainer(args.eval_do_concat_batches, padding_index=-100),
            "action_geodesic_error_rot": EvalLoopContainer(args.eval_do_concat_batches, padding_index=-100),
            "action_mse_error_width": EvalLoopContainer(args.eval_do_concat_batches, padding_index=-100),
        }

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            if step >= self.num_eval_batches:
                print(f"DEBUG: Stopping eval at step {step} due to num_eval_batches limit.")  # DEBUG
                break

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            inputs = self._prepare_inputs(inputs)

            # Log first batch of evaluation to WandB
            if step == 0 and self.is_world_process_zero() and self.args.report_to and "wandb" in self.args.report_to:
                print("DEBUG: Attempting to log EVAL predictions to WandB...")  # DEBUG
                try:
                    with torch.no_grad():
                        # Perform a forward pass purely for visualization logging
                        outputs = model(**inputs)
                        self._log_vla_predictions(inputs, outputs, prefix="eval")
                except Exception as e:
                    logger.warning(f"Failed to log eval VLA predictions: {e}")
                    print(f"DEBUG ERROR: Eval logging failed: {e}")  # DEBUG
                    traceback.print_exc()

            with torch.no_grad():
                # FIX: Unbatch inputs into a list of examples for compute_metrics
                # utils.py expects a list of dicts, but inputs is a dict of batched tensors
                batch_sz = inputs["input_ids"].shape[0]
                examples_list = [
                    {k: v[i] for k, v in inputs.items()}
                    for i in range(batch_sz)
                ]

                metrics_per_step = self.compute_metrics(
                    model=model, examples=examples_list
                )

            if is_torch_xla_available():
                xm.mark_step()

            for key, value in metrics_per_step.items():
                # Update containers
                gathered_values = self.gather_function(value.repeat(batch_size))
                containers[key].add(gathered_values.detach())

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                for key in containers:
                    containers[key].to_cpu_and_numpy()

                del metrics_per_step
                torch.cuda.empty_cache()

            # [Add this small debug print in the loop]
            if step % 10 == 0:
                print(f"DEBUG: Eval Step {step} processed.")

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        for key in containers:
            containers[key] = containers[key].get_arrays()

        # FIXME(lingxuan): it's not right for there may be invalid prediction
        # but there's almost no invalid prediction even during the early phase of training
        # so we do not consider it for now
        num_samples = observed_num_examples

        metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        for metric_key, value in containers.items():
            if isinstance(value, list) and value:
                metrics[f"{metric_key_prefix}_{metric_key}"] = np.concatenate(value).mean().item()
            elif isinstance(value, np.ndarray):
                metrics[f"{metric_key_prefix}_{metric_key}"] = value.mean().item()

        if hasattr(self, "jit_compilation_time" ):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        We only evaluate `num_eval_datasets` datasets in a round-robin manner.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if (
            isinstance(eval_dataset, dict) and
            len(eval_dataset) > self.num_eval_datasets
        ):
            # use round-robin strategy to select `num_eval_datasets` evaluation datasets
            total_eval_datasets = len(self.eval_dataset_names)
            next_eval_dataset_names = [
                self.eval_dataset_names[
                    (self.eval_dataset_index + i) % total_eval_datasets
                ]
                for i in range(self.num_eval_datasets)
            ]

            # select designated evaluation datasets
            eval_dataset = {
                eval_dataset_name: eval_dataset[eval_dataset_name]
                for eval_dataset_name in next_eval_dataset_names
            }
            # maintain the index of the evaluation dataset
            self.eval_dataset_index = (
                (self.eval_dataset_index + self.num_eval_datasets)
                    % total_eval_datasets
            )

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Overridden compute_loss to inject WandB logging for VLA predictions.
        Calls super() to maintain all original Trainer logic (FSDP, deepspeed, etc).
        """
        # DEBUG: Check if this function is called
        if self.state.global_step % 10 == 0 and self.accelerator.is_main_process:
            print(f"DEBUG: Inside compute_loss at step {self.state.global_step}")

        # 1. Create a shallow copy of inputs for logging.
        #    Standard Trainer.compute_loss() might .pop("labels") from 'inputs'.
        logging_inputs = {k: v for k, v in inputs.items()}

        # 2. Call the parent loss computation
        #    Force return_outputs=True to get logits for logging.
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        # 3. Custom Logging Logic
        #    Check strictly for main process and logging steps to avoid slowdowns.
        if (self.args.report_to and "wandb" in self.args.report_to and
                self.state.global_step % self.args.logging_steps == 0 and
                self.accelerator.is_main_process):

            print(f"DEBUG: Triggering Training Visualization at step {self.state.global_step}")  # DEBUG
            try:
                self._log_vla_predictions(logging_inputs, outputs, prefix="train")
            except Exception as e:
                # Don't crash training if logging fails
                logger.warning(f"Failed to log VLA predictions to WandB: {e}")
                print(f"DEBUG ERROR: Train logging failed: {e}")  # DEBUG
                traceback.print_exc()

        # 4. Return what the caller expected
        return (loss, outputs) if return_outputs else loss

    def _log_vla_predictions(self, inputs, outputs, prefix="train"):
        """
        Helper to decode and log VLA inputs/outputs to WandB.
        Requires processor, vae, and normalizer to be attached to self.
        """
        print(f"DEBUG: Entering _log_vla_predictions ({prefix})")  # DEBUG

        # Ensure helper objects exist (must be attached in train.py)
        if not hasattr(self, 'processor') or not hasattr(self, 'vae') or not hasattr(self, 'normalizer'):
            print("DEBUG FAIL: Missing processor, vae, or normalizer in Trainer.")  # DEBUG
            return

        tokenizer = self.processor.tokenizer
        vae = self.vae
        normalizer = self.normalizer

        # Use the first sample in the batch
        idx = 0
        input_ids = inputs["input_ids"][idx]

        # Determine labels: check inputs first, then fallback to self.label_names if standard trainer logic removed them
        labels = inputs.get("labels")
        if labels is None:
            print("DEBUG FAIL: 'labels' not found in inputs dict.")  # DEBUG
            print(f"DEBUG: Available keys: {inputs.keys()}")
            return

            # outputs.logits shape: [batch, seq_len, vocab_size]
        pred_logits = outputs.logits[idx]
        pred_token_ids = torch.argmax(pred_logits, dim=-1)

        # --- 1. Decode Text (Instruction) ---
        # Find where the assistant response starts (where labels are not -100)
        response_mask = (labels[idx] != -100)
        if not response_mask.any():
            print("DEBUG FAIL: No valid labels (all -100) for this sample.")  # DEBUG
            return

            # Everything before the response is the instruction
        response_start_idx = torch.where(response_mask)[0][0]
        instruction_text = tokenizer.decode(input_ids[:response_start_idx], skip_special_tokens=True)
        print(f"DEBUG: Instruction detected: {instruction_text[:50]}...")  # DEBUG

        # --- 2. Decode Actions ---
        gt_segment_ids = labels[idx][response_mask]
        pred_segment_ids = pred_token_ids[response_mask]

        print(f"DEBUG: GT Segment shape: {gt_segment_ids.shape}, Pred Segment shape: {pred_segment_ids.shape}")

        vocab_size = tokenizer.vocab_size

        def decode_actions_from_tokens(token_ids):
            # Reverse collator mapping: action_tokens = vocab_size - token_id - 1
            # Note: This logic depends on the specific collator used in train.py
            vae_indices = vocab_size - token_ids - 1

            # FIX: Filter indices strictly within [0, codebook_size)
            # The model might hallucinate tokens that result in valid positive integers
            # but are larger than the VAE codebook (e.g. text tokens).
            valid_mask = (vae_indices >= 0) & (vae_indices < self.vae_codebook_size)

            if not valid_mask.any():
                print("DEBUG: No valid VAE indices found after reversing vocab.")  # DEBUG
                return None

            valid_indices = vae_indices[valid_mask]

            with torch.no_grad():
                try:
                    # Decode via VAE. Expected input often [batch, seq_len] or [seq_len]
                    if valid_indices.dim() == 1:
                        valid_indices = valid_indices.unsqueeze(0)

                    # DEBUG: FIX for AttributeError 'MultiVQVAE' object has no attribute 'device'
                    # We access the device via parameters since VAE is a custom module
                    vae_device = next(vae.parameters()).device

                    if valid_indices.device != vae_device:
                        valid_indices = valid_indices.to(vae_device)

                    actions = vae.decode(valid_indices)
                    # Result is [1, seq_len, action_dim], squeeze batch
                    actions = actions.squeeze(0)
                except Exception as e:
                    print(f"DEBUG ERROR during VAE decode: {e}")
                    return None
            return actions

        gt_actions = decode_actions_from_tokens(gt_segment_ids)
        pred_actions = decode_actions_from_tokens(pred_segment_ids)

        if gt_actions is None or pred_actions is None:
            print("DEBUG FAIL: GT or Pred actions could not be decoded.")  # DEBUG
            return

        # --- 3. Un-normalize ---
        # Move to CPU and numpy
        try:
            gt_actions = normalizer.unnormalize(gt_actions).float().cpu().numpy()
            pred_actions = normalizer.unnormalize(pred_actions).float().cpu().numpy()
        except Exception as e:
            print(f"DEBUG ERROR during normalization: {e}")
            return

        # --- 4. Plotting ---
        # Dynamically determine dimensions
        if len(gt_actions.shape) < 2:
            print(f"DEBUG FAIL: Action shape invalid: {gt_actions.shape}")  # DEBUG
            return  # Safety check

        print(f"DEBUG: Generating plot. Dims: {gt_actions.shape[1]}")  # DEBUG

        num_dims = gt_actions.shape[1]
        fig, axes = plt.subplots(num_dims, 1, figsize=(8, 2 * num_dims), sharex=True)
        if num_dims == 1: axes = [axes]

        for d in range(num_dims):
            axes[d].plot(gt_actions[:, d], label="Ground Truth", color="black", linestyle="--", alpha=0.7)

            # Crop prediction if lengths differ (e.g. if one stopped early or ran longer)
            min_len = min(len(gt_actions), len(pred_actions))
            axes[d].plot(pred_actions[:min_len, d], label="Prediction", color="blue")

            axes[d].set_ylabel(f"Dim {d}")
            if d == 0: axes[d].legend(loc="upper right")

        plt.suptitle(f"Step {self.state.global_step}")
        plt.tight_layout()

        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_image = Image.open(buf)
        plt.close(fig)

        # Log to wandb
        print(f"DEBUG: Sending to WandB ({prefix})...")  # DEBUG
        wandb.log({
            f"{prefix}/action_plot": wandb.Image(plot_image, caption=instruction_text[:100]),
        }, step=self.state.global_step)
        print("DEBUG: WandB log successful.")