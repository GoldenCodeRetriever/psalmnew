import os
import torch
import shutil
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch.distributed as dist
from typing import Optional
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available, is_torch_tpu_available,is_accelerate_available
if is_apex_available():
    from apex import amp
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union



import torch

from packaging import version
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


class LLaVATrainer(Trainer):
    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.
    #
    #     Subclass and override to inject custom behavior.
    #
    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #
    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #     if dist.is_available():
    #         dist.barrier()
    #     import ipdb;ipdb.set_trace()
    #     if hasattr(self.train_dataset,'cur_dataset_index'):
    #         self.train_dataset.update_dataset_index()
    #     print(self.train_dataset.cur_dataset_index)
    #
    #
    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)
    #
    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)
    #
    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #
    #     if self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         self.accelerator.backward(loss)
    #
    #     return loss.detach() / self.args.gradient_accumulation_steps

    
    def _save_checkpoint(self, model, trial, metrics=None):
        # 部分微调时仅保存适配器权重
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            # 非部分微调时，调用父类方法保存完整模型
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
    
    # 配合部分微调的模型保存逻辑
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    # 维护多损失的历史记录
    def update_history_loss_dict(self,outputs):
        if not hasattr(self,'history_loss_dict'):
            self.history_loss_dict = {}
        for name, value in outputs.items():
            if 'loss' in name and name != 'loss':
                if name not in self.history_loss_dict:
                    self.history_loss_dict[name] = value.item()
                else:
                    if value != 0:
                        self.history_loss_dict[name] = value.item()


    def compute_loss(self, model, inputs, return_outputs=False):
        """
                How the loss is computed by Trainer. By default, all models return the loss in the first element.

                Subclass and override for custom behavior.
                """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if isinstance(outputs, dict) and 'loss_dice' in outputs:
                loss_dict = {}
                for name,value in outputs.items():
                    if 'loss' in name and name != 'loss':
                        loss_value = value.item()
                        if loss_value == 0 and hasattr(self,'history_loss_dict'):
                            loss_value = self.history_loss_dict[name]
                        loss_dict[name] = loss_value
                self.update_history_loss_dict(outputs)
                # loss_mask = outputs["loss_mask"].item() if isinstance(outputs, dict) else 0
                # loss_dice = outputs["loss_dice"].item() if isinstance(outputs, dict) else 0
                # loss_SEG_class = outputs["loss_SEG_class"].item() if isinstance(outputs, dict) else 0
                # loss_class_name_class = outputs["loss_class_name_class"].item() if isinstance(outputs, dict) else 0
                # loss_dict = {
                #     'loss_mask':loss_mask,
                #     'loss_dice': loss_dice,
                #     'loss_SEG_class':loss_SEG_class,
                #     'loss_class_name_class': loss_class_name_class
                # }
                self.log(loss_dict)

        return (loss, outputs) if return_outputs else loss
    
    
    # def training_step(self, model, inputs):
    #     if self.args.local_rank == 0 and self.state.global_step % 100 == 0:
    #         self._save_visualizations(inputs, model)
        
    #     return super().training_step(model, inputs)
    
    # def _save_visualizations(self, inputs, model):
    #     """保存掩码可视化"""
    #     save_dir = "./outputs/mask_visualizations"
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # 获取图像和预测掩码
    #     if 'image' in inputs:
    #         image = inputs['image'][0]  # batch 中第一个样本
            
    #         # 转换为 numpy（如果是 tensor）
    #         if isinstance(image, torch.Tensor):
    #             image = image.cpu().numpy()
            
    #         # 如果是 CHW 格式，转换为 HWC
    #         if image.shape[0] in [1, 3, 4]:
    #             image = image.transpose(1, 2, 0)
            
    #         # 归一化到 0-255
    #         image = (image * 255).astype(np.uint8)
            
    #         # 获取预测掩码（如果有）
    #         if 'instances' in inputs:
    #             instances = inputs['instances'][0]
    #             if hasattr(instances, 'pred_masks'):
    #                 mask = instances.pred_masks[0].cpu().numpy()
                    
    #                 # 创建组合图
    #                 combined = self._create_combined_image(image, mask)
                    
    #                 # 保存
    #                 save_path = os.path.join(
    #                     save_dir, 
    #                     f"step_{self.state.global_step}_mask.png"
    #                 )
    #                 Image.fromarray(combined).save(save_path)
    
    # @staticmethod
    # def _create_combined_image(image, mask, alpha=0.5):
    #     """创建原图 + 掩码叠加图"""
    #     # 确保 mask 是二值的
    #     if mask.dtype != np.uint8:
    #         mask = (mask > 0.5).astype(np.uint8) * 255
        
    #     # 转换 mask 为 RGB
    #     mask_rgb = np.zeros_like(image)
    #     mask_rgb[mask > 0] = [0, 255, 0]  # 绿色
        
    #     # 叠加
    #     combined = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
        
    #     return combined.astype(np.uint8)
    


    # def training_step(self, model, inputs) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.
    #
    #     Subclass and override to inject custom behavior.
    #
    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #
    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #
    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)
    #
    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)
    #
    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #
    #     if self.do_grad_scaling:
    #         self.scaler.scale(loss).backward()
    #     elif self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         self.accelerator.backward(loss)
    #
    #     return loss.detach() / self.args.gradient_accumulation_steps