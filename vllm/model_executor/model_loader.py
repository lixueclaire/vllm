"""Utilities for selecting and loading models."""
import contextlib
from typing import Optional, Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig
import time

from vllm.config import ModelConfig, LoRAConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.weight_utils import (get_quant_config,
                                              initialize_dummy_weights)


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model(model_config: ModelConfig,
              lora_config: Optional[LoRAConfig] = None) -> nn.Module:
    start_time = time.perf_counter()
    model_class = _get_model_architecture(model_config.hf_config)
    get_model_archi_time = time.perf_counter()
    print(f"      get model architecture time {get_model_archi_time - start_time:.2f} seconds.") 

    # Get the (maybe quantized) linear method.
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config.quantization,
                                        model_config.model,
                                        model_config.hf_config,
                                        model_config.download_dir)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")
        linear_method = quant_config.get_linear_method()
    get_linear_method_time = time.perf_counter()
    print(f"      get linear method time {get_linear_method_time - get_model_archi_time:.2f} seconds.") 

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device("cuda"):
            if getattr(model_class, "supports_lora", False):
                model = model_class(model_config.hf_config, linear_method,
                                    lora_config)
            elif lora_config:
                raise ValueError(
                    f"Model {model_class.__name__} does not support LoRA, "
                    "but LoRA is enabled. Support for this model may "
                    "be added in the future. If this is important to you, "
                    "please open an issue on github.")
            else:
                model = model_class(model_config.hf_config, linear_method)
        get_model_class_time = time.perf_counter()
        print(f"        get model class time {get_model_class_time - get_linear_method_time:.2f} seconds.") 
        if model_config.load_format == "dummy":
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model, model_config.download_dir,
                               model_config.load_format, model_config.revision)
        
        load_weights_time = time.perf_counter()
        print(f"        load weights time {load_weights_time - get_model_class_time:.2f} seconds.") 
    return model.eval()

def get_model_without_weights(model_config: ModelConfig,
              lora_config: Optional[LoRAConfig] = None) -> nn.Module:
    start_time = time.perf_counter()
    model_class = _get_model_architecture(model_config.hf_config)
    get_model_archi_time = time.perf_counter()
    print(f"      get model architecture time {get_model_archi_time - start_time:.2f} seconds.") 

    # Get the (maybe quantized) linear method.
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config.quantization,
                                        model_config.model,
                                        model_config.hf_config,
                                        model_config.download_dir)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")
        linear_method = quant_config.get_linear_method()
    get_linear_method_time = time.perf_counter()
    print(f"      get linear method time {get_linear_method_time - get_model_archi_time:.2f} seconds.") 

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device("cuda"):
            if getattr(model_class, "supports_lora", False):
                model = model_class(model_config.hf_config, linear_method,
                                    lora_config)
            elif lora_config:
                raise ValueError(
                    f"Model {model_class.__name__} does not support LoRA, "
                    "but LoRA is enabled. Support for this model may "
                    "be added in the future. If this is important to you, "
                    "please open an issue on github.")
            else:
                model = model_class(model_config.hf_config, linear_method)
        get_model_class_time = time.perf_counter()
        print(f"        get model class time {get_model_class_time - get_linear_method_time:.2f} seconds.") 
        if model_config.load_format == "dummy":
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        # else:
        #    # Load the weights from the cached or downloaded files.
        #    model.load_weights(model_config.model, model_config.download_dir,
        #                      model_config.load_format, model_config.revision)
        
        load_weights_time = time.perf_counter()
        print(f"        load weights time {load_weights_time - get_model_class_time:.2f} seconds.") 
    return model.eval()
