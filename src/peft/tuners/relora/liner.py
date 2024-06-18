import os
import math
import json
from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
import bitsandbytes.functional as bnbF

from transformers import AutoModelForCausalLM, AutoConfig

from loguru import logger


# The code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class ReLoRaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: int = 1,
        lora_dropout: float = 0.1,
        lora_only: bool = False,
        weight_data=None,
        bias_data=None,
        trainable_scaling: bool = False,
        bias=True,
        device=None,
        dtype=None,
        quantize=False,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
    ):
        """Wraps linear layer x W into x W + x W_a @ W_b * lora_alpha / r
        
        Notice that scale = lora_alpha / r.
        """
        nn.Module.__init__(self)
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        if lora_only:
            self.weight = None
            self.bias = None
        else:
            # if full model weight + lora weight
            if bias_data is None:
                bias_data = torch.zeros(out_features, device=device, dtype=dtype, requires_grad=True) if bias else None
            self.bias = nn.Parameter(bias_data) if bias else None

            if weight_data is None:
                # note that our trainable weight are W_a and W_b
                weight_data = torch.zeros(out_features, in_features, device=device, dtype=dtype, requires_grad=False)

            if quantize is None:
                self.weight = nn.Parameter(weight_data, requires_grad=False)
            elif quantize == "4bit":
                self.weight = bnb.nn.Params4bit(
                    weight_data,
                    requires_grad=False,
                    compress_statistics=bnb_4bit_use_double_quant,
                    quant_type=bnb_4bit_quant_type,
                )
            elif quantize == "8bit":
                logger.warning("Int8 currently does not support merge_and_reinit! It will fail")
                self.weight = bnb.nn.Int8Params(
                    weight_data,
                    requires_grad=False,
                )
            else:
                raise ValueError(f"Unknown quantize type: {quantize}")

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.lora_only = lora_only
        self.trainable_scaling = trainable_scaling
        self.quantize = quantize

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.zeros_(self.lora_B.weight)
            if trainable_scaling:
                self.scaling = nn.Parameter(torch.tensor([1.]), requires_grad=True)
            else:
                self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            if not self.lora_only:
                self.weight.requires_grad = False
    
    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()

        return self.scaling

    @torch.no_grad()
    def merge_and_reinit(self):
        if self.lora_only:
            print("WARNING: Skipping merge and reinit, because only lora parameters are used")
            return

        if not self.quantize:
            self.weight.data += self.lora_B.weight @ self.lora_A.weight * self._post_lora_scale()
        elif self.quantize == "4bit":
            self.weight: bnb.nn.Params4bit
            _weight_fp = torch.empty(self.weight.data.shape, dtype=self.lora_B.weight.dtype, device=self.weight.data.device)
            bnbF.dequantize_4bit(self.weight.data, self.weight.quant_state, out=_weight_fp)
            _weight_fp += self.lora_B.weight @ self.lora_A.weight * self._post_lora_scale()
            self.weight.data, self.weight.quant_state = bnbF.quantize_4bit(
                _weight_fp,
                quant_type=self.weight.quant_type,
                compress_statistics=self.weight.compress_statistics,
            )
            del _weight_fp
        elif self.quantize == "8bit":
            self.weight: bnb.nn.Int8Params
            _weight_fp = torch.empty(self.weight.data.shape, dtype=torch.bfloat16, device=self.weight.data.device)
            # !out assigned inplace
            bnbF.dequantize_blockwise(self.weight.data, self.self.lora_B.weight.dtype, out=_weight_fp)
            _weight_fp += self.lora_B.weight @ self.lora_A.weight * self._post_lora_scale()
            self.weight.data, self.weight.quant_state = bnbF.quantize_blockwise(
                _weight_fp,
                self.weight.quant_state,
                out=self.weight.data,
            )
            del _weight_fp
        else:
            raise ValueError(f"Unknown quantize type: {self.quantize}")

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        nn.init.zeros_(self.lora_B.weight)
        if self.trainable_scaling:
            nn.init.zeros_(self.scaling)

    def forward(self, x: torch.Tensor):
        if self.lora_only:
            # just lora
            return self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale()

        if self.quantize == "4bit":
            result = bnb.matmul_4bit(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        elif self.quantize == "8bit":
            result = bnb.matmul(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        else:
            result = F.linear(x, self.weight, bias=self.bias)

        if self.r > 0:
            result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale()
        return result