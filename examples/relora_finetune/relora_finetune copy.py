import torch

import transformers
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

import datasets
import wandb
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE
dataset_id = "roneneldan/TinyStories"
model_id = "Maykeye/TinyLLama-v0"
BATCH_SIZE = 24
MAX_LENGTH = 128
USE_PEFT = True
TRAIN_LN = True
NUM_TRAINING_STEPS = 10_000

def training_function():

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset_train = load_dataset(dataset_id, data_files={"train": "TinyStories-train.txt"}, split="train")
    dataset_validation = load_dataset(dataset_id, data_files={"test": "TinyStories-valid.txt"}, split="test")

    device = "cuda:1"

    model_config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_config(model_config)

    if USE_PEFT:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

    model = get_peft_model(peft_config, model)

    for name, param in model.named_parameters():
        if TRAIN_LN and "ln_" in name:
            param.requires_grad = True
        if "lm_head" in name:
            param.requires_grad = True
        if "transformer.wte" in name:
            param.requires_grad = True
        if "transformer.wpe" in name:
            param.requires_grad = True

    model.print_trainable_parameters()

model = model.to(device)

n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
p_trainable_params = n_trainable_params / n_total_params

trainable_params = (p for p in model.parameters() if p.requires_grad)
trainable_params_names = [name for name, p in model.named_parameters() if p.requires_grad]

optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1_000, num_training_steps=NUM_TRAINING_STEPS)

_config = {
    "using_peft": USE_PEFT,
    "layer_norm_trainable": TRAIN_LN,
    "peft_config": peft_config.to_dict(),
    "total_params": n_total_params,
    "trainable_params": n_trainable_params,
    "percent_trainable_params": p_trainable_params,
    "name_trainable_params": trainable_params_names,
    "dataset": "c4",
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "model": model_config.to_dict(),
    "scheduler": "linear",
    "device": str(device),
}

wandb.init(project="peft_pretraining", config=_config)
pbar = tqdm(total=NUM_TRAINING_STEPS)



if __name__ == "__main__":
    # launch training
    training_function()
