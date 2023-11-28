import torch
import random
import numpy as np
import subprocess
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelWithLMHead,
    AutoModelForSeq2SeqLM
)
import math
import deepspeed


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)

def launch_cmd(cmd):
    print(f"Running:\n{cmd}")
    p = subprocess.Popen(cmd, cwd=os.getcwd(), shell=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Failed to run:\n{cmd}")
    
def load_actor(actor_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        actor_name_or_path, fast_tokenizer=True, padding_side='left'
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    actor_config = AutoConfig.from_pretrained(actor_name_or_path)
    actor = AutoModelForCausalLM.from_pretrained(
        actor_name_or_path,
        from_tf=bool(".ckpt" in actor_name_or_path),
        config=actor_config,
    ).cuda()
    actor.config.end_token_id = tokenizer.eos_token_id
    actor.config.pad_token_id = actor.config.eos_token_id
    actor.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
    
    # actor = deepspeed.init_inference(
    #     model=actor,
    #     dtype=torch.half,  # dtype of the weights (fp16)
    #     replace_with_kernel_inject=True,  # replace the model with the kernel injector
    #     max_out_tokens=actor.config.max_position_embeddings,  # max number of tokens to generate
    # )
    return actor, tokenizer

def load_actor_with_logit(actor_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        actor_name_or_path, fast_tokenizer=True, padding_side='left'
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    # actor = AutoModelWithLMHead.from_pretrained(
    actor = AutoModelForCausalLM.from_pretrained(
    # actor = AutoModelForSeq2SeqLM.from_pretrained(
        actor_name_or_path,
    ).cuda()
    actor.config.end_token_id = tokenizer.eos_token_id
    actor.config.pad_token_id = actor.config.eos_token_id
    
    return actor, tokenizer