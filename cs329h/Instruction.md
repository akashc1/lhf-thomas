# Fine-tuning language models
There are two common pipelines for fine-tuning language models: 
1. Direct Preference Optimization (DPO)
2. Proximal Policy Optimization (PPO)
We have examples of both pipelines in **dpoppo.ipynb**.
Model: GPT2
Dataset: Anthropic/hh-rlhf
You can find two functions **run_dpo()** and **run_ppo()** in that notebook.

# Evaluating fine-tuned models
To evaluate fine-tuned models on **hh-rlhf** dataset, you can use below datasets:
- OpenBookQA
- HellaSwag
- Lambada
- ARC-Challenge
- ARC-Easy
- TriviaQA

There is a standard pipeline for evaluating these datasets. 
You can refer [Repo](https://github.com/EleutherAI/lm-evaluation-harness).
We have included above repo in this project. To run evaluation, you first need to install repo by
```
cd lm-evaluation-harness-0.3.0
pip install -e .
```
To run evaluation, please follow below commands
```
cd lm-evaluation-harness-0.3.0
python main.py \
    --model [model_id] \
    --tasks [dataset_name] \
    --output_path [output_file].json
    --device [gpu_id]
```
For example, if you want to evaluate **GPT2** on **HellaSwag** dataset and save results in **output.json**, please run
```
cd lm-evaluation-harness-0.3.0
python main.py \
    --model gpt2 \
    --tasks hellaswag \
    --output_path output.json
    --device 0
```
You can find the task name for dataset in this [table](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md)

Example "output.json"
```
{
  "results": {
    "hellaswag": {
      "acc": 0.2891854212308305,
      "acc_stderr": 0.00452457589295296,
      "acc_norm": 0.31139215295757816,
      "acc_norm_stderr": 0.004621163476949214
    }
  },
  "versions": {
    "hellaswag": 0
  },
  "config": {
    "model": "gpt2",
    "model_args": "",
    "num_fewshot": 0,
    "batch_size": null,
    "device": "0",
    "no_cache": false,
    "limit": null,
    "bootstrap_iters": 100000,
    "description_dict": {}
  }
}
```

# Access to computer
ssh test@4.tcp.us-cal-1.ngrok.io -p 17552
password: drugdiscovery123
