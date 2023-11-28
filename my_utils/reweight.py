import torch
import torch.nn as nn
from torch.nn import functional as F
from tensordict import TensorDict
from tqdm import tqdm
import wandb
import numpy as np
from transformers import OPTForCausalLM, LlamaForCausalLM, GPT2LMHeadModel
from transformers import PreTrainedModel, PretrainedConfig
from torch.utils.data import DataLoader
from preference_dataset import TOKENIZER_LEFT, featurize_sentence, SYNFUNCGP
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import bundles

plt.rcParams.update(bundles.iclr2023())


class Reweighted_GPT2LMHeadModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(PretrainedConfig())
        self.config = config
        self.P = GPT2LMHeadModel.from_pretrained("gpt2")
        # self.P = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        # self.P = LlamaForCausalLM.from_pretrained("../llama_hf_7B")

        self.W = GPT2LMHeadModel.from_pretrained("gpt2")
        # self.W = OPTForCausalLM.from_pretrained("facebook/opt-350m")

        # adapt vocab size of W for P
        self.adaptor = nn.Linear(self.W.config.vocab_size, self.P.config.vocab_size)

        self.reward = nn.Linear(self.P.config.vocab_size, 1)

        for p in self.P.parameters():
            p.requires_grad = False

        model_parameters = filter(lambda p: p.requires_grad, self.W.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"N trainable params W: {params}")

    def parameters(self):
        return (
            list(self.W.parameters())
            + list(self.adaptor.parameters())
            + list(self.reward.parameters())
        )

    def forward(self, idx, attn_mask):
        # print(torch.cuda.max_memory_allocated(device="cuda:0"))
        P = self.P(input_ids=idx, attention_mask=attn_mask)
        P = P.logits.detach()

        W = self.W(input_ids=idx, attention_mask=attn_mask)
        W = self.adaptor(W.logits)

        prob_P = F.softmax(P, dim=-1)
        prob_W = F.softmax(W, dim=-1)

        prob = prob_P * prob_W
        prob = prob / prob.sum(-1, keepdim=True)

        return prob

    def loss_policy(self, idx):
        loss = 0
        for i in range(self.config["max_new_tokens"]):
            probs = self(idx=idx, attn_mask=None)[:, [-1], :]
            dist = torch.distributions.Categorical(probs=probs)
            idx_next = dist.sample()

            if self.config["reward_feature"] == "log_prob":
                reward = dist.log_prob(idx_next)[..., 0].detach()
            else:
                reward = self.reward(probs)[..., 0, 0].detach()

            loss = (
                loss
                - dist.log_prob(idx_next) * reward
                - self.config["entropy_bonus_coeff"] * dist.entropy()
            )
            idx = torch.cat((idx, idx_next), dim=-1)

        return loss.mean()

    # def generate(self, idx):
    #     for _ in range(self.config["max_new_tokens"]):
    #         probs = self(idx=idx, attn_mask=None)[:, [-1], :]
    #         dist = torch.distributions.Categorical(probs=probs)
    #         idx_next = dist.sample()
    #         idx = torch.cat((idx, idx_next), dim=-1)

    #     return idx


class Reweighter_Trainer:
    def __init__(self, config, train_data, val_data, collate_fn):
        self.config = config
        self.train_dataloader = DataLoader(
            train_data,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )
        self.val_dataloader = DataLoader(
            val_data,
            batch_size=self.config["batch_size_val"],
            shuffle=True,
            collate_fn=collate_fn,
        )

        self.log_softmax = nn.LogSoftmax(dim=0)
        self.model = Reweighted_GPT2LMHeadModel(config=self.config).to(
            self.config["device"]
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )

    def get_batch(self, split):
        dataloader = self.train_dataloader if split == "train" else self.val_dataloader
        batch = next(iter(dataloader))
        batch_size = (
            self.config["batch_size"]
            if split == "train"
            else self.config["batch_size_val"]
        )
        return TensorDict(batch, batch_size=[batch_size], device=self.config["device"])

    def train(self):
        if self.config["wandb_log"]:
            wandb.init(
                project=self.config["wandb_project"],
                name=self.config["wandb_run_name"],
                config=self.config,
            )

        for iter in tqdm(range(self.config["max_iters"])):
            batch = self.get_batch("train")

            reward_loss = self.loss_reward(batch)
            reward_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            batch_prompt = batch["prompt_ids"][:1]
            policy_loss = self.model.loss_policy(batch_prompt)
            policy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss = reward_loss + self.config["policy_weight"] * policy_loss

            if self.config["wandb_log"]:
                wandb.log({"train/loss": train_loss.item()})

            if (iter + 0) % self.config["eval_interval"] == 0:
                with torch.no_grad():
                    feature_reward = []
                    for _ in range(16):
                        batch = self.get_batch("val")
                        reward_loss = self.loss_reward(batch)
                        policy_loss = self.model.loss_policy(batch["prompt_ids"][:2])
                        val_loss = (
                            reward_loss + self.config["policy_weight"] * policy_loss
                        )

                        tldr = torch.tensor(TOKENIZER_LEFT("TL;DR:")["input_ids"]).to(
                            self.config["device"]
                        )
                        tldr = tldr[None].repeat(batch["prompt_ids"].shape[0], 1)
                        idx = torch.cat([batch["prompt_ids"], tldr], dim=-1)

                        text = self.model.generate(idx.clone())
                        for i in range(text.shape[0]):
                            generative_answer = text[i][len(idx[i]) :]
                            generative_answer = TOKENIZER_LEFT.decode(
                                generative_answer.tolist()
                            )
                            feature = featurize_sentence(generative_answer)
                            reward = SYNFUNCGP(feature)
                            feature_reward.append([feature, reward])

                if self.config["wandb_log"]:
                    wandb.log({"val/loss": val_loss.item()})

                feature_reward = np.array(feature_reward)
                ground_truth = SYNFUNCGP(np.linspace(0, 1, 1000).reshape(-1, 1))

                plt.figure()
                plt.plot(
                    np.linspace(0, 1, 1000),
                    ground_truth,
                    label="ground truth",
                    color="black",
                    alpha=0.5,
                )
                plt.scatter(
                    feature_reward[:, 0], feature_reward[:, 1], s=1, label="generated"
                )
                sns.kdeplot(
                    feature_reward[:, 0], fill=False, color="blue", label="generated"
                )
                plt.legend()
                plt.xlabel("Feature")
                plt.ylabel("Reward")
                plt.savefig(
                    f"results/{self.config['wandb_run_name']}/feature_reward_{iter}.png"
                )
                plt.savefig(
                    f"results/{self.config['wandb_run_name']}/feature_reward_{iter}.pdf"
                )

                print_text = False
                if print_text:
                    print(
                        f"Iter: {iter}, Train loss: {train_loss}, Val loss: {val_loss} \n \n"
                        f"Generated: {TOKENIZER_LEFT.decode(text[0].tolist())} \n \n"
                        f"Chosen: {TOKENIZER_LEFT.decode(batch['chosen_ids'][0].tolist())} \n \n"
                        f"Rejected: {TOKENIZER_LEFT.decode(batch['rejected_ids'][0].tolist())} \n \n"
                    )

    def loss_reward(self, batch):
        rewards = []
        for i in ["chosen", "rejected"]:
            pai = torch.cat([batch["prompt_ids"], batch[f"{i}_ids"]], dim=-1)
            pai_attn_mask = torch.cat(
                [batch["prompt_attn_masks"], batch[f"{i}_attn_masks"]], dim=-1
            )
            pai_input = pai[..., :-1]
            pai_target = pai[..., 1:]
            pai_attn_mask_input = pai_attn_mask[..., :-1]
            pai_attn_mask_target = pai_attn_mask[..., 1:]

            probs = self.model(idx=pai_input, attn_mask=pai_attn_mask_input)

            if self.config["reward_feature"] == "log_prob":
                dist = torch.distributions.Categorical(probs=probs)
                log_prob = dist.log_prob(pai_target)
                reward = (pai_attn_mask_target * log_prob).sum(-1)
            else:
                reward = self.model.reward(probs)

            rewards.append(reward)

        rewards = torch.stack(rewards, dim=-1)
        loss = -self.log_softmax(rewards)[0].mean()

        return loss
