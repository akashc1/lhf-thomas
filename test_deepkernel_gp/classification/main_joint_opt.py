#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Test performance of Pretrained GP on classification."""

from tqdm import tqdm
import torch
from torch import Tensor
from datasets import load_dataset
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.mlls import VariationalELBO
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import numpy as np
import sys
import re
import string
import nltk
from nltk.corpus import stopwords
from my_utils.rembedder import RSentenceTransformer
from my_utils.gp import VariationalDirichletGPModel
from my_utils.calibration import compute_calibration

def removing(text):
    s_words = stopwords.words("english")
    a = " ".join(i for i in text.split(" ") if i not in s_words)
    return a


def stemming(text):
    snow_stemmer = nltk.SnowballStemmer("english")
    text = " ".join(snow_stemmer.stem(i) for i in text.split(" "))
    return text


def cleaning(a):
    a = str(a).lower()
    a = re.sub("\[.*?\]", "", a)
    a = re.sub("[%s]" % re.escape(string.punctuation), "", a)
    a = re.sub("\n", "", a)
    a = re.sub("https?://\S+|www\.\S+", "", a)
    a = re.sub("<.*?>+", "", a)
    a = re.sub("\w*\d\w*", "", a)

    a = removing(a)
    a = stemming(a)

    return a


def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def compute_embedded_data(
    split,
    dataset_path,
    embedder,
    preprocess_batch_size,
    text_key,
    label_key,
    label_mapping,
):
    dset = load_dataset(*dataset_path, split=split)
    num_samples = min(20000, len(dset))

    dset_idx = torch.randperm(len(dset))[:num_samples]
    dset = dset.select(dset_idx)

    queries = []
    labels = []

    for batch_start in tqdm(range(0, num_samples, preprocess_batch_size)):
        batch_end = min(batch_start + preprocess_batch_size, num_samples)
        bs = dset[batch_start:batch_end]
        if dataset == "twitter":
            text = [cleaning(x) for x in bs[text_key]]
            label = [label_mapping[l] for l in bs[label_key]]
            remove_idx = []
            for i, l in enumerate(label):
                if l == -1:
                    remove_idx.append(i)
            for i in sorted(remove_idx, reverse=True):
                del text[i]
                del label[i]
            labels = labels + label
        else:
            text = bs[text_key]
            labels = labels + bs[label_key]

        # Method 1
        # queries_embeddings = embedder.encode(text, convert_to_tensor=True)

        # Method 2
        # features = embedder.tokenize(text)
        # features = batch_to_device(features, embedder.device)
        # features = embedder.forward(features) # Pass through the Transformer model
        # queries_embeddings = features['sentence_embedding']
        # >>> preprocess_batch_size x embedding_size

        # Method 3 #######################################
        features = embedder.embedder[0].tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        features = batch_to_device(features, embedder.embedder.device)
        # >>> preprocess_batch_size x max_seq_length

        # Convert token ids to one-hot vectors
        features_onehot = torch.nn.functional.one_hot(
            features["input_ids"], num_classes=embedder.embedder[0].tokenizer.vocab_size
        ).float()
        # >>> preprocess_batch_size x max_seq_length x vocab_size

        queries_embeddings = embedder.encode(features_onehot)
        queries.append(queries_embeddings.cpu())

    queries = torch.cat(queries, dim=0)
    labels = torch.tensor(labels).reshape(-1)

    queries = queries.to(device)
    # >>> n_data x embedding_size
    
    labels = labels.to(device)
    # >>> n_data

    for lb in labels.unique():
        print(f"Number of class {lb}:", torch.sum(labels == lb))
    print("Number of samples:", queries.shape[0])

    return queries, labels


def set_dataset_args(dataset):
    if dataset == "imdb":
        dataset_path = ["jahjinx/IMDb_movie_reviews"]
        text_key = "text"
        label_key = "label"
        split_train = "train"
        split_test = "test"
        label_mapping = None
    elif dataset == "twitter":
        dataset_path = ["sentiment140"]
        text_key = "text"
        label_key = "sentiment"
        split_train = "train"
        split_test = "test"
        label_mapping = {0: 0, 2: -1, 4: 1}
    elif dataset == "poem":
        dataset_path = ["poem_sentiment"]
        text_key = "verse_text"
        label_key = "label"
        split_train = "train"
        split_test = "test"
        label_mapping = None
    elif dataset == "twitter2":
        dataset_path = ["zeroshot/twitter-financial-news-sentiment"]
        text_key = "text"
        label_key = "label"
        split_train = "train"
        split_test = "validation"
        label_mapping = None
    elif dataset == "tweet_eval":
        dataset_path = ["tweet_eval", "emotion"]
        text_key = "text"
        label_key = "label"
        split_train = "train"
        split_test = "test"
        label_mapping = None
    else:
        raise NotImplementedError
    return (
        dataset_path,
        text_key,
        label_key,
        split_train,
        split_test,
        label_mapping,
    )

# Basic setup
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
dataset = sys.argv[1]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
similarity_model = "all-mpnet-base-v1"
embedder = RSentenceTransformer(similarity_model).to(device)
bounds = [-1, 1]
batch_size = 200
preprocess_batch_size = 100
dataset_args = set_dataset_args(dataset)
dataset_path = dataset_args[0]
text_key = dataset_args[1]
label_key = dataset_args[2]
split_train = dataset_args[3]
split_test = dataset_args[4]
label_mapping = dataset_args[5]
num_samples_ppd = 256
num_bins = 20
num_epochs = 50
nll_coeff = 0.1

# Prepare data
data = []
for split in [split_train, split_test]:
    data_ = compute_embedded_data(
        split=split,
        dataset_path=dataset_path,
        embedder=embedder,
        preprocess_batch_size=preprocess_batch_size,
        text_key=text_key,
        label_key=label_key,
        label_mapping=label_mapping,
    )
    data.append(data_)
train_queries, train_labels = data[0]
test_queries, test_labels = data[1]

# Fitting GP model
print("Fitting GP model...")
likelihood = DirichletClassificationLikelihood(
    train_labels, learn_additional_noise=True
)
num_classes = likelihood.num_classes
model = VariationalDirichletGPModel(
    queries=train_queries,
    responses=likelihood.transformed_targets,
    bounds=bounds,
    num_classes=likelihood.num_classes,
    likelihood=likelihood,
)
model.train()
likelihood.train()
mll = VariationalELBO(
    likelihood=likelihood,
    model=model,
    num_data=model.num_data,
)
mll = mll.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for i in range(num_epochs):
    for mb_i in range((train_queries.shape[0]-1) // batch_size+1):
        # Get the mb corresponding to mb_i
        mb_start = mb_i * batch_size
        mb_end = min((mb_i+1)*batch_size, train_queries.shape[0])
        
        mb_X = train_queries[mb_start:mb_end]
        # >>> mb_size x embedding_size

        mb_Y = likelihood.transformed_targets[:, mb_start:mb_end]
        # >>> num_classes x mb_size

        dist = model(mb_X)

        mll_loss = -mll(dist, mb_Y).sum()

        # Note that below sampling step defines a mixture of 
        # Categorical distributions, which is equivalent to
        # a single Categorical distribution, whose parameters
        # are the weighted average of the mixture components'
        # probability vector.
        # https://stats.stackexchange.com/questions/454533/
        # em-algorithm-for-mixture-of-categorical-
        # distributions-instantly-stabilizes
        mixture_logits = dist.sample(
            sample_shape=torch.Size([num_samples_ppd])
        )
        # >>> num_samples_ppd x num_classes x batch_size x 1

        mixture_logits = mixture_logits.squeeze(-1)
        # >>> num_samples_ppd x num_classes x batch_size

        mixture_logits = mixture_logits.permute(2, 0, 1)
        rescaled_mixture_logits = model.platt_scale(mixture_logits)
        # >>> batch_size x num_samples_ppd x num_classes

        rescaled_mixture_probs = rescaled_mixture_logits.softmax(-1)
        # >>> batch_size x num_samples_ppd x num_classes

        rescaled_probs = rescaled_mixture_probs.mean(dim=1)
        # >>> batch_size x num_classes

        ppd = Categorical(probs=rescaled_probs)

        mb_Y = train_labels[mb_start:mb_end]
        # >>> n_data x num_classes

        nll_loss = -ppd.log_prob(mb_Y).sum()
        loss = mll_loss + nll_coeff * nll_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch: {i}, MLL Loss: {mll_loss.item()}", end="\r", flush=True)
        print(f"Epoch: {i}, NLL Loss: {nll_loss.item()}", end="\r", flush=True)


# Test GP model
model.eval()
likelihood.eval()
pred_ys = []
pred_ys_tc = []
pred_conds = []
pred_conds_tc = []
real_ys = []
for mb_i in range((test_queries.shape[0]-1) // batch_size+1):
    # Get the mb corresponding to mb_i
    mb_start = mb_i * batch_size
    mb_end = min((mb_i + 1) * batch_size, test_queries.shape[0])
    mb_test_X = test_queries[mb_start:mb_end]
    mb_test_Y = test_labels[mb_start:mb_end]

    with torch.no_grad():        
        posterior = model.posterior(mb_test_X)
        mixture_logits = posterior.sample(
            sample_shape=torch.Size([num_samples_ppd])
        )
        # >>> num_samples_ppd x num_classes x batch_size x 1

        mixture_logits = mixture_logits.squeeze(-1)
        # >>> num_samples_ppd x num_classes x batch_size

        mixture_logits = mixture_logits.permute(2, 0, 1)
        rescaled_mixture_logits = model.platt_scale(mixture_logits)
        # >>> batch_size x num_samples_ppd x num_classes

        mixture_probs = mixture_logits.softmax(-1)
        # >>> batch_size x num_samples_ppd x num_classes
        probs = mixture_probs.mean(dim=1)
        # >>> batch_size x num_classes
        
        rescaled_mixture_probs = rescaled_mixture_logits.softmax(-1)
        # >>> batch_size x num_samples_ppd x num_classes
        rescaled_probs = rescaled_mixture_probs.mean(dim=1)
        # >>> batch_size x num_classes

    pred_y = probs.argmax(-1)
    pred_y_tc = rescaled_probs.argmax(-1)
    # >>> batch_size

    pred_ys.append(pred_y)
    pred_ys_tc.append(pred_y_tc)
    pred_conds.append(probs)
    pred_conds_tc.append(rescaled_probs)
    real_ys.append(mb_test_Y)

pred_ys = torch.cat(pred_ys, dim=0)
pred_ys_tc = torch.cat(pred_ys_tc, dim=0)
real_ys = torch.cat(real_ys, dim=0)
# >>> n_data

pred_conds = torch.cat(pred_conds, dim=0)
pred_conds_tc = torch.cat(pred_conds_tc, dim=0)
# >>> n_data x num_classes

# Calculate accuracy
acc = torch.sum(pred_ys == real_ys) / pred_ys.shape[0]
acc_tc = torch.sum(pred_ys_tc == real_ys) / pred_ys_tc.shape[0]
print("Test Accuracy: %.3f" % (acc))
print("Test Accuracy w Re-calibration: %.3f" % (acc_tc))

# Calculate calibration
cal_outputs = []
for p_mean in [pred_conds, pred_conds_tc]:
    result = compute_calibration(
        y=real_ys,
        p_mean=p_mean,
        num_bins=num_bins,
        num_classes=num_classes,
    )
    cal_outputs.append(result)
cals, cals_tc = cal_outputs

for c in range(num_classes):
    # For each class
    print("===== Class %d =====" % (c))
    mean_conf, acc_tab = cals[c]["reliability_diag"]
    mean_conf_tc, acc_tab_tc = cals_tc[c]["reliability_diag"]
    data = [
        dict(
            data=mean_conf,
            acc_tab=acc_tab,
            label="No calibration"
        ),
        dict(
            data=mean_conf_tc,
            acc_tab=acc_tab_tc,
            label="Platt's scaling"
        ),
    ]
    print("Plotting calibration curve...")
    plt.figure()
    for i in data:
        plt.scatter(
            i["data"], i["acc_tab"], label=i["label"],
        )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"calibration_curve_{dataset}_{c}.pdf")
    plt.close()
