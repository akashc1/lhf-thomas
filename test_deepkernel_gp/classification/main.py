#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Test performance of Pretrained GP on classification."""

import torch
from tqdm import tqdm
from torch import Tensor
from datasets import load_dataset
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.mlls import VariationalELBO
from my_utils.rembedder import RSentenceTransformer
from my_utils.gp import VariationalDirichletGPModel
from my_utils.calibration import compute_calibration
from botorch.fit import fit_gpytorch_mll
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import numpy as np
import sys
import re
import string
import nltk
from nltk.corpus import stopwords

dataset = sys.argv[1]
if dataset == "imdb":
    dataset_path = ["jahjinx/IMDb_movie_reviews"]
    text_key = "text"
    label_key = "label"
    split_train = "train"
    split_test = "test"

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

elif dataset == "twitter2":
    dataset_path = ["zeroshot/twitter-financial-news-sentiment"]
    text_key = "text"
    label_key = "label"
    split_train = "train"
    split_test = "validation"

elif dataset == "tweet_eval":
    dataset_path = ["tweet_eval", "emotion"]
    text_key = "text"
    label_key = "label"
    split_train = "train"
    split_test = "test"

s_words = stopwords.words("english")
snow_stemmer = nltk.SnowballStemmer("english")


def removing(text):
    a = " ".join(i for i in text.split(" ") if i not in s_words)
    return a


def stemming(text):
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
    text_key="text",
    label_key="label",
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


# setting seed
seed = 0
device = "cuda:0" if torch.cuda.is_available() else "cpu"
similarity_model = "all-mpnet-base-v1"
embedder = RSentenceTransformer(similarity_model).to(device)
bounds = [-1, 1]
num_epochs = 20
batch_size = 200
preprocess_batch_size = 100

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


data = []
for split in [split_train, split_test]:
    data_ = compute_embedded_data(
        split=split,
        dataset_path=dataset_path,
        embedder=embedder,
        preprocess_batch_size=preprocess_batch_size,
        text_key=text_key,
        label_key=label_key,
    )
    data.append(data_)
train_queries, train_labels = data[0]
test_queries, test_labels = data[1]

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

# Fit hyperparameters
mll = fit_gpytorch_mll(mll)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# for i in range(num_epochs):
#     for minibatch_i in range(
#         (train_queries.shape[0] - 1) // batch_size + 1
#     ):
#         # Get the minibatch corresponding to minibatch_i
#         minibatch_start = minibatch_i * batch_size
#         minibatch_end = min(
#             (minibatch_i + 1) * batch_size, train_queries.shape[0]
#         )
#         minibatch_test_X = train_queries[minibatch_start:minibatch_end]
#         minibatch_test_Y = likelihood.transformed_targets[:, minibatch_start:minibatch_end]
#         output = model(minibatch_test_X)
#         loss = -mll(
#             output,
#             minibatch_test_Y
#         ).sum()
#         print("Epoch: %03d, Loss: %.3f" % (i, loss.item()), end="\r", flush=True)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

# Tuning temperature
print("Tuning temperature...")

# Infer posterior
with torch.no_grad():
    posterior = model.posterior(train_queries)
    sampled_posteriors = posterior.sample(sample_shape=torch.Size([256]))
    # >>> 256 x n_data x num_classes x 1

    logits = sampled_posteriors.mean(dim=0).squeeze(-1)
    logits = logits.permute(1, 0)
    # >>> n_data x num_classes

# Calculate NLL before temperature scaling
before_temperature_nll = (
    -torch.distributions.categorical.Categorical(logits=logits)
    .log_prob(train_labels)
    .mean()
    .item()
)
print("Before temperature - NLL: %.3f" % (before_temperature_nll,))

# Optimize the temperature w.r.t. NLL
optimizer = torch.optim.LBFGS(
    [model.temperature], lr=0.01, max_iter=2000, line_search_fn="strong_wolfe"
)

def eval():
    optimizer.zero_grad()
    loss = (
        -Categorical(
            logits=model.temperature_scale(logits)
        )
        .log_prob(train_labels)
        .mean()
    )

    print("NLL:", loss.item(), end="\r", flush=True)
    loss.backward()
    return loss


optimizer.step(eval)

# Calculate NLL after temperature scaling
after_temperature_nll = (
    -Categorical(logits=model.temperature_scale(logits))
    .log_prob(train_labels)
    .mean()
    .item()
)
print("Optimal temp:", model.temperature.cpu().detach().numpy().tolist())
print("After temp - NLL: %.3f" % (after_temperature_nll,))

# Test model accuracy with test data and minibatches
model.eval()
likelihood.eval()
num_correct = 0
num_total = 0

# Test calibration of GP model
# Initialize a bining array for storing the predicted probabilities
print("Testing GP model...")
binning_size = 0.1
binning_array = np.arange(0, 1 + binning_size, binning_size)
bining_count = np.zeros((binning_array.shape[0] - 1,))
num_correct_in_bin = np.zeros((binning_array.shape[0] - 1,))

pred_ys = []
pred_ys_tc = []
pred_conds = []
pred_conds_tc = []
real_ys = []
for minibatch_i in range((test_queries.shape[0] - 1) // batch_size + 1):
    # Get the minibatch corresponding to minibatch_i
    minibatch_start = minibatch_i * batch_size
    minibatch_end = min((minibatch_i + 1) * batch_size, test_queries.shape[0])
    minibatch_test_X = test_queries[minibatch_start:minibatch_end]
    minibatch_test_Y = test_labels[minibatch_start:minibatch_end]

    # Get prediction
    with torch.no_grad():
        posterior = model.posterior(minibatch_test_X)
        sampled_posteriors = posterior.sample(sample_shape=torch.Size([256]))
        logits = sampled_posteriors.mean(dim=0).squeeze(-1)
        p_y_cond_x = logits.softmax(dim=0)
        p_y_cond_x_tc = (model.temperature_scale(logits.T).T).softmax(dim=0)
    # >>> num_classes x minibatch_size

    pred_y = torch.argmax(p_y_cond_x, dim=0)  # predicted class
    pred_y_tc = torch.argmax(p_y_cond_x_tc, dim=0)  # predicted class
    # >>> minibatch_size

    pred_ys.append(pred_y)
    pred_ys_tc.append(pred_y_tc)
    pred_conds.append(p_y_cond_x.permute(1, 0))
    pred_conds_tc.append(p_y_cond_x_tc.permute(1, 0))
    real_ys.append(minibatch_test_Y)

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
print("Calculating calibration...")
cals = compute_calibration(
    y=real_ys, p_mean=pred_conds, num_bins=20, num_classes=num_classes
)
cals_tc = compute_calibration(
    y=real_ys, p_mean=pred_conds_tc, num_bins=20, num_classes=num_classes
)

for c in cals:
    # For each class
    print("===== Class %d =====" % (c))
    cal = cals[c]
    cal_tc = cals_tc[c]
    # print("ECE: %.3f" % (cal['ece']))
    # print("MCE: %.3f" % (cal['mce']))
    mean_conf, acc_tab = cal["reliability_diag"]
    mean_conf_tc, acc_tab_tc = cal_tc["reliability_diag"]

    print("Plotting calibration curve...")
    plt.figure()
    plt.plot(
        mean_conf,
        acc_tab,
        marker="o",
        linestyle="None",
        label="No calibration",
    )
    plt.plot(
        mean_conf_tc,
        acc_tab_tc,
        marker="o",
        linestyle="None",
        label="Platt's scaling",
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"calibration_curve_{dataset}_{c}.pdf")
    plt.close()
