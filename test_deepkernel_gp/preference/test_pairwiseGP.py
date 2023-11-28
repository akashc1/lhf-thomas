from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from my_utils.gp import VariationalPreferentialGP as VPGP
from gpytorch.mlls import VariationalELBO
import torch
from tqdm import tqdm
from botorch.models.pairwise_gp import (
    PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
)
from botorch.models.transforms import Normalize



def compute_embedded_data(split, dataset_path, embedder, preprocess_batch_size):

    dset = load_dataset(dataset_path, split=split)
    num_samples = min(len(dset), len(dset))
    queries = []
    chosens = []
    rejecteds = []

    for batch_start in tqdm(range(0, num_samples, preprocess_batch_size)):
        batch_end = min(batch_start + preprocess_batch_size, num_samples)
        bs = dset[batch_start:batch_end]
        c = [p + s for p, s in zip(bs["prompt"], bs["chosen"])]
        r = [p + s for p, s in zip(bs["prompt"], bs["rejected"])]

        chosen_embeddings = embedder.encode(c, convert_to_tensor=True)
        # >>> preprocess_batch_size x embedding_size
        
        rejected_embeddings = embedder.encode(r, convert_to_tensor=True)
        # >>> preprocess_batch_size x embedding_size

        chosens.append(chosen_embeddings)
        rejecteds.append(rejected_embeddings)

    chosens = torch.cat(chosens, dim=0)
    rejecteds = torch.cat(rejecteds, dim=0)
    queries = torch.stack([chosens, rejecteds], dim=1)
    embedding_size = chosens.shape[-1]
    queries = queries.reshape(-1, embedding_size)

    prefers = torch.arange(queries.shape[0])
    prefers = prefers.reshape(-1, 2)
    # prefer = [4, 7] meaning prefer element with index 4 over element 
    # with index 7 in the queries tensor

    return queries, prefers

# setting seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_path = "Dahoas/rm-static"
similarity_model = "all-mpnet-base-v1"
embedder = SentenceTransformer(similarity_model)
bounds = [-1, 1]
num_epochs = 1000
batch_size = 6
preprocess_batch_size = 1000

train_queries, train_prefers = compute_embedded_data(
    split="test",
    dataset_path=dataset_path, 
    embedder=embedder,
    preprocess_batch_size=preprocess_batch_size,
)
test_queries, test_prefers = compute_embedded_data(
    split="test",
    dataset_path=dataset_path,
    embedder=embedder,
    preprocess_batch_size=preprocess_batch_size,
)

breakpoint()

reward_model = PairwiseGP(
    train_queries.cuda(),
    train_prefers.cuda(),
    input_transform=Normalize(d=train_prefers.shape[-1])
)
mll = PairwiseLaplaceMarginalLogLikelihood(
    reward_model.likelihood, reward_model
)
reward_model.train()

optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.01)
for epoch in range(num_epochs):
    # minibatch training
    idx_train = torch.randperm(train_queries.shape[0])[:batch_size]
    output = reward_model(train_queries[idx_train].cuda())
    loss = -mll(output, train_prefers[idx_train].cuda()).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch} - Loss: {loss.item():>4.3f}")

with torch.no_grad():
    reward_model.eval()

    idx_test = torch.randperm(test_queries.shape[0])[:batch_size]
    data = dict(
        train=train_queries[idx_train],
        test=test_queries[idx_test],
    )
    for key, queries in data.items():
        preference_score = reward_model.posterior(queries.cuda())
        preference_score = preference_score.mean.squeeze(-1)
        predicted_label = preference_score.argmax(dim=-1)
        accuracy = (predicted_label == 1).float()
        print(f"{key} accuracy:", accuracy.sum().item() / batch_size)

    reward_model.train()
