from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class EmbedderBase(ABC):
    @abstractmethod
    def embed(self, sentence):
        pass

    def __call__(self, sentence, g2e_transl):
        r""""Adapt generator vocab to embedder vocab and then embed.
        Args:
            sentence: One-hot tensor of shape 
                (batch_size, max_length, vocab_size_generator)

            g2e_transl: A function that takes a one-hot tensor of shape
                (batch_size, max_length, vocab_size_generator) and returns
                a (batch_size, max_length, vocab_size_embedder) tensor.

        Returns: A tensor of shape (batch_size, embedding_size)
        """
        return self.embed(g2e_transl(sentence))

    def to(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        return self


class RSentenceTransformer(EmbedderBase):
    def __init__(self, embedder_name):
        self.embedder = SentenceTransformer(embedder_name)

        # Converting `Embedding` layers to `Linear` layers
        # to maintain differentiability with respect to input
        tfm = self.embedder[0]._modules['auto_model']
        tfm_word_embedding = tfm.embeddings.word_embeddings
        self.vocab_size_embedder, self.embedding_size = \
            tfm_word_embedding.weight.shape

        new_we = nn.Linear(
            self.vocab_size_embedder, self.embedding_size,
            bias=False, 
            device=self.embedder.device
        )

        new_we.weight = nn.parameter.Parameter(
            tfm_word_embedding.weight.T
        )

        self.embedder[0]._modules['auto_model']\
            .embeddings.word_embeddings = new_we

        # Freeze the model
        self.embedder.eval()
        for param in self.embedder.parameters():
            param.requires_grad = False

    def embed(self, sentence):
        if self.embedder.device != self.device:
            self.embedder.to(self.device)

        sentence_shape = sentence.shape
        sentence = sentence.reshape(-1, *sentence_shape[-2:])
        # >>> (batch_size*n_restart*q) 
        # ... x max_length x vocab_size_generator

        # Embedding one-hot vector
        inputs_embeds = self.embedder[0]._modules['auto_model']\
            .embeddings.word_embeddings(sentence)
        # >>> (batch_size*n_restart*q)
        # ... x max_length x embedding_size

        input_shape = inputs_embeds.size()[:-1]

        features = {}
        features['attention_mask'] = torch.ones(
            input_shape, device=self.device, dtype=self.dtype
        )
        features['token_embeddings'] = self.embedder[0]\
            ._modules['auto_model'].forward(
                inputs_embeds=inputs_embeds
            )['last_hidden_state']
        # >>> (batch_size*n_restart*q)
        # ... x max_seq_length x embedding_size

        for i in range(1, len(self.embedder)):
            self.embedder[i](features)
            # >>> (batch_size*n_restart*q) x embedding_size

        sentence_embedding = features['sentence_embedding']
        return sentence_embedding.reshape(
            *sentence_shape[:-2], -1
        )
        # >>> batch_size x n_restart x q x embedding_size


class IdentityEmbbeder(EmbedderBase):
    def embed(self, sentence):
        return sentence
