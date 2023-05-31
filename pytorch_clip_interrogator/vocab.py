"""
Copyright 2023 by Sergei Belousov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
from operator import itemgetter
from typing import Dict, List, Optional
# FAISS index
import faiss
# numpy
import numpy as np
# torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# CLIP Model
from transformers import CLIPProcessor, CLIPModel
# Text dataset
import datasets
from datasets import load_dataset
# utils
from tqdm.auto import tqdm
from .utils import *


class Vocab:
    """ Prompts Corpus with their CLIP embeddings.

    Args:
        dataset (datasets.arrow_dataset.Dataset): input text dataset.
        index (faiss.IndexFlatIP): FAISS index with CLIP embeddings.
    """
    def __init__(self, dataset: datasets.arrow_dataset.Dataset, index: faiss.IndexFlatIP):
        self.dataset = dataset
        self.index = index

    def __call__(self, embedding: np.ndarray, k: int = 1) -> List[str]:
        """ Search nearest prompt by index.

        Args:
            embedding (np.ndarray): normalized CLIP embedding.
            k (int): return top-k nearest prompts.
        Returns:
            list of nearest prompts.
        """
        scores, ids = self.index.search(embedding, k)
        output = []
        for i in range(ids.shape[0]):
            output.append(itemgetter(*ids.tolist()[i])(self.dataset["text"]))
        return output

    def save_pretrained(self, path: str) -> None:
        """ Save pretrained vocab to disk.

        Args:
            path (str): path to store vocab.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        faiss.write_index(self.index, os.path.join(path, "vocab.index"))
        self.dataset.to_csv(os.path.join(path, "vocab.dataset"))

    @classmethod
    def from_pretrained(cls, path: str):
        """ Load pretrained vocab from disk.

        Args:
            path (str): vocab path.
        """
        index = faiss.read_index(os.path.join(path, "vocab.index"))
        dataset = datasets.arrow_dataset.Dataset.from_csv(os.path.join(path, "vocab.dataset"))
        return cls(dataset, index)

    @classmethod
    def from_corpus(
            cls,
            path: str,
            model: CLIPModel,
            processor: CLIPProcessor,
            batch_size: int = 32,
            device: str = "cuda"
    ):
        """ Build vocab of prompts with their CLIP embeddings.

        Args:
            path (str): path to text file.
            model (CLIPModel): CLIP model.
            processor (CLIPProcessor): clip processor.
            batch_size (int): batch size.
            device (str): target device.
        Returns:
            Prompt corpus.
        """
        # initialize index
        index = faiss.IndexFlatIP(model.text_model.final_layer_norm.weight.size()[0])
        # load dataset
        dataset = load_dataset("text", data_files=path)
        loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=False)
        # process dataset
        with torch.inference_mode():
            for batch in tqdm(loader):
                inputs = processor(
                    text=batch["text"],
                    return_tensors="pt",
                    padding=True
                )
                text_features = model.get_text_features(**to_device(inputs, device))
                index.add(F.normalize(text_features, p=2, dim=1).float().cpu().numpy())
        return cls(dataset["train"], index)
