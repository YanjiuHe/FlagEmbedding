import json
import random
from typing import Optional
import os

import faiss
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import numpy as np
from tqdm import tqdm

from FlagEmbedding.abc.inference import AbsEmbedder


class HardNegativeMiner:
    def __init__(
        self,
        model: AbsEmbedder,
        use_gpu_for_searching: bool = False,
    ):
        self.model = model
        self.use_gpu_for_searching = use_gpu_for_searching

    def _create_index(self, embeddings: np.ndarray):
        index = faiss.IndexFlatIP(len(embeddings[0]))
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.use_gpu_for_searching:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co=co)
        index.add(embeddings)
        return index

    def _batch_search(
        self,
        index: faiss.Index,
        query: np.ndarray,
        topk: int = 200,
        batch_size: int = 64
    ):
        all_scores, all_inxs = [], []
        for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
            batch_query = query[start_index:start_index + batch_size]
            batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
            all_scores.extend(batch_scores.tolist())
            all_inxs.extend(batch_inxs.tolist())
        return all_scores, all_inxs

    def _get_corpus(self, candidate_pool: str):
        corpus = []
        with open(candidate_pool, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                corpus.append(line['text'])
        return corpus

    def mine(
        self,
        input_file: str,
        output_file: str,
        candidate_pool: Optional[str] = None,
        sample_range: str = "10-210",
        negative_number: int = 15,
    ):
        corpus = []
        queries = []
        train_data = []
        for line in open(input_file):
            line = json.loads(line.strip())
            train_data.append(line)
            corpus.extend(line['pos'])
            if 'neg' in line:
                corpus.extend(line['neg'])
            queries.append(line['query'])

        if candidate_pool is not None:
            if not isinstance(candidate_pool, list):
                candidate_pool = self._get_corpus(candidate_pool)
            corpus = list(set(candidate_pool))
        else:
            corpus = list(set(corpus))

        print(f'Inferencing embedding for queries (number={len(queries)})--------------')
        q_vecs = self.model.encode_queries(queries=queries)
        print(f'Inferencing embedding for corpus (number={len(corpus)})--------------')
        p_vecs = self.model.encode(sentences=corpus)

        if isinstance(p_vecs, dict):
            p_vecs = p_vecs["dense_vecs"]
        if isinstance(q_vecs, dict):
            q_vecs = q_vecs["dense_vecs"]

        print('Create index and search------------------')
        sample_range_tup = [int(x) for x in sample_range.split('-')]
        index = self._create_index(p_vecs)
        _, all_inxs = self._batch_search(index, q_vecs, topk=sample_range_tup[-1])
        assert len(all_inxs) == len(train_data)

        for i, data in enumerate(train_data):
            query = data['query']
            inxs = all_inxs[i][sample_range_tup[0]:sample_range_tup[1]]
            filtered_inx = []
            for inx in inxs:
                if inx == -1: break
                if corpus[inx] not in data['pos'] and corpus[inx] != query:
                    filtered_inx.append(inx)

            if len(filtered_inx) > negative_number:
                filtered_inx = random.sample(filtered_inx, negative_number)
            data['neg'] = [corpus[inx] for inx in filtered_inx]

        with open(output_file, 'w') as f:
            for data in train_data:
                if len(data['neg']) < negative_number:
                    samples = random.sample(corpus, negative_number - len(data['neg']) + len(data['pos']))
                    samples = [sent for sent in samples if sent not in data['pos']]
                    data['neg'].extend(samples[: negative_number - len(data['neg'])])
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
