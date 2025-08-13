import json
import random
from typing import Optional, List
import os

import faiss
import torch
# import torch_npu # 如果你使用NPU
# from torch_npu.contrib import transfer_to_npu
import numpy as np
from tqdm import tqdm

from FlagEmbedding.abc.inference import AbsEmbedder

class HardNegativeMiner:
    # __init__ 和 _create_index 方法保持不变
    def __init__(
        self,
        model: AbsEmbedder,
        use_gpu_for_searching: bool = False,
    ):
        self.model = model
        self.use_gpu_for_searching = use_gpu_for_searching

    def _create_index(self, embeddings: np.ndarray, use_gpu: bool):
        index = faiss.IndexFlatIP(len(embeddings[0]))
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if use_gpu:
            # 使用 try-except 来优雅地处理没有 GPU 的情况
            try:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                index = faiss.index_cpu_to_all_gpus(index, co=co)
            except Exception as e:
                print(f"Failed to use GPU for Faiss index: {e}. Falling back to CPU.")
        index.add(embeddings)
        return index

    def _get_corpus(self, data_sources: List[str]) -> List[str]:
        corpus = set()
        for source in data_sources:
            if os.path.isfile(source):
                with open(source, "r", encoding="utf-8") as f:
                    for line in f:
                        line = json.loads(line.strip())
                        corpus.update(line.get('pos', []))
                        # 初始语料库也可以包含一些 neg
                        corpus.update(line.get('neg', []))
            else: # 如果源是目录
                # (可以添加处理目录的逻辑)
                pass
        return list(corpus)

    def build_corpus_index(
        self,
        corpus_sources: List[str],
        output_dir: str,
        epoch: int,
        faiss_use_gpu: bool = False,
    ):
        """
        只编码语料库，构建并保存索引和语料库本身。
        """
        print(f"Building corpus and index for epoch {epoch}...")
        
        # 1. 收集语料库
        # 我们需要一个稳定的语料库来源，这里假设是原始训练文件
        corpus = self._get_corpus(corpus_sources)
        print(f"Collected {len(corpus)} unique passages for the corpus.")

        # 2. 编码语料库
        print(f'Inferencing embedding for corpus (number={len(corpus)})')
        p_vecs = self.model.encode(sentences=corpus)
        if isinstance(p_vecs, dict):
            p_vecs = p_vecs["dense_vecs"]
        
        # 3. 创建并保存 FAISS 索引
        index = self._create_index(p_vecs, use_gpu=faiss_use_gpu)
        index_path = os.path.join(output_dir, f"corpus_index_epoch_{epoch}.faiss")
        # 如果索引在GPU上，需要移回CPU才能保存
        if hasattr(faiss, 'index_gpu_to_cpu'):
             if faiss.get_num_gpus() > 0 and isinstance(index, faiss.GpuIndex):
                index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index, index_path)

        # 4. 保存语料库文本
        corpus_path = os.path.join(output_dir, f"corpus_texts_epoch_{epoch}.json")
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f)

        print(f"Corpus index saved to {index_path}")
        print(f"Corpus texts saved to {corpus_path}")
        return index_path, corpus_path