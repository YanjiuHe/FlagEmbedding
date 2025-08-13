import os
import math
import random
import logging
import datasets
import numpy as np
import torch.distributed as dist
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer, 
    DataCollatorWithPadding,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
import torch
import json
import faiss
import threading
from .AbsArguments import AbsEmbedderDataArguments, AbsEmbedderTrainingArguments

logger = logging.getLogger(__name__)


class AbsEmbedderTrainDataset(Dataset):
    """
    Abstract class for training dataset.
    增加了对在线难负例挖掘模式的支持。
    """
    def __init__(
        self,
        args: "AbsEmbedderDataArguments", # 使用引号避免循环导入
        tokenizer: PreTrainedTokenizer,
        # === 新增参数 ===
        online_mining: bool = False
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.shuffle_ratio = args.shuffle_ratio
        # === 新增属性 ===
        self.online_mining = online_mining

        train_datasets = []
        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                if not (data_dir.endswith('.json') or data_dir.endswith('.jsonl')): continue
                temp_dataset = self._load_dataset(data_dir)
                if len(temp_dataset) == 0: continue
                train_datasets.append(temp_dataset)
            else:
                for file in os.listdir(data_dir):
                    if not (file.endswith('.json') or file.endswith('.jsonl')): continue
                    temp_dataset = self._load_dataset(os.path.join(data_dir, file))
                    if len(temp_dataset) == 0: continue
                    train_datasets.append(temp_dataset)
        self.dataset = datasets.concatenate_datasets(train_datasets)

    def _load_dataset(self, file_path: str):
        """Load dataset from path.

        Args:
            file_path (str): Path to load the datasets from.

        Raises:
            ValueError: `pos_scores` and `neg_scores` not found in the features of training data

        Returns:
            datasets.Dataset: Loaded HF dataset.
        """
        if dist.get_rank() == 0:
            logger.info(f'loading data from {file_path} ...')

        temp_dataset = datasets.load_dataset('json', data_files=file_path, split='train', cache_dir=self.args.cache_path)
        if len(temp_dataset) > self.args.max_example_num_per_dataset:
            temp_dataset = temp_dataset.select(random.sample(list(range(len(temp_dataset))), self.args.max_example_num_per_dataset))
        if not self.args.knowledge_distillation:
            if 'pos_scores' in temp_dataset.column_names:
                temp_dataset = temp_dataset.remove_columns(['pos_scores'])
            if 'neg_scores' in temp_dataset.column_names:
                temp_dataset = temp_dataset.remove_columns(['neg_scores'])
        else:
            if 'pos_scores' not in temp_dataset.column_names or 'neg_scores' not in temp_dataset.column_names:
                raise ValueError(f"`pos_scores` and `neg_scores` not found in the features of training data in {file_path}, which is necessary when using knowledge distillation.")
        return temp_dataset

    def _shuffle_text(self, text):
        """shuffle the input text.

        Args:
            text (str): Input text.

        Returns:
            str: Shuffled text.
        """
        if self.shuffle_ratio > 0 and len(text) > 100 and random.random() < self.shuffle_ratio:
            split_text = []
            chunk_size = len(text)//3 + 1
            for i in range(0, len(text), chunk_size):
                split_text.append(text[i:i+chunk_size])
            random.shuffle(split_text)
            return " ".join(split_text)
        else:
            return text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # 从 HF Dataset 获取原始数据字典，这一步是关键
        data = self.dataset[item]

        # --- 如果是在线挖掘模式 ---
        if self.online_mining:
            # 在这种模式下，我们只提供 query 和 positives
            # Collator 将负责挖掘 negatives
            query = data['query']
            # (可选) 应用 instruction
            if self.args.query_instruction_for_retrieval is not None:
                query = self.args.query_instruction_format.format(
                    data.get('prompt', self.args.query_instruction_for_retrieval),
                    query
                )
            
            positives = data['pos']
            
            # (可选) 应用 instruction
            if self.args.passage_instruction_for_retrieval is not None:
                positives = [
                    self.args.passage_instruction_format.format(
                        self.args.passage_instruction_for_retrieval, p
                    ) for p in positives
                ]
            
            # 返回一个字典，格式与 Collator 的期望严格对应
            return {
                "query": query,
                "positives": positives
            }
        
        # --- 如果是原始模式 (保持不变) ---
        else:
            train_group_size = self.args.train_group_size

            query = data['query']
            if self.args.query_instruction_for_retrieval is not None:
                query = self.args.query_instruction_format.format(
                    data['prompt'] if 'prompt' in data else self.args.query_instruction_for_retrieval,
                    query
                )

            passages = []
            teacher_scores = []

            assert isinstance(data['pos'], list) and isinstance(data['neg'], list)

            pos_idx = random.choice(list(range(len(data['pos']))))
            passages.append(self._shuffle_text(data['pos'][pos_idx]))

            neg_all_idx = list(range(len(data['neg'])))
            if len(data['neg']) < train_group_size - 1:
                num = math.ceil((train_group_size - 1) / len(data['neg']))
                neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
            else:
                neg_idxs = random.sample(neg_all_idx, self.args.train_group_size - 1)
            for neg_idx in neg_idxs:
                passages.append(data['neg'][neg_idx])

            if self.args.knowledge_distillation:
                assert isinstance(data['pos_scores'], list) and isinstance(data['neg_scores'], list)
                teacher_scores.append(data['pos_scores'][pos_idx])
                for neg_idx in neg_idxs:
                    teacher_scores.append(data['neg_scores'][neg_idx])
                if not all(isinstance(score, (int, float)) for score in teacher_scores):
                    raise ValueError(f"pos_score or neg_score must be digit")
            else:
                teacher_scores = None

            if self.args.passage_instruction_for_retrieval is not None:
                passages = [
                    self.args.passage_instruction_format.format(
                        self.args.passage_instruction_for_retrieval, p
                    )
                    for p in passages
                ]

            return query, passages, teacher_scores

@dataclass
class AbsEmbedderCollator(DataCollatorWithPadding):
    """
    The abstract embedder collator.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    sub_batch_size: int = -1

    def __call__(self, features):
        queries = [f[0] for f in features]
        passages = [f[1] for f in features]
        teacher_scores = [f[2] for f in features]
        if teacher_scores[0] is None:
            teacher_scores = None
        elif isinstance(teacher_scores[0], list):
            teacher_scores = sum(teacher_scores, [])

        if isinstance(queries[0], list):
            queries = sum(queries, [])
        if isinstance(passages[0], list):
            passages = sum(passages, [])

        queries_inputs = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None
        )
        passages_inputs = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors=None
        )

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.tokenizer.pad(
                queries_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors
            )
            d_collated = self.tokenizer.pad(
                passages_inputs,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors
            )
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(queries_inputs['attention_mask']), i + batch_size)
                sub_features = {}
                for k, v in queries_inputs.items():
                    sub_features[k] = v[start:end]
                q_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.query_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors
                ))

            d_collated = []
            for i in range(0, len(passages_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(passages_inputs['attention_mask']), i + batch_size)
                sub_features = {}

                for k, v in passages_inputs.items():
                    sub_features[k] = v[start:end]
                d_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors
                ))
        return {
            "queries": q_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": False
        }


class AbsEmbedderSameDatasetTrainDataset(AbsEmbedderTrainDataset):
    """Abstract class for training dataset that samples batches from same dataset.

    Args:
        args (AbsEmbedderDataArguments): Data arguments.
        default_batch_size (int): The default batch size for training.
        seed (int): Random seed.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        process_index (int, optional): Current process index. Defaults to 0.
        num_processes (int, optional): Total number of processes. Defaults to 1.
    """
    def __init__(
        self,
        args: AbsEmbedderDataArguments,
        default_batch_size: int,
        seed: int,
        tokenizer: PreTrainedTokenizer,
        process_index: int=0,
        num_processes: int=1
    ):
        self.args = args
        self.shuffle_ratio = args.shuffle_ratio
        self.defaut_batch_size = default_batch_size
        self.deterministic_generator = np.random.default_rng(seed)
        self.tokenizer = tokenizer
        self.process_index = process_index
        self.num_processes = num_processes

        self.step = 0

        train_datasets = []
        each_data_idxs = []
        batch_size_idxs = []
        no_in_batch_neg_flags = []
        cur_all_num = 0

        small_threshold = args.small_threshold
        drop_threshold = args.drop_threshold

        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                # Add `no_in_batch_neg` **suffix** to `data_dir` to indicate that this dataset does not use in-batch negatives
                no_in_batch_neg_flag = data_dir.split('.')[-2].endswith('no_in_batch_neg')
                if not (data_dir.endswith('.json') or data_dir.endswith('.jsonl')): continue
                temp_dataset = self._load_dataset(data_dir)

                if len(temp_dataset) == 0 or len(temp_dataset) < small_threshold: continue
                else:
                    train_datasets.append(temp_dataset)
                    each_data_idxs.append(np.arange(len(temp_dataset)) + cur_all_num)
                    cur_all_num += len(temp_dataset)
                    batch_size_idxs.append(self._get_file_batch_size(temp_dataset, default_batch_size))
                    no_in_batch_neg_flags.append(no_in_batch_neg_flag)

            else:
                small_datasets = []
                small_batch_size = math.inf

                # Add `no_in_batch_neg` **suffix** to `data_dir` to indicate that this dataset does not use in-batch negatives
                no_in_batch_neg_flag = data_dir.endswith('no_in_batch_neg')
                for file in os.listdir(data_dir):
                    if not (file.endswith('.json') or file.endswith('.jsonl')): continue
                    temp_dataset = self._load_dataset(os.path.join(data_dir, file))

                    if len(temp_dataset) == 0: continue
                    elif len(temp_dataset) < small_threshold:
                        small_datasets.append(temp_dataset)
                        small_batch_size = min(small_batch_size, self._get_file_batch_size(temp_dataset, default_batch_size))
                    else:
                        train_datasets.append(temp_dataset)
                        each_data_idxs.append(np.arange(len(temp_dataset)) + cur_all_num)
                        cur_all_num += len(temp_dataset)
                        batch_size_idxs.append(self._get_file_batch_size(temp_dataset, default_batch_size))
                        no_in_batch_neg_flags.append(no_in_batch_neg_flag)

                if len(small_datasets) > 0:
                    small_dataset = datasets.concatenate_datasets(small_datasets)
                    if len(small_dataset) >= drop_threshold:
                        train_datasets.append(small_dataset)
                        each_data_idxs.append(np.arange(len(small_dataset)) + cur_all_num)
                        cur_all_num += len(small_dataset)
                        batch_size_idxs.append(small_batch_size)
                        no_in_batch_neg_flags.append(no_in_batch_neg_flag)

        self.dataset = datasets.concatenate_datasets(train_datasets)
        self.each_data_idxs = each_data_idxs
        self.datasets_inxs = np.arange(len(each_data_idxs))
        self.batch_size_idxs = batch_size_idxs
        self.no_in_batch_neg_flags = no_in_batch_neg_flags

        self.refresh_epoch()

    def _load_dataset(self, file_path: str):
        """Load datset from given path.

        Args:
            file_path (str): The path to load or download from HF hub.

        Returns:
            datasets.Dataset: The loaded dataset.
        """
        if dist.get_rank() == 0:
            logger.info(f'loading data from {file_path} ...')

        temp_dataset = datasets.load_dataset('json', data_files=file_path, split='train', cache_dir=self.args.cache_path)
        if len(temp_dataset) > self.args.max_example_num_per_dataset:
            temp_dataset = temp_dataset.select(random.sample(list(range(len(temp_dataset))), self.args.max_example_num_per_dataset))
        if not self.args.knowledge_distillation:
            if 'pos_scores' in temp_dataset.column_names:
                temp_dataset = temp_dataset.remove_columns(['pos_scores'])
            if 'neg_scores' in temp_dataset.column_names:
                temp_dataset = temp_dataset.remove_columns(['neg_scores'])
        return temp_dataset

    @staticmethod
    def _get_file_batch_size(temp_dataset: datasets.Dataset, default_batch_size: int):
        """Get the appropriate batch size for the dataset.

        Args:
            temp_dataset (datasets.Dataset): Loaded :data:`datasets.Dataset` object.
            default_batch_size (int): The default batch size to use if not specified in the dataset.

        Returns:
            int: The final batch size to use.
        """
        if 'batch_size' in temp_dataset.column_names:
            return temp_dataset['batch_size'][0]
        if 'type' in temp_dataset.column_names:
            data_type = temp_dataset['type'][0]
            if 'symmetric' in data_type:
                return default_batch_size // 2  # make the symmetric data have smaller batch size
        return default_batch_size

    def refresh_epoch(self):
        """
        Refresh data for epoch.
        """
        logger.info(f'-- Rank {self.process_index}: refresh data --')
        self.deterministic_generator.shuffle(self.datasets_inxs)

        batch_datas = []
        for dataset_inx in self.datasets_inxs:
            self.deterministic_generator.shuffle(self.each_data_idxs[dataset_inx])
            cur_batch_size = self.batch_size_idxs[dataset_inx]*self.num_processes
            no_in_batch_neg_flag = self.no_in_batch_neg_flags[dataset_inx]
            for start_index in range(0, len(self.each_data_idxs[dataset_inx]), cur_batch_size):
                # judge the last batch's length
                if len(self.each_data_idxs[dataset_inx]) - start_index < cur_batch_size:
                    break
                batch_datas.append((
                    self.each_data_idxs[dataset_inx][start_index:start_index+cur_batch_size],
                    no_in_batch_neg_flag
                ))
        self.deterministic_generator.shuffle(batch_datas)
        self.batch_datas = batch_datas
        self.step = 0

    def __len__(self):
        return len(self.batch_datas) * self.num_processes

    def __getitem__(self, _):
        batch_indices, no_in_batch_neg_flag = self.batch_datas[self.step]    # extend here
        cur_batch_size = int(len(batch_indices) / self.num_processes)
        batch_indices = batch_indices[self.process_index * cur_batch_size: (self.process_index + 1) * cur_batch_size]
        batch_data = self.dataset[batch_indices]
        self.step += 1
        queries, passages, teacher_scores = self._create_batch_data(batch_raw_data=batch_data)
        return queries, passages, teacher_scores, no_in_batch_neg_flag

    def _get_train_group_size(self, batch_raw_data):
        """Get the training group size and data type.

        Args:
            batch_raw_data (datasets.Dataset): One batch of raw data.

        Returns:
            int: The training group size.
            str: The type of data for the task.
        """
        if 'type' in batch_raw_data:
            data_type = batch_raw_data['type'][0]
            if data_type in ['only_1neg']:
                return 2, data_type
            elif data_type in ['symmetric_class']:
                return min(len(batch_raw_data['neg'][0]) + 1, self.args.train_group_size), data_type
            else:
                return self.args.train_group_size, data_type
        elif 'train_group_size' in batch_raw_data:
            train_group_size = batch_raw_data['train_group_size'][0]
            if isinstance(train_group_size, int) and train_group_size > 0:
                return train_group_size, None
            else:
                return self.args.train_group_size, None
        return self.args.train_group_size, None

    def _create_batch_data(self, batch_raw_data):
        """Create a comple batch of data with queries, documents and teacher scores.

        Args:
            batch_raw_data (datasets.Dataset): One batch of raw data.

        Returns:
            List[str]: Queries with instruction format.
            List[str]: Documents with instruction format.
            List[float]: Teacher scores for model distillation.
        """
        queries, passages, teacher_scores = [], [], []

        train_group_size, data_type = self._get_train_group_size(batch_raw_data)

        for i in range(len(batch_raw_data['query'])):
            if data_type is not None:
                assert batch_raw_data['type'][i] == data_type, f"Data type is not consistent in the same batch"

            queries.append(
                self.args.query_instruction_format.format(
                    batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
                    batch_raw_data['query'][i]
                )
            )
            tmp_passages = []
            pos_idx = random.choice(list(range(len(batch_raw_data['pos'][i]))))
            pos = self._shuffle_text(batch_raw_data['pos'][i][pos_idx])
            tmp_passages.append(pos)

            neg_all_idx = list(range(len(batch_raw_data['neg'][i])))
            if len(batch_raw_data['neg'][i]) < train_group_size - 1:
                num = math.ceil((train_group_size - 1) / len(batch_raw_data['neg'][i]))
                neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
            else:
                neg_idxs = random.sample(neg_all_idx, train_group_size - 1)
            for neg_idx in neg_idxs:
                tmp_passages.append(batch_raw_data['neg'][i][neg_idx])

            if self.args.knowledge_distillation:
                if 'pos_scores' in batch_raw_data and batch_raw_data['pos_scores'][i] is not None:
                    teacher_scores.append(batch_raw_data['pos_scores'][i][pos_idx])
                for neg_idx in neg_idxs:
                    if 'neg_scores' in batch_raw_data and batch_raw_data['neg_scores'][i] is not None:
                        teacher_scores.append(batch_raw_data['neg_scores'][i][neg_idx])
            else:
                teacher_scores = None

            if data_type is not None and data_type in ['symmetric_sts', 'symmetric_clustering']:
                tmp_passages = [
                    self.args.query_instruction_format.format(
                        batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
                        p
                    ) for p in tmp_passages
                ]
            else:
                if self.args.passage_instruction_for_retrieval is not None:
                    tmp_passages = [
                        self.args.passage_instruction_format.format(
                            self.args.passage_instruction_for_retrieval, p
                        ) for p in tmp_passages
                    ]

            passages.extend(tmp_passages)

            if teacher_scores is not None:
                if len(teacher_scores) > 0 and len(passages) > 0:
                    assert len(teacher_scores) == len(passages)

        return queries, passages, teacher_scores


@dataclass
class AbsEmbedderSameDatasetCollator(DataCollatorWithPadding):
    """
    EmbedCollator for SameDataset.
    Note that after using this collator, the training_args should be set as:
    
    ``training_args.per_device_train_batch_size = 1``
    
    ``training_args.dataloader_num_workers = 0    # avoid multi-processing``
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    sub_batch_size: int = -1

    def __call__(self, features):
        queries = features[0][0]
        passages = features[0][1]
        teacher_scores = features[0][2]
        no_in_batch_neg_flag = features[0][3]

        queries_inputs = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None
        )
        passages_inputs = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors=None
        )

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.tokenizer.pad(
                queries_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            d_collated = self.tokenizer.pad(
                passages_inputs,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(queries_inputs['attention_mask']), i + batch_size)
                sub_features = {}
                for k, v in queries_inputs.items():
                    sub_features[k] = v[start:end]
                q_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.query_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

            d_collated = []
            for i in range(0, len(passages_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(passages_inputs['attention_mask']), i + batch_size)
                sub_features = {}

                for k, v in passages_inputs.items():
                    sub_features[k] = v[start:end]
                d_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

        if isinstance(teacher_scores, list) and len(teacher_scores) == 0:
            teacher_scores = None

        return {
            "queries": q_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": no_in_batch_neg_flag
        }


class EmbedderTrainerCallbackForDataRefresh(TrainerCallback):
    """
    Callback class to inspect the state of the training loop and take decision.
    """
    def __init__(self, train_dataset: AbsEmbedderSameDatasetTrainDataset):
        self.train_dataset = train_dataset

    def on_epoch_end(
        self,
        args: AbsEmbedderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event called at the end of an epoch.
        """
        self.train_dataset.refresh_epoch()


@dataclass
class OnlineHardNegativeCollator:
    tokenizer: PreTrainedTokenizer
    model: torch.nn.Module
    output_dir: str
    query_max_len: int = 32
    passage_max_len: int = 128
    negative_number: int = 15
    sample_range: str = "10-210"
    use_gpu_for_searching: bool = False
    
    # 为多进程加载添加线程锁，防止重复加载
    _lock = threading.Lock()
    _current_index_path: str = field(init=False, default=None)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self):
        self.index = None
        self.corpus: list[str] = []
        self.sample_range_tup = [int(x) for x in self.sample_range.split('-')]
        self.device = next(self.model.parameters()).device
        self.meta_file_path = os.path.join(self.output_dir, "current_index.meta")

    def update_index_for_new_epoch(self, epoch: int, output_dir: str):
        # 这个方法不再直接加载，而是只更新路径
        index_path = os.path.join(output_dir, f"corpus_index_epoch_{epoch}.faiss")
        corpus_path = os.path.join(output_dir, f"corpus_texts_epoch_{epoch}.json")
        
        if os.path.exists(index_path) and os.path.exists(corpus_path):
            print(f"[Collator Update] New index and corpus paths are set for epoch {epoch}.")
            self._index_path = index_path
            self._corpus_path = corpus_path
            # 重置内部状态，强制下次调用时重新加载
            self.index = None
            self.corpus = []
        else:
            print(f"[Collator Update] WARN: Index/corpus for epoch {epoch} not found. Paths not updated.")
    
    def _ensure_index_is_loaded(self):
        with self._lock:
            # 1. 检查 meta 文件是否存在
            if not os.path.exists(self.meta_file_path):
                # 第一次调用时 meta 文件可能不存在，这很正常
                return

            # 2. 读取最新的路径信息
            try:
                with open(self.meta_file_path, 'r', encoding='utf-8') as f:
                    latest_meta = json.load(f)
                latest_index_path = latest_meta.get("index_path")
            except (json.JSONDecodeError, FileNotFoundError):
                # 文件可能正在被写入，暂时读取失败，下次再试
                return

            # 3. 检查当前加载的索引是否已是最新
            if self._current_index_path == latest_index_path:
                return  # 已经是最新了，无需操作

            # 4. 加载新的索引
            print(f"[Worker PID: {os.getpid()}] New index detected. Loading from {latest_index_path}")
            try:
                self.index = faiss.read_index(latest_index_path)
                with open(latest_meta["corpus_path"], 'r', encoding='utf-8') as f:
                    self.corpus = json.load(f)
                
                # 更新内部状态，记录当前已加载的路径
                self._current_index_path = latest_index_path
                print(f"[Worker PID: {os.getpid()}] Successfully loaded new index.")
            except Exception as e:
                print(f"[Worker PID: {os.getpid()}] ERROR loading new index: {e}")
                # 加载失败，不更新 _current_index_path，下次会重试

    def __call__(self, features: list[dict[str, any]]) -> dict[str, any]:
        # `features` 是一个 batch 的数据, e.g., [{'query': q1, 'positives': [p1, p2]}, ...]
        self._ensure_index_is_loaded()
        queries = [f['query'] for f in features]
        all_positives = [f['positives'] for f in features]

        # 1. 准备 passages
        # 遵循原版逻辑，每个query配一个positive和多个negative
        final_passages = []
        
        self._ensure_index_is_loaded()
        
        if self.index is None:
            # 只有在 _ensure_index_is_loaded 之后，index 仍然是 None
            # 才打印警告，并且只打印一次
            if not hasattr(self, '_initial_warned'):
                print(f"[Worker PID: {os.getpid()}] WARN: Index not available. Using random negatives.")
                self._initial_warned = True
            hard_negatives = self._get_random_negatives([f['positives'] for f in features])
        else:
             hard_negatives = self._find_hard_negatives([f['query'] for f in features], [f['positives'] for f in features])

        for i in range(len(queries)):
            # 从positives列表中随机选一个
            positive_passage = random.choice(all_positives[i])
            final_passages.append(positive_passage)

            # 获取挖掘到的或随机的负例
            negs = hard_negatives[i]
            
            # 如果负例不够，用随机负例补充
            if len(negs) < self.negative_number:
                num_needed = self.negative_number - len(negs)
                negs.extend(self._get_random_negatives_for_one_sample(all_positives[i], num_needed))
            
            final_passages.extend(negs)
            
        # 2. Tokenize, 模仿原始 Collator 的行为
        queries_inputs = self.tokenizer(
            queries,
            padding=False, # 先不padding
            truncation=True,
            max_length=self.query_max_len,
        )
        passages_inputs = self.tokenizer(
            final_passages,
            padding=False,
            truncation=True,
            max_length=self.passage_max_len,
        )

        # 3. Pad, 模仿原始 Collator 的行为
        q_collated = self.tokenizer.pad(
            queries_inputs,
            padding=True,
            return_tensors='pt',
        )
        d_collated = self.tokenizer.pad(
            passages_inputs,
            padding=True,
            return_tensors='pt',
        )


        target_device = next(self.model.parameters()).device

        # 2. 将嵌套字典中的所有张量移动到目标设备
        q_collated_on_device = {k: v.to(target_device) for k, v in q_collated.items()}
        d_collated_on_device = {k: v.to(target_device) for k, v in d_collated.items()}

        # 返回与原始Collator完全相同的格式
        return {
            "queries": q_collated_on_device,
            "passages": d_collated_on_device,
            "teacher_scores": None,  # 在这个方案中不处理知识蒸馏
            "no_in_batch_neg_flag": False # 或者根据你的逻辑设置
        }

    def _find_hard_negatives(self, queries: list[str], all_positives: list[list[str]]) -> list[list[str]]:
        # 切换模型到评估模式并禁用梯度


        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            q_inputs = self.tokenizer(
                queries, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=self.query_max_len
            )
            q_inputs_on_device = {key: val.to(next(self.model.parameters()).device) for key, val in q_inputs.items()}

            q_vecs = self.model.encode(q_inputs_on_device).cpu().numpy().astype(np.float32)

        # 恢复模型状态
        if was_training:
            self.model.train()

        # 如果索引是空的，直接返回随机负例，避免在 index.search 崩溃
        if self.index is None:
            print("警告: 索引未加载，无法执行难负例搜索，将返回随机负例。")
            return self._get_random_negatives(all_positives)

        # 在索引中搜索
        top_k = self.sample_range_tup[1]
        _, all_inxs = self.index.search(q_vecs, k=top_k)

        # ... (剩余的筛选逻辑保持不变) ...
        batch_negatives = []
        for i in range(len(queries)):
            pos_set = set(all_positives[i])
            query = queries[i]
            inxs = all_inxs[i][self.sample_range_tup[0]:self.sample_range_tup[1]]
            hard_negs = []
            for inx in inxs:
                if inx == -1: continue
                candidate_text = self.corpus[inx]
                if candidate_text != query and candidate_text not in pos_set:
                    hard_negs.append(candidate_text)
                if len(hard_negs) >= self.negative_number:
                    break
            batch_negatives.append(hard_negs)
            
        return batch_negatives

    def _get_random_negatives(self, all_positives: list[list[str]]) -> list[list[str]]:
        """为整个 batch 生成随机负例"""
        batch_negatives = []
        if not self.corpus:
            # 极端情况：如果连语料库都没有，就返回空列表
            return [[] for _ in all_positives]
            
        for positives in all_positives:
            batch_negatives.append(self._get_random_negatives_for_one_sample(positives, self.negative_number))
        return batch_negatives

    def _get_random_negatives_for_one_sample(self, positives: list[str], num_to_get: int) -> list[str]:
        """为一个样本生成随机负例"""
        pos_set = set(positives)
        negs = []
        if not self.corpus: return negs
        
        while len(negs) < num_to_get:
            rand_doc = random.choice(self.corpus)
            if rand_doc not in pos_set:
                negs.append(rand_doc)
        return negs