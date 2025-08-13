# callbacks.py
import os
import torch
import torch.distributed as dist
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import json
import logging
import sys


from .hn_miner import HardNegativeMiner 
from .model_wrapper import create_encoding_model_wrapper # 将你原来的 _create_encoding_functions 提取出来

class IndexUpdateCallback(TrainerCallback):
    def __init__(self, data_args, training_args, tokenizer):
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.original_train_data_path = data_args.train_data if data_args.train_data else None

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.original_train_data_path:
            return

        epoch = int(state.epoch)
        
        # 1. 只有主进程执行索引构建任务
        if args.local_rank <= 0:
            print(f"Rank {args.local_rank}: Starting corpus index update for end of epoch {epoch}...")

            training_model = kwargs['model']
            
            # 使用包装器来获得推理能力
            inference_model_wrapper = create_encoding_model_wrapper(
                training_model, self.tokenizer, self.data_args.passage_max_len, self.training_args
            )
            
            miner = HardNegativeMiner(
                model=inference_model_wrapper,
                # 注意：这里的 use_gpu_for_searching 指的是构建索引时的设备
                use_gpu_for_searching=getattr(self.training_args, 'use_gpu_for_searching', False)
            )

            # 构建并保存索引
            index_path, corpus_path = miner.build_corpus_index(
                corpus_sources=self.original_train_data_path,
                output_dir=args.output_dir,
                epoch=epoch
            )
            meta_file_path = os.path.join(args.output_dir, "current_index.meta")
            meta_data = {
                "index_path": index_path,
                "corpus_path": corpus_path
            }
            with open(meta_file_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f)
            
            print(f"Updated index meta file at {meta_file_path}")
            
            # 恢复模型到训练模式
            training_model.train()

        # 2. 设置屏障，等待主进程完成文件写入
        if dist.is_initialized():
            dist.barrier()

        print(f"Rank {args.local_rank}: DataCollator ready for next epoch.")