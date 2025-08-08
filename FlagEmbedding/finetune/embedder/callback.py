import os
import torch
import torch.distributed as dist  # <--- 导入 distributed
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl, PreTrainedTokenizer
from .hn_miner import HardNegativeMiner
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainDataset, AbsEmbedderDataArguments, AbsEmbedderModel

class DynamicHardNegativeMiningCallback(TrainerCallback):
    def __init__(self, data_args: AbsEmbedderDataArguments, training_args: TrainingArguments, tokenizer: PreTrainedTokenizer, trainer: Trainer):
        # ... (init 方法保持不变) ...
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.original_train_data_path = data_args.train_data[0] if data_args.train_data else None

    # ... (内部的 _create_encoding_functions 方法保持不变) ...
    def _create_encoding_functions(self, model: AbsEmbedderModel, tokenizer: PreTrainedTokenizer):
        """
        创建一个包装器，使训练模型能够像推理模型一样处理字符串列表。
        """
        # 将模型切换到评估模式，并禁用梯度计算，这对于推理至关重要
        model.eval()
        device = next(model.parameters()).device # 获取模型所在的设备

        def encode_wrapper(sentences: list, batch_size: int = 256):
            all_embeddings = []
            with torch.no_grad(): # 确保不计算梯度
                for start_index in tqdm(range(0, len(sentences), batch_size), desc="Encoding Sentences"):
                    sentences_batch = sentences[start_index:start_index + batch_size]
                    # 使用tokenizer处理字符串
                    inputs = tokenizer(
                        sentences_batch, 
                        padding=True, 
                        truncation=True, 
                        return_tensors='pt', 
                        max_length=self.data_args.passage_max_len
                    ).to(device)
                    
                    # 调用模型的底层encode方法
                    embeddings = model.encode(inputs).cpu().numpy()
                    all_embeddings.append(embeddings)
            
            return np.concatenate(all_embeddings, axis=0)
        
        class ModelWrapper:
            def encode(self, sentences, **kwargs):
                return encode_wrapper(sentences)

            def encode_queries(self, queries, **kwargs):
                return encode_wrapper(queries)

        return ModelWrapper()


    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if int(state.epoch) == 0 or not self.original_train_data_path:
            return

        # 1. 只有主进程执行挖掘任务
        if args.local_rank <= 0:
            print(f"Rank {args.local_rank}: Starting hard negative mining for epoch {int(state.epoch)}...")

            training_model = kwargs['model']
            
            inference_model_wrapper = self._create_encoding_functions(training_model, self.tokenizer)

            output_dir = args.output_dir
            new_negatives_file = os.path.join(output_dir, f"mined_negatives_epoch_{int(state.epoch)}.json")

            miner = HardNegativeMiner(
                model=inference_model_wrapper,
                use_gpu_for_searching=getattr(self.training_args, 'use_gpu_for_searching', False)
            )
        # ... (miner.mine 调用保持不变) ...
            miner.mine(
                input_file=self.original_train_data_path,
                output_file=new_negatives_file,
                candidate_pool=getattr(self.data_args, 'candidate_pool', None),
                sample_range=getattr(self.data_args, 'range_for_sampling', "10-210"),
                negative_number=getattr(self.data_args, 'negative_number', 15),
            )

            print(f"Finished hard negative mining. New negatives saved to {new_negatives_file}")
            training_model.train()

        # 2. 设置屏障，等待主进程完成文件写入
        if dist.is_initialized():
            dist.barrier()

        # 3. 所有进程现在都可以安全地加载新数据集
        print(f"Rank {args.local_rank}: Loading new dataset for epoch {int(state.epoch)}...")
        output_dir = args.output_dir
        new_negatives_file = os.path.join(output_dir, f"mined_negatives_epoch_{int(state.epoch)}.json")

        new_data_args = deepcopy(self.data_args)
        new_data_args.train_data = [new_negatives_file]

        new_dataset = AbsEmbedderTrainDataset(args=new_data_args, tokenizer=self.tokenizer)
        
        self.trainer.train_dataset = new_dataset
        # The trainer's dataloader will be re-created automatically.

        print(f"Rank {args.local_rank}: Trainer's dataset updated with new hard negatives for epoch {int(state.epoch)}.")