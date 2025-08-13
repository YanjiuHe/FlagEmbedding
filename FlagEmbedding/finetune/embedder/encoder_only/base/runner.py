import logging
import os
import json
from abc import ABC
from typing import List, Optional
from typing import Tuple
from transformers import (
    AutoModel, AutoConfig,
    AutoTokenizer, PreTrainedTokenizer
)

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderRunner, AbsEmbedderModel, EmbedderTrainerCallbackForDataRefresh
from FlagEmbedding.abc.finetune.embedder.AbsDataset import AbsEmbedderTrainDataset, OnlineHardNegativeCollator
from FlagEmbedding.finetune.embedder.callback import IndexUpdateCallback
# 假设你的新组件都在这些路径下
from .modeling import BiEncoderOnlyEmbedderModel
from .trainer import EncoderOnlyEmbedderTrainer

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderRunner(AbsEmbedderRunner):
    """
    Finetune Runner for base embedding models.
    """
    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
        """Load tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code
        )
        base_model = AutoModel.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code
        )

        num_labels = 1
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        logger.info('Config: %s', config)

        model = BiEncoderOnlyEmbedderModel(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
        return tokenizer, model


    def load_trainer(self) -> EncoderOnlyEmbedderTrainer:
        """Load the trainer.

        Returns:
            EncoderOnlyEmbedderTrainer: Loaded trainer instance.
        """
        
        # --- 新逻辑分支：在线难负例挖掘 ---
        if self.training_args.dynamic_hn_mining:
            logger.info("使用在线难负例挖掘策略 (Online Hard Negative Mining Strategy).")
            
            # 1. 替换为新的数据集和整理器
            # 注意: self.train_dataset 和 self.data_collator 是在 AbsEmbedderRunner 的 __init__ 中
            # 由 self.load_dataset() 创建的。我们需要在这里覆盖它们。
            logger.info("以在线挖掘模式重新加载 AbsEmbedderTrainDataset...")
            self.train_dataset = AbsEmbedderTrainDataset(
                args=self.data_args,
                tokenizer=self.tokenizer,
                online_mining=True  # <-- 核心改动在这里
            )
            
            logger.info("创建 OnlineHardNegativeCollator...")
            self.data_collator = OnlineHardNegativeCollator(
                tokenizer=self.tokenizer,
                model=self.model,  # 将已经加载的模型实例传递进去
                query_max_len=self.data_args.query_max_len,
                passage_max_len=self.data_args.passage_max_len,
                negative_number=self.data_args.negative_number,
                sample_range=self.data_args.range_for_sampling,
                use_gpu_for_searching=getattr(self.training_args, 'use_gpu_for_searching', False),
                output_dir=self.training_args.output_dir
            )

            # 2. 创建 Trainer 实例
            trainer = EncoderOnlyEmbedderTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer
            )

            # 3. 添加新的 Callback
            logger.info("添加 IndexUpdateCallback...")
            index_update_callback = IndexUpdateCallback(
                data_args=self.data_args,
                training_args=self.training_args,
                tokenizer=self.tokenizer
            )
            trainer.add_callback(index_update_callback)

            # (可选) 我们不再需要原始的 EmbedderTrainerCallbackForDataRefresh
            # 因为我们的新 Dataset 每次只返回一个 (query, pos) 对，不需要在 batch 内部刷新

            # (可选但推荐) 增加训练开始前创建初始索引的逻辑
            self._create_initial_index_if_needed()
            
            return trainer

        # --- 原始逻辑分支：保持不变 ---
        else:
            logger.info("使用默认训练策略。")
            trainer = EncoderOnlyEmbedderTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer
            )
            if self.data_args.same_dataset_within_batch:
                trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
            
            # 这里是原始的难样本挖掘回调，现在被我们的新逻辑替代了
            # if self.training_args.dynamic_hn_mining:
            #     from FlagEmbedding.finetune.embedder.callback import DynamicHardNegativeMiningCallback
            #     trainer.add_callback(DynamicHardNegativeMiningCallback(self.data_args, self.training_args, self.tokenizer, trainer))

            return trainer


    def _create_initial_index_if_needed(self):
        """在训练开始前，为第一个epoch创建初始索引 (仅在主进程执行)"""
        import torch.distributed as dist
        
        # 只有主进程负责创建索引
        if self.training_args.local_rank > 0:
            if dist.is_initialized():
                dist.barrier()
            return
            
        logger.info("为 epoch 0 创建初始索引...")
        # 延迟导入，避免不使用该功能时也需要安装faiss
        from FlagEmbedding.finetune.embedder.hn_miner import HardNegativeMiner 
        from FlagEmbedding.finetune.embedder.model_wrapper import create_encoding_model_wrapper
        
        # 创建一个临时的推理模型包装器
        inference_model_wrapper = create_encoding_model_wrapper(
            self.model, self.tokenizer, self.data_args.passage_max_len, self.training_args
        )
        miner = HardNegativeMiner(
            model=inference_model_wrapper,
            use_gpu_for_searching=getattr(self.training_args, 'use_gpu_for_searching', False)
        )
        
        # 构建并保存索引
        index_path, corpus_path = miner.build_corpus_index(
            corpus_sources=self.data_args.train_data,
            output_dir=self.training_args.output_dir,
            epoch=0, # 使用-1作为初始索引的标记
            faiss_use_gpu=getattr(self.training_args, 'use_gpu_for_searching', False)
        )

        meta_file_path = os.path.join(self.training_args.output_dir, "current_index.meta")
        meta_data = {
            "index_path": index_path,
            "corpus_path": corpus_path
        }
        with open(meta_file_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f)
        print(f"Created initial index meta file at {meta_file_path}")
        

        print("--------已加载初始索引，准备开始训练--------")
        
        # 恢复模型到训练状态 (如果 model_wrapper 改变了它)
        self.model.train()
        
        # 设置屏障，确保所有进程在主进程完成索引创建和加载后才继续
        if dist.is_initialized():
            dist.barrier()