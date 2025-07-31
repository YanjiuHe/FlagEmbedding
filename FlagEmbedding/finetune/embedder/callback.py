import os
from copy import deepcopy
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from .hn_miner import HardNegativeMiner
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainDataset, AbsEmbedderDataArguments


class DynamicHardNegativeMiningCallback(TrainerCallback):
    def __init__(self, data_args: AbsEmbedderDataArguments, training_args: TrainingArguments):
        self.data_args = data_args
        self.training_args = training_args
        self.original_train_data_path = data_args.train_data

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if int(state.epoch) == 0:
            # Skip mining for the first epoch
            return

        print(f"Starting hard negative mining for epoch {int(state.epoch)}...")

        model = kwargs['model']

        # Define paths for the new hard negative file
        output_dir = args.output_dir
        new_negatives_file = os.path.join(output_dir, f"mined_negatives_epoch_{int(state.epoch)}.json")

        # Instantiate the miner and mine new negatives
        miner = HardNegativeMiner(model, use_gpu_for_searching=getattr(self.training_args, 'use_gpu_for_searching', False))
        miner.mine(
            input_file=self.original_train_data_path[0], # The original script expects a single file
            output_file=new_negatives_file,
            candidate_pool=getattr(self.data_args, 'candidate_pool', None),
            sample_range=getattr(self.data_args, 'range_for_sampling', "10-210"),
            negative_number=getattr(self.data_args, 'negative_number', 15),
        )

        print(f"Finished hard negative mining. New negatives saved to {new_negatives_file}")

        # Create a new dataset with the mined negatives
        new_data_args = deepcopy(self.data_args)
        new_data_args.train_data = [new_negatives_file]

        tokenizer = kwargs['tokenizer']
        new_dataset = AbsEmbedderTrainDataset(args=new_data_args, tokenizer=tokenizer)

        # Update the trainer's dataset
        trainer = kwargs['trainer']
        trainer.train_dataset = new_dataset

        print("Trainer's dataset updated with new hard negatives.")
