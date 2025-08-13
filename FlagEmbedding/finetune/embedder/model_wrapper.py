# In model_wrapper.py

from tqdm import tqdm
import torch
import numpy as np
from transformers import TrainingArguments

def create_encoding_model_wrapper(model, tokenizer, max_length, training_args: TrainingArguments):
    """
    创建一个设备感知的模型包装器。
    它会使用 training_args.device 来确保编码在正确的设备 (GPU/NPU) 上执行。
    """
    # 1. 获取目标设备，这是唯一可靠的来源
    target_device = training_args.device
    print(f"模型包装器将使用设备: {target_device} 进行编码")

    # 2. 将模型移动到目标设备，并设置为评估模式
    #    这对于初始索引构建至关重要，对于epoch结束时的调用也是安全的（重复移动无害）。
    model.to(target_device)
    model.eval()

    def encode_wrapper(sentences: list, batch_size: int = 256):
        all_embeddings = []
        # 禁用梯度计算以节省内存和加速
        with torch.no_grad():
            for start_index in tqdm(range(0, len(sentences), batch_size), desc="Encoding Sentences for Index"):
                sentences_batch = sentences[start_index:start_index + batch_size]
                # 3. 将输入数据也移动到目标设备
                inputs = tokenizer(
                    sentences_batch, padding=True, truncation=True, return_tensors='pt', max_length=max_length
                ).to(target_device)
                
                # 4. 在目标设备上执行编码，然后将结果移回CPU转为numpy
                embeddings = model.encode(inputs).cpu().numpy()
                all_embeddings.append(embeddings)
        
        # 5. [可选但推荐] 将模型移回CPU，以释放GPU显存给后续的训练。
        #    Trainer在开始训练时会再次把它移到GPU，所以这是个好习惯。
        #model.cpu()

        return np.concatenate(all_embeddings, axis=0)
    
    class ModelWrapper:
        def encode(self, sentences, **kwargs):
            return encode_wrapper(sentences)
        def encode_queries(self, queries, **kwargs):
            return encode_wrapper(queries)

    return ModelWrapper()