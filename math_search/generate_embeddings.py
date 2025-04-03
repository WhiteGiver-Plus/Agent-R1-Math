import torch
import torch.nn.functional as F
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from tqdm import tqdm

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_embeddings():
    # 加载预处理数据
    print("加载预处理的数据...")
    try:
        with open('olympiads_combined_data.pkl', 'rb') as f:
            combined_solutions = pickle.load(f)
    except FileNotFoundError:
        print("未找到预处理数据文件，请先运行prepare_data.py")
        return
    
    print(f"加载了 {len(combined_solutions)} 个解决方案")
    
    # 加载模型和分词器
    print("加载多语言E5模型...")
    tokenizer = AutoTokenizer.from_pretrained('e5_large')
    model = AutoModel.from_pretrained('e5_large')
    
    # 将模型移至GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")

    # 创建嵌入
    print("创建文本嵌入...")
    embeddings_list = []

    # 使用批处理来处理大量文本
    batch_size = 1024
    for i in tqdm(range(0, len(combined_solutions), batch_size)):
        batch_texts = combined_solutions[i:i+batch_size]
        
        batch_dict = tokenizer(batch_texts, max_length=512, padding=True, 
                              truncation=True, return_tensors='pt')
        
        # 将输入移至GPU
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        with torch.no_grad():
            outputs = model(**batch_dict)
        
        batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        
        # 转移回CPU并转为half精度以节省内存
        embeddings_list.append(batch_embeddings.cpu().half().numpy())

    # 合并所有嵌入
    all_embeddings = np.vstack(embeddings_list)
    print(f"创建了 {all_embeddings.shape[0]} 个嵌入向量，每个维度为 {all_embeddings.shape[1]}")
    
    # 创建字典存储嵌入和原始文本
    embedding_dict = {
        'embeddings': all_embeddings,
        'texts': combined_solutions
    }
    
    # 保存字典
    with open('olympiads_embeddings.pkl', 'wb') as f:
        pickle.dump(embedding_dict, f)
    
    print("嵌入字典已保存为 olympiads_embeddings.pkl")
    return embedding_dict

if __name__ == "__main__":
    generate_embeddings() 