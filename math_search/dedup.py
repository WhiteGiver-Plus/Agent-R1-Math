import faiss
import pickle
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import random
from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

# Global variables to store loaded resources
INDEX = None
TEXTS = None
MODEL = None
TOKENIZER = None
DEVICE = None

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def load_resources(device='cuda'):
    """加载FAISS索引、文本数据和模型"""
    print("加载FAISS索引...")
    try:
        index = faiss.read_index("olympiads_solution_index.faiss")
    except RuntimeError:
        print("未找到FAISS索引文件，请先运行create_index.py")
        return None, None, None, None
    
    print("加载文本数据...")
    try:
        with open('olympiads_solution_texts.pkl', 'rb') as f:
            texts = pickle.load(f)
    except FileNotFoundError:
        print("未找到文本数据文件，请先运行create_index.py")
        return index, None, None, None
    
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained('e5_large')
    model = AutoModel.from_pretrained('e5_large')
    
    # 检查设备可用性
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    device = torch.device(device)
    model = model.to(device)
    
    return index, texts, model, tokenizer

def initialize_resources(device='cuda'):
    """初始化全局资源"""
    global INDEX, TEXTS, MODEL, TOKENIZER, DEVICE
    INDEX, TEXTS, MODEL, TOKENIZER = load_resources(device)
    DEVICE = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    return INDEX is not None and TEXTS is not None and MODEL is not None and TOKENIZER is not None

def get_detailed_instruct(task_description: str, query: str) -> str:
    """为查询添加任务指令"""
    return f'Instruct: {task_description}\nQuery: {query}'

def encode_query(queries: List[str]):
    """使用与训练相同的模型编码查询(支持批量)"""
    global MODEL, TOKENIZER, DEVICE
    
    # 处理查询 - 添加指令格式
    task_description = "Given a math search query, retrieve relevant math proof relevant to the query"
    input_texts = [get_detailed_instruct(task_description, query) for query in queries]
    
    # 分词和编码
    inputs = TOKENIZER(input_texts, max_length=512, padding=True, 
                       truncation=True, return_tensors='pt')
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # 获取嵌入
    with torch.no_grad():
        outputs = MODEL(**inputs)
    
    # 池化和归一化
    query_embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    
    # 转换为numpy数组
    return query_embeddings.cpu().numpy().astype(np.float32)

def remove_duplicates_from_external(external_dataset_name, prompt_key='prompt', 
                                   similarity_threshold=0.9, top_k=6, 
                                   batch_size=2000, num_examples=5):
    """
    移除与外部数据集(GAIA)中的prompt相似的条目
    
    参数:
        external_dataset_name: HuggingFace数据集名称
        prompt_key: 数据集中提示文本的键名
        similarity_threshold: 相似度阈值，高于此值的条目被视为重复
        top_k: 为每个条目检查的最相似条目数量
        batch_size: 批处理大小
        num_examples: 要显示的查询示例数量
    
    返回:
        去重后的文本列表和对应的嵌入
    """
    print("初始化检索资源...")
    if not initialize_resources():
        print("初始化资源失败，退出")
        return None, None
    
    # 确保全局索引已正确初始化
    if INDEX is None:
        print("错误: 全局索引未能正确初始化")
        return None, None
    
    # 加载嵌入
    print("加载嵌入数据...")
    try:
        with open('olympiads_embeddings.pkl', 'rb') as f:
            embedding_dict = pickle.load(f)
            embeddings = embedding_dict['embeddings']
            texts = embedding_dict['texts']
    except FileNotFoundError:
        print("未找到嵌入数据文件，请先运行generate_embeddings.py")
        return None, None
    
    # 加载GAIA数据集
    print(f"加载数据集: {external_dataset_name}...")
    try:
        external_dataset = load_dataset(external_dataset_name)
        # 通常使用'train'分割，但根据实际情况可能需要调整
        external_prompts = []
        for split in external_dataset:
            if prompt_key in external_dataset[split].features:
                external_prompts.extend([item[prompt_key] for item in external_dataset[split]])
        
        if not external_prompts:
            raise KeyError(f"数据集中没有找到'{prompt_key}'字段")
            
        print(f"加载了 {len(external_prompts)} 个外部提示")
    except Exception as e:
        print(f"加载外部数据集失败: {e}")
        return None, None
    
    print(f"原始数据集包含 {len(texts)} 个条目")
    
    # 标记要移除的条目
    to_remove = set()
    
    # 展示一些查询示例
    example_indices = random.sample(range(len(external_prompts)), min(num_examples, len(external_prompts)))
    print("\n查询示例:")
    for i in example_indices:
        print(f"查询 {i+1}: {external_prompts[i][:100]}...")
    print()
    
    # 批处理查询
    print("查找与外部数据集相似的条目...")
    for i in tqdm(range(0, len(external_prompts), batch_size)):
        batch_prompts = external_prompts[i:i+batch_size]
        
        # 构建查询
        queries = [f"@math_search {prompt}" for prompt in batch_prompts]
        
        # 批量编码查询
        query_embeddings = encode_query(queries)
        
        # 确保查询嵌入不为空
        if query_embeddings is None or len(query_embeddings) == 0:
            print(f"警告: 批次 {i//batch_size + 1} 的查询嵌入为空，跳过此批次")
            continue
            
        # 批量搜索相似条目
        try:
            scores, indices = INDEX.search(query_embeddings, top_k)
        except Exception as e:
            print(f"搜索过程中出错: {e}")
            return None, None
        
        # 处理每个查询的结果
        for j in range(len(batch_prompts)):
            batch_indices = indices[j]
            batch_scores = scores[j]
            
            # 显示随机选定的示例的结果
            if i+j in example_indices:
                print(f"\n查询示例结果 {i+j+1}:")
                print(f"原始查询: {batch_prompts[j][:100]}...")
                for k in range(len(batch_indices)):
                    idx = batch_indices[k]
                    score = batch_scores[k]
                    if idx < len(texts):
                        print(f"  相似条目 {k+1} (相似度: {score:.4f}): {texts[idx][:100]}...")
            
            # 移除相似度过高的条目
            for k in range(len(batch_indices)):
                idx = batch_indices[k]
                score = batch_scores[k]
                
                if idx < len(texts) and score > similarity_threshold:
                    to_remove.add(idx)
    
    # 创建去重后的数据
    remove_indices = sorted(list(to_remove))
    print(f"\n找到 {len(remove_indices)} 个需要删除的条目")
    
    # 创建保留索引的掩码
    keep_mask = np.ones(len(texts), dtype=bool)
    keep_mask[remove_indices] = False
    
    # 过滤文本和嵌入
    filtered_texts = [texts[i] for i in range(len(texts)) if keep_mask[i]]
    filtered_embeddings = embeddings[keep_mask]
    
    print(f"去重后数据集包含 {len(filtered_texts)} 个条目")
    
    # 保存去重后的数据
    new_embedding_dict = {
        'embeddings': filtered_embeddings,
        'texts': filtered_texts
    }
    
    with open('olympiads_embeddings_deduped.pkl', 'wb') as f:
        pickle.dump(new_embedding_dict, f)
    print("去重后的嵌入已保存为 olympiads_embeddings_deduped.pkl")
    
    # 展示一些被删除的条目示例
    if remove_indices:
        sample_size = min(5, len(remove_indices))
        sample_indices = random.sample(remove_indices, sample_size)
        print("\n删除条目示例:")
        for idx in sample_indices:
            print(f"  - {texts[idx][:200]}...")
    
    return filtered_texts, filtered_embeddings

def main():
    """去重主函数"""
    print("开始执行数据去重...")
    # 请替换为实际的GAIA数据集名称，例如 "OpenLemur/gaia"
    external_dataset_name = "LIMR"
    
    # 根据实际的数据集结构调整prompt_key参数
    filtered_texts, filtered_embeddings = remove_duplicates_from_external(
        external_dataset_name,
        prompt_key='prompt',  # 根据实际数据结构调整
        batch_size=1024,
        num_examples=5
    )
    
    if filtered_texts is not None and filtered_embeddings is not None:
        print("数据去重完成!")
        
        # 重新创建FAISS索引
        try:
            print("为去重数据创建新的FAISS索引...")
            dimension = filtered_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(filtered_embeddings)
            
            # 保存FAISS索引
            faiss.write_index(index, "olympiads_solution_index_deduped.faiss")
            print("新的FAISS索引已创建并保存为 olympiads_solution_index_deduped.faiss")
            
            # 保存文本到单独的文件以便后续搜索
            with open('olympiads_solution_texts_deduped.pkl', 'wb') as f:
                pickle.dump(filtered_texts, f)
            print("去重后的文本数据已保存为 olympiads_solution_texts_deduped.pkl")
        except ImportError:
            print("未安装FAISS库，跳过索引创建")

if __name__ == "__main__":
    main()
