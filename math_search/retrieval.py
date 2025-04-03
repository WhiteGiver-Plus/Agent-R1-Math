import torch
import torch.nn.functional as F
import numpy as np
import pickle
import faiss
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from typing import List, Tuple

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

# 全局变量存储加载的资源
GLOBAL_INDEX = None
GLOBAL_TEXTS = None
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_DEVICE = None

def initialize_resources(device='cuda'):
    """初始化全局资源"""
    global GLOBAL_INDEX, GLOBAL_TEXTS, GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_DEVICE
    GLOBAL_INDEX, GLOBAL_TEXTS, GLOBAL_MODEL, GLOBAL_TOKENIZER = load_resources(device)
    GLOBAL_DEVICE = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    return GLOBAL_INDEX is not None and GLOBAL_TEXTS is not None and GLOBAL_MODEL is not None and GLOBAL_TOKENIZER is not None

def get_detailed_instruct(task_description: str, query: str) -> str:
    """为查询添加任务指令"""
    return f'Instruct: {task_description}\nQuery: {query}'

def encode_query(queries: List[str]):
    """使用与训练相同的模型编码查询(支持批量)"""
    global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_DEVICE
    
    # 处理查询 - 添加指令格式
    task_description = "Given a math search query, retrieve relevant math proof relevant to the query"
    input_texts = [get_detailed_instruct(task_description, query) for query in queries]
    
    # 分词和编码
    inputs = GLOBAL_TOKENIZER(input_texts, max_length=512, padding=True, 
                       truncation=True, return_tensors='pt')
    inputs = {k: v.to(GLOBAL_DEVICE) for k, v in inputs.items()}
    
    # 获取嵌入
    with torch.no_grad():
        outputs = GLOBAL_MODEL(**inputs)
    
    # 池化和归一化
    query_embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    
    # 转换为numpy数组
    return query_embeddings.cpu().numpy().astype(np.float32)

def retrieve(queries: List[str], num: int = 5) -> List[List[Tuple[str, float]]]:
    """
    检索与多个查询最相似的条目
    
    参数:
        queries: 查询文本列表
        num: 每个查询要返回的最相似条目数量
        
    返回:
        包含每个查询的结果列表，每个结果为(文本, 相似度分数)元组的列表
    """
    global GLOBAL_INDEX, GLOBAL_TEXTS
    
    # 如果输入是单个字符串，转换为列表
    if isinstance(queries, str):
        queries = [queries]
    
    # 检查资源是否已加载
    if GLOBAL_INDEX is None or GLOBAL_TEXTS is None:
        print("资源未初始化，请先调用initialize_resources()")
        return []
    
    # 编码查询
    print(f"编码 {len(queries)} 个查询")
    query_embeddings = encode_query(queries)
    
    # 搜索最相似的条目
    print(f"为每个查询搜索最相似的 {num} 个条目...")
    scores, indices = GLOBAL_INDEX.search(query_embeddings, num)
    
    # 整理结果
    all_results = []
    for i in range(len(queries)):
        results = []
        for j in range(num):
            idx = indices[i][j]
            score = scores[i][j]
            if idx < len(GLOBAL_TEXTS):
                results.append((GLOBAL_TEXTS[idx], float(score)))
        all_results.append(results)
    
    return all_results

def display_results(queries: List[str], all_results: List[List[Tuple[str, float]]]):
    """格式化显示多个查询的检索结果"""
    # 如果输入是单个字符串，转换为列表
    if isinstance(queries, str):
        queries = [queries]
        
    for q_idx, (query, results) in enumerate(zip(queries, all_results)):
        print(f"\n===== 查询 {q_idx+1}: '{query}' =====")
        for i, (text, score) in enumerate(results, 1):
            # 裁剪文本，以便显示不会太长
            max_display_length = 500
            displayed_text = text[:max_display_length] + "..." if len(text) > max_display_length else text
            print(f"\n结果 #{i} (相似度: {score:.4f}):")
            print("-" * 80)
            print(displayed_text)
            print("-" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="检索与查询相似的数学问题解决方案")
    parser.add_argument("queries", type=str, nargs='+', help="查询文本(一个或多个)")
    parser.add_argument("--num", type=int, default=3, help="每个查询要返回的结果数量（默认为3）")
    
    args = parser.parse_args()
    
    all_results = retrieve(args.queries, args.num)
    display_results(args.queries, all_results)
