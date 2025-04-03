import numpy as np
import pickle
import os

def create_faiss_index():
    # 检查FAISS是否已安装
    try:
        import faiss
    except ImportError:
        print("错误: 未安装FAISS库。请使用 pip install faiss-cpu 或 pip install faiss-gpu 安装")
        return
    
    # 加载嵌入字典
    print("加载嵌入数据...")
    try:
        with open('olympiads_embeddings.pkl', 'rb') as f:
            embedding_dict = pickle.load(f)
    except FileNotFoundError:
        print("未找到嵌入数据文件，请先运行generate_embeddings.py")
        return
    
    # 获取嵌入向量和文本
    embeddings = embedding_dict['embeddings']
    texts = embedding_dict['texts']
    
    print(f"加载了 {len(texts)} 个文本的嵌入向量")
    
    # 确保嵌入是float32类型（FAISS要求）
    if embeddings.dtype != np.float32:
        print(f"将嵌入从 {embeddings.dtype} 转换为 float32")
        embeddings = embeddings.astype(np.float32)
    
    # 创建FAISS索引
    print("创建FAISS索引...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 内积相似度（余弦相似度，因为向量已经归一化）
    index.add(embeddings)
    
    # 保存FAISS索引
    faiss.write_index(index, "olympiads_solution_index.faiss")
    print("FAISS索引已创建并保存为 olympiads_solution_index.faiss")
    
    # 保存文本到单独的文件以便后续搜索
    with open('olympiads_solution_texts.pkl', 'wb') as f:
        pickle.dump(texts, f)
    print("文本数据已保存为 olympiads_solution_texts.pkl")
    
    return index

if __name__ == "__main__":
    create_faiss_index() 