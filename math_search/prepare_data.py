import pickle
from datasets import load_dataset
from tqdm import tqdm

def prepare_olympiads_data():
    # 加载NuminaMath数据集
    print("正在加载NuminaMath-1.5数据集...")
    dataset = load_dataset("aimo")

    # 过滤出source为olympiads的数据
    olympiads_data = dataset["train"].filter(lambda x: x["source"] == "olympiads")
    print(f"找到 {len(olympiads_data)} 条奥林匹克数据")

    # 过滤出problem_is_valid和solution_is_valid均为True的数据
    valid_data = olympiads_data.filter(lambda x: x["problem_is_valid"] == "Yes" and x["solution_is_valid"] == "Yes" )
    print(f"其中有效数据（problem_is_valid和solution_is_valid均为True）: {len(valid_data)} 条")

    # 合并problem和solution字段，并过滤掉总长度超过3500的数据
    combined_data = []
    for item in tqdm(valid_data):
        # 处理problem
        problem_text = item["problem"]
        solution_text = item["solution"]
        
        # 合并并检查长度
        combined_text = f"Problem:\n{problem_text}\nSolution:\n{solution_text}"
        if len(combined_text) < 3500:
            combined_data.append(combined_text)

    print(f"处理了 {len(combined_data)} 个符合条件的数据（长度小于3500）")
    
    # 保存处理好的数据
    with open('olympiads_combined_data.pkl', 'wb') as f:
        pickle.dump(combined_data, f)
    
    print("预处理数据已保存到 olympiads_combined_data.pkl")
    return combined_data

if __name__ == "__main__":
    prepare_olympiads_data() 