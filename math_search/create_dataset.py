# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GAIR/LIMR dataset to parquet format
"""

import os
import datasets
# from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/limr')

    args = parser.parse_args()

    data_source = 'GAIR/LIMR'

    dataset = datasets.load_dataset(data_source)

    instruction_following = """You can use the tools provided to you to answer the question. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
When you have the final answer, you can output the final answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
"""

    # Process each split in the dataset
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.get('question', '')
            question = question_raw + ' ' + instruction_following
            
            answer_raw = example.get('answer', '')
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_raw
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'id': example.get('id', f"{split}_{idx}"),
                    'question': question_raw,
                }
            }
            return data

        return process_fn

    # Process each split available in the dataset
    processed_datasets = {}
    for split in dataset.keys():
        processed_datasets[split] = dataset[split].map(
            function=make_map_fn(split), 
            with_indices=True
        )
        
        # Optional: Select a subset for testing
        if split == 'test' or split == 'validation':
            processed_datasets[split] = processed_datasets[split].select(range(min(100, len(processed_datasets[split]))))

    # Save to parquet
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    for split, processed_dataset in processed_datasets.items():
        processed_dataset.to_parquet(os.path.join(local_dir, f'{split}.parquet'))

