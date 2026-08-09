import json
import os
import random
import time

SOURCE_FILE_PATH = './data/dolly_15k.txt'
OUTPUT_FOLDER_PATH = './data/exp3'
ITEMS_PER_CATEGORY = 20

def sample_and_export_prompts(input_file, output_folder, count_per_cat=20):
    # Seed the random number generator using high-precision time for dynamic randomness
    random.seed(time.time_ns())
    
    target_categories = {
        "closed_qa",
        "open_qa",
        "classification",
        "information_extraction",
        "summarization"
    }
    
    # Storage for filtered items grouped by category
    category_pools = {cat: [] for cat in target_categories}
    
    # Read line-by-line JSON dataset
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                cat = item.get("category")
                if cat in category_pools:
                    category_pools[cat].append(item)
            except json.JSONDecodeError:
                continue

    os.makedirs(output_folder, exist_ok=True)
    
    id_counter = 1

    # Randomly select and write prompts to separate JSON files
    for cat in target_categories:
        pool = category_pools[cat]
        sample_count = min(len(pool), count_per_cat)
        selected_samples = random.sample(pool, sample_count)
        
        for sample in selected_samples:
            prompt_id = f"UP{id_counter:03d}"
            
            output_data = {
                "prompt_id": prompt_id,
                "task_category": sample.get("category", cat),
                "instruction": sample.get("instruction", ""),
                "context": sample.get("context", "")
            }
            
            file_path = os.path.join(output_folder, f"{prompt_id}.json")
            with open(file_path, "w", encoding="utf-8") as out_f:
                json.dump(output_data, out_f, indent=2, ensure_ascii=False)
                
            id_counter += 1

    print(f"Successfully processed and generated {id_counter - 1} files in directory '{output_folder}'.")

if __name__ == "__main__":
    # Replace 'dataset.json' with your input dataset file path
    sample_and_export_prompts(SOURCE_FILE_PATH, OUTPUT_FOLDER_PATH, ITEMS_PER_CATEGORY)