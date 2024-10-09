# source_paths
iconqa_data_root_path = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/iconQA/iconqa_data'

# arguments for saving
dir2save_img = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train'
path2save_json = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/text_files_small_dataset/iconQA_blank.json'

# templates
TEMPLATE = {
    "id": "{sample_id}",
    "image": "finetune_datasets/iconQA_blank/train/{sample_id}.png",
    "conversations": [
        {
            "from": "human",
            "value": '<image>\n{question} Fill in the blanks in (_) or answer this question'
        },
        {
            "from": "gpt",
            "value": "{answer}"
        }
    ]
}


import shutil
import os
import json
from tqdm import tqdm

pid_splits_json = os.path.join(iconqa_data_root_path, 'pid_splits.json')
train_root = os.path.join(iconqa_data_root_path, 'iconqa/train')
val_root = os.path.join(iconqa_data_root_path, 'iconqa/val')
test_root = os.path.join(iconqa_data_root_path, 'iconqa/test')

train_blank_root = os.path.join(train_root, 'fill_in_blank')

train_dir_names = os.listdir(train_blank_root)

train_list = []

os.makedirs(dir2save_img, exist_ok=True)
os.makedirs(os.path.dirname(path2save_json), exist_ok=True)

for dir_name in tqdm(train_dir_names):
    sample_path = os.path.join(train_blank_root, dir_name)
    sample_json = os.path.join(sample_path, 'data.json')
    img_path = os.path.join(sample_path, 'image.png')

    with open(sample_json, 'r') as file:
        sample_data = json.load(file)
    question = sample_data['question']
    answer = sample_data['answer']

    sample_data = {
        'sample_id': dir_name,
        'question': question,
        'answer': answer,
    }

    formatted_sample = {
        key: (value.format(**sample_data)
            if isinstance(value, str)
            else [{k: v.format(**sample_data)
                for k, v in item.items()} for item in value])
        for key, value in TEMPLATE.items()
    }

    train_list.append(formatted_sample)

    shutil.copy2(img_path,
                 os.path.join(dir2save_img, "finetune_datasets/iconQA_blank/train/"+dir_name+".png"))


with open(path2save_json, 'w') as outfile:
    json.dump(train_list, outfile, indent=4)

print('Preprocessing succeeds!')

