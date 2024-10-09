# source_paths
sqa_train_images_path = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/scienceQA/train'
sqa_problem_json = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/scienceQA/problems.json'

# arguments for saving
dir2save_img = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train'
path2save_json = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/text_files_small_dataset/scienceQA.json'

# templates
TEMPLATE = {
    "id": "{sample_id}",
    "image": "finetune_datasets/scienceQA/train/{sample_id}.png",
    "conversations": [
        {
            "from": "human",
            "value": '<image>\n{question} {choices}'
        },
        {
            "from": "gpt",
            "value": "{answer}"
        }
    ]
}

ANSWER_MAPS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

import shutil
import os
import json
from tqdm import tqdm

with open(sqa_problem_json, 'r') as file:
    train_data = json.load(file)


train_list = []
count_train_none_img = 0

os.makedirs(dir2save_img, exist_ok=True)
os.makedirs(os.path.dirname(path2save_json), exist_ok=True)

for idx, sample in tqdm(train_data.items()):
    img_file_name = sample['image']
    question = sample['question']
    answer_idx = sample['answer']
    choices = sample['choices']
    split = sample['split']

    if split != 'train':
        continue

    if img_file_name == None:
        count_train_none_img += 1
        continue

    sample_path = os.path.join(sqa_train_images_path, idx)
    img_path = os.path.join(sample_path, img_file_name)

    answer = ANSWER_MAPS[int(answer_idx)]

    choices_str = ''
    for i, c in enumerate(choices):
        choices_str += ANSWER_MAPS[i] + '.' + c + ', ' if i != len(choices)-1 else ANSWER_MAPS[i] + '.' + c
    sample_data = {
        'sample_id': idx,
        'question': question,
        'answer': answer,
        'choices': choices_str
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
                 os.path.join(dir2save_img, "finetune_datasets/scienceQA/train/"+idx+".png"))


with open(path2save_json, 'w') as outfile:
    json.dump(train_list, outfile, indent=4)

print(f"There are {count_train_none_img} samples without image in training set!")
print(f"{len(train_list)} training samples with image")
print('Preprocessing succeeds!')
