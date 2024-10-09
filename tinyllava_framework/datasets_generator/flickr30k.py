# source_paths
flickr30k_images_path = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/flickr30k/flickr30k-images'
flickr_annotations_30k_csv = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/flickr30k/flickr_annotations_30k.csv'

# arguments for preprocessing
select_which_cap = 'random' # 0, 1, 2, 3, 4, random
select_which_ins_temp = 'random' # 0, 1, 2, 3, 4, 5, 6, 7, 8, random

# arguments for saving
dir2save_img = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train'
path2save_json = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/text_files_small_dataset/flockr30k.json'

# templates
INSTRUCTION_TEMPLATE = [
    "<image>\nShare a concise interpretation of the image provided.",
    "<image>\nRender a clear and concise summary of the photo.",
    "<image>\nWrite a terse but informative summary of the picture.",
    "<image>\nOffer a succinct explanation of the picture presented.",
    "<image>\nDescribe the image concisely.",
    "<image>\nProvide a brief description of the given image.",
    "<image>\nCreate a compact narrative representing the image presented.",
    "<image>\nRelay a brief, clear account of the picture shown.",
    "<image>\nSummarize the visual content of the image."
]

TEMPLATE = {
    "id": "{img_id}",
    "image": "finetune_datasets/flockr30k/train/{filename}",
    "conversations": [
        {
            "from": "human",
            "value": "{instruct_template}"
        },
        {
            "from": "gpt",
            "value": "{cap}"
        }
    ]
}

import pandas as pd
import random
import shutil
import os
import json
from tqdm import tqdm

df = pd.read_csv(flickr_annotations_30k_csv)

train_list = []
val_list = []
test_list = []

os.makedirs(dir2save_img, exist_ok=True)
os.makedirs(os.path.dirname(path2save_json), exist_ok=True)

for index, row in tqdm(df.iterrows()):
    raw = json.loads(row['raw'])
    sentids = row['sentids']
    split = row['split']
    filename = row['filename']
    img_id = row['img_id']

    if select_which_cap == 'random':
        id2select = random.choice([0, 1, 2, 3, 4])
    else:
        id2select = int(select_which_cap)
        assert id2select in [0, 1, 2, 3, 4], "There're only 5 captions to select!"

    if select_which_ins_temp == 'random':
        temp2select = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
    else:
        temp2select = int(select_which_ins_temp)
        assert temp2select in [0, 1, 2, 3, 4, 5, 6, 7, 8], "There're only 9 instruction templates to select!"

    sample_data = {
        'img_id': img_id,
        'filename': filename,
        'instruct_template': INSTRUCTION_TEMPLATE[temp2select],
        'cap': raw[id2select]
    }

    formatted_sample = {
        key: (value.format(**sample_data)
            if isinstance(value, str)
            else [{k: v.format(**sample_data)
                for k, v in item.items()} for item in value])
        for key, value in TEMPLATE.items()
    }

    # formatted_sample = {key: value.format(**sample_data) for key, value in TEMPLATE.items()}
    train_list.append(formatted_sample) if split == 'train' else None
    val_list.append(formatted_sample) if split == 'val' else None
    test_list.append(formatted_sample) if split == 'test' else None

    if split == 'train':
        shutil.copy2(os.path.join(flickr30k_images_path, filename),
                     os.path.join(dir2save_img, "finetune_datasets/flockr30k/train/"+filename))


with open(path2save_json, 'w') as outfile:
    json.dump(train_list, outfile, indent=4)

print(f"The size of training set is {len(train_list)}!")
print('Preprocessing succeeds!')
