from tqdm import tqdm
import json

PROMPT_TEXTCAPS = "Provide a one-sentence caption for the provided image"

orig_path = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/text_files/llava_v1_5_mix665k.json'
dest_path = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/text_files_small_dataset/text_caps.json'


with open(orig_path, 'r') as file:
    data = json.load(file)

dict_new = []

for sample in tqdm(data):
    if 'image' in sample and 'textvqa' in sample['image']:
        dict_new.append(sample)

print(f"The new dataset includes {len(dict_new)} samples.") 

# save as json
with open(dest_path, 'w') as json_file:
    json.dump(dict_new, json_file, indent=4)