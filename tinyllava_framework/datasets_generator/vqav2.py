vqav2_train_annotation_json = "/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/vqav2/v2_mscoco_train2014_annotations.json"
vqav2_train_question_json = "/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/vqav2/v2_OpenEnded_mscoco_train2014_questions.json"

path2save_json = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/text_files_small_dataset/vqav2_20k.json'

NUM_TRAINING_SET = 20000

import json, random
from tqdm import tqdm

TEMPLATE = {
    "id": "{question_id}",
    "image": "coco/train2017/{image_id}.jpg",
    "conversations": [
        {
            "from": "human",
            "value": '<image>\n{question} Answer the question using a single word or phrase.'
        },
        {
            "from": "gpt",
            "value": "{answer}"
        }
    ]
}


with open(vqav2_train_annotation_json, 'r') as file:
    train_annotations = json.load(file)['annotations']

with open(vqav2_train_question_json, 'r') as file:
    train_questions = json.load(file)['questions']
assert NUM_TRAINING_SET <= len(train_questions), "The size of randomly sampled set cannot be larger than the original one."
sampled_questions = random.sample(train_questions, NUM_TRAINING_SET)

new_train_annotations = {}
for anno in train_annotations:
    new_train_annotations[anno['question_id']] = anno

train_list = []

for q in tqdm(sampled_questions):
    question = q['question']
    image_id = q['image_id']
    question_id = q['question_id']
    
    train_annotation = new_train_annotations.get(question_id)
    assert image_id == train_annotation['image_id']
    answers = train_annotation['answers']
    multiple_choice_answer = train_annotation['multiple_choice_answer']

    sample_data = {
        'question_id': question_id,
        'question': question,
        'answer': multiple_choice_answer,
        'image_id': str(image_id).zfill(12)
    }

    formatted_sample = {
        key: (value.format(**sample_data)
            if isinstance(value, str)
            else [{k: v.format(**sample_data)
                for k, v in item.items()} for item in value])
        for key, value in TEMPLATE.items()
    }

    train_list.append(formatted_sample)

with open(path2save_json, 'w') as outfile:
    json.dump(train_list, outfile, indent=4)

print(f'The training set has totally {len(train_list)} samples.')
print('Preprocessing succeeds!')