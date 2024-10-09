okvqa_train_question_json = "/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/okvqa/OpenEnded_mscoco_train2014_questions.json"
okvqa_train_annotation_json = "/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/okvqa/mscoco_train2014_annotations.json"

path2save_json = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/text_files_small_dataset/okvqa.json'

import json, random
from tqdm import tqdm

TEMPLATE = {
    "id": "{question_id}",
    "image": "coco/train2017/{image_id}.jpg",
    "conversations": [
        {
            "from": "human",
            "value": '<image>\n{question}'
        },
        {
            "from": "gpt",
            "value": "{answer}"
        }
    ]
}


with open(okvqa_train_annotation_json, 'r') as file:
    train_annotations = json.load(file)['annotations']

with open(okvqa_train_question_json, 'r') as file:
    train_questions = json.load(file)['questions']


new_train_annotations = {}
for anno in train_annotations:
    new_train_annotations[anno['question_id']] = anno

train_list = []

for q in tqdm(train_questions):
    question = q['question']
    image_id = q['image_id']
    question_id = q['question_id']
    
    train_annotation = new_train_annotations.get(question_id)
    assert image_id == train_annotation['image_id']
    answers = train_annotation['answers']

    answer = random.choices(answers, k=1)[0]['answer']

    sample_data = {
        'question_id': question_id,
        'question': question,
        'answer': answer,
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

print('Preprocessing succeeds!')
