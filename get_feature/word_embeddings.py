import json
import pickle

from transformers import AutoTokenizer, RobertaModel
from transformers import logging
import torch
import platform

logging.set_verbosity_error()
OS = platform.system()
ROOT_DIR = 'data/AGQA/' if OS == 'Windows' else '~/data/AGQA/'


tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

inputs = tokenizer("This is a sample input text.", return_tensors="pt")
print('inputs:', inputs)
outputs = model(**inputs)
# outputs.__dict__.keys()
# ['last_hidden_state', 'pooler_output', 'hidden_states', 'past_key_values', 'attentions', 'cross_attentions']

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape, '\n', last_hidden_states)


split = 'test'
balanced_qa = json.load(open(ROOT_DIR + f'AGQA2/AGQA_balanced/{split}_balanced.txt', mode='r', encoding='utf8'))
scene_graphs = pickle.load(open(ROOT_DIR + f'AGQA_scene_graphs/AGQA_{split}_stsgs.pkl', mode='rb'))
idx2eng = json.load(open(ROOT_DIR + 'ENG.txt', mode='r', encoding='utf8'))

for i, (question_id, q_value) in enumerate(balanced_qa.items()):
    if i > 5:
        break
    print('question_id:', question_id)

    word_list = []
    sg_grounding = q_value["sg_grounding"]
    for idxs, sg_key_list in sg_grounding.items():
        if len(sg_key_list) == 0:
            continue
        for sg_key in sg_key_list:
            verbs = scene_graphs[sg_key]['names']
            verbs = list(map(lambda x: idx2eng[x], verbs))
            word_list.extend(verbs)

    print('word_list:', word_list)
    print()