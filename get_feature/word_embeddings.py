from transformers import AutoTokenizer, RobertaModel
from transformers import logging
logging.set_verbosity_error()
import torch


tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

inputs = tokenizer("This is a sample input text.", return_tensors="pt")
print('inputs:', inputs)
outputs = model(**inputs)
# outputs.__dict__.keys()
# dict_keys(['last_hidden_state', 'pooler_output', 'hidden_states',
# 'past_key_values', 'attentions', 'cross_attentions'])

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape, '\n', last_hidden_states)

