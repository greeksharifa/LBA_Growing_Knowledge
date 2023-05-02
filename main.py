import json
import pickle
import os

from transformers import AutoTokenizer, RobertaModel
from transformers import logging
import torch
import platform

from configs.load_configs import get_config_path
from get_feature.load_data import get_data
from get_feature.get_knowledge_from_sg import get_knowledge
from get_feature.get_phrase_from_question import get_phrase_of_question



def main():
    config_path = get_config_path()

    balanced_qa, scene_graphs, idx2eng = get_data('test')

    print('loading {:<25s}: {:<75s} ...'.format('grounded_knowledge', config_path['grounded_knowledge_path']), end='')
    grounded_knowledge = get_knowledge(qa=balanced_qa, scene_graphs=scene_graphs, idx2eng=idx2eng,
                                       save_path=config_path['grounded_knowledge_path'], overwrite=False, verbose=False)
    print('done. len is:', len(grounded_knowledge))

    print('loading {:<25s}: {:<75s} ...'.format('phrases_path', config_path['phrases_path']), end='')
    phrases = get_phrase_of_question(qa=balanced_qa,
                                     save_path=config_path['phrases_path'], overwrite=False, verbose=False)
    print('done. len is:', len(phrases))


if __name__ == "__main__":
    main()
