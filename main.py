import json
import pickle
import os
from pprint import pprint

from transformers import AutoTokenizer, RobertaModel
from transformers import logging
import torch
import platform

from configs.load_configs import get_config_path
from get_feature.get_attention_score import get_attention_score
from get_feature.load_data import get_data
from get_feature.get_knowledge_from_sg import get_knowledge
from get_feature.get_phrase_from_question import get_phrase_of_question



def main():
    print('torch.cuda.is_available():', torch.cuda.is_available())
    config_path = get_config_path()

    balanced_qa, scene_graphs, idx2eng = get_data('test')

    grounded_knowledge = get_knowledge(qa=balanced_qa, scene_graphs=scene_graphs, idx2eng=idx2eng,
                                       save_path=config_path['grounded_knowledge_path'], overwrite=False, verbose=False)

    phrases = get_phrase_of_question(qa=balanced_qa,
                                     save_path=config_path['phrases_path'], overwrite=True, verbose=True)

    for i, question_id in enumerate(balanced_qa.keys()):
        if i>=5:
            break
        knowledge_sentences = grounded_knowledge[question_id]
        # phrases_sentences = phrases[question_id]
        phrases_sentences = [t[1] for t in phrases[question_id]]
        print('\n\nquestion_id         :', question_id)
        print('knowledge_sentences :')
        pprint(knowledge_sentences)
        print('phrases_sentences   :')
        pprint(phrases_sentences)
        attention_score = get_attention_score(knowledge_sentences=knowledge_sentences, phrases_sentences=phrases_sentences)
        print('attention_score:')
        print(attention_score)
        attention_score_average = torch.mean(attention_score, dim=0)
        print(attention_score_average, '<- average by column')
        uncertain_phrase_idx = torch.argmin(attention_score_average)
        print('uncertain_phrase_idx:', uncertain_phrase_idx)
        uncertain_phrases = phrases_sentences[uncertain_phrase_idx]
        print('uncertain_phrases:', uncertain_phrases)

        # TODO:
        # ChatCaptioner/Video_ChatCaptioner/generate_caption_msvd.py 을 수정해서
        # prompt 수정: uncertainty를 반영하게
        # uniform frame sampling 대신 question에 grounding된 scene graph에 포함된 frame만 선택하여
        # 해당 frame들에 대해 질문 생성하기





if __name__ == "__main__":
    main()
