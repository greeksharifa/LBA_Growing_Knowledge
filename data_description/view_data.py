import pickle
from pprint import pprint
import os
import json
import platform

OS = platform.system()
ROOT_DIR = '~/data/AGQA/'
if OS == 'Windows':
    ROOT_DIR = 'data/'


def view_scene_graphs(split: str = 'train'):
    """
    from AGQA paper: Suppl 6.2 (page 13)
    Action Genome’s spatio-temporal scene graphs [19] annotate five sampled frames from each Charades action [47].
    Each frame contains object annotations with lists of the contact, spatial, and attention relationships
        between the subject and the object [19].
    We generate questions and answers based on these spatio-temporal scene graphs.
    Inaccurate or incomplete scene graph data can lead to uninformative and incorrect question generation.
    Since the scene graph annotations in Action Genome are often noisy, inconsistent, and sparse,
        we augment them using the following techniques to minimize errors:
        Duplication, Inconsistency, Sparsity, Entailments, Uncertainty in action localization.
    -----------------------------------------------------------------------------------------------
    from Action Genome paper: 2. Related Work (page 3)
    Action Genome’s annotation pipeline: For every
    action, we uniformly sample 5 frames across the action and
    annotate the person performing the action along with the
    objects they interact with. We also annotate the pairwise
    relationships between the person and those objects. Here,
    we show a video with 4 actions labelled, resulting in 20
    (= 4 × 5) frames annotated with scene graphs. The objects
    are grounded back in the video as bounding boxes.
    -----------------------------------------------------------------------------------------------
    test key: len=1814, dict_keys(['YSKX3', 'T5ECU', 'AAH6R', '015XE', ..., 'LSKA2'])

    scene graph의 각 key(=vid)마다 type별로 들어가 있는 정보가 다름.
    ex. scene_graphs['YSKX3']의 key로는 '000105', 'c077/1', 'o4/000105', 'r1/000105', 'v025/000270' 등이 있음.
    '000105'는 현 vid(video)에 포함되는 frame id,
    'o4/000105', 'r1/000105', 'v025/000270'는 해당 frame에 나타나는 object, relation(attention or verb),
    'c077/1'는 action을 의미함. '/' 뒤의 숫자는 현재 vid에서 몇 번째 'c077' action인지를 나타냄. '1': 15918, '2': 219, '3': 12개

        scene_graphs['YSKX3']['000105']['type']         : 'frame'
            ['id', 'secs', 'type', 'metadata', 'objects', 'attention', 'contact', 'spatial', 'verb', 'actions', 'next', 'prev']
        scene_graphs['YSKX3']['c077/1']['type']         : 'action'
            ['id', 'charades', 'phrase', 'type', 'start', 'end', 'length', 'objects', 'attention', 'contact', 'spatial',
            'verb', 'metadata', 'all_f', 'object_id', 'verb_id',
            'next_discrete', 'prev_discrete', 'next_instance', 'prev_instance', 'while']
        scene_graphs['YSKX3']['o4/000105']['type']      : 'object'
            ['id', 'type', 'class', 'attention', 'contact', 'spatial', 'verb', 'visible', 'bbox', 'metadata',
            'frame_num', 'secs', 'next', 'prev']
        scene_graphs['YSKX3']['r1/000105']['type']      : 'attention'
            ['id', 'type', 'class', 'objects', 'metadata', 'frame', 'secs', 'next', 'prev']
        scene_graphs['YSKX3']['v025/000270']['type']    : 'verb'
            ['id', 'type', 'class', 'objects', 'metadata', 'frame', 'secs', 'next', 'prev']


    type: key가 각각
        frame       = "숫자6자리"
        action      = "c.../X"
        object      = "o.../숫자6자리"
        attention   = "r.../숫자6자리"
        verb        = "v.../숫자6자리"

    example)
    key(frame id) = 'YSKX3'
    value: dict{
        '000105'(=frame_id): {
            id        (<class 'str'>       ) : '000105',
            secs      (<class 'float'>     ) : 6.3,
            type      (<class 'str'>       ) : 'frame',
            metadata  (<class 'str'>       ) : 'test',    test / train
            objects   (<class 'dict'>      ) : [connecting object nodes]
            {
                'names'(<class 'list'>     ) : ['o4'], (=blanket)
                'vertices'(<class 'list'>  ) : len 1 [
                    이 frame과 연관된 objects(type: dict)를 그냥 reference해서 넣어두었다...
                ]
            }
            attention (<class 'dict'>      ) : [connecting attention nodes].    names: ['r1']    'looking at'
            contact   (<class 'dict'>      ) : [connecting contact nodes].      names: ['r22']   'touching'
            spatial   (<class 'dict'>      ) : [connecting spatial nodes].      names: ['r7']    'behind'
            verb      (<class 'dict'>      ) : [connecting verb nodes].         names: ['v026']  'tidying'
            actions   (<class 'dict'>      ) : [connecting action nodes].       names: ['c075']  'tidying up a blanket'
            next      (<class 'dict'>      ) : next frame or None.              names: ['r1']    'looking at'
            prev      (<class 'dict'>      ) : previous frame or None.          None
        },
        '000134': {...},
        '000163': {...},
        ...,
        'c077/1': {...}, ...,                       # "c077": "putting a pillow somewhere"
        'o4/000105': {...}, ...,                    # "o4"  : "blanket"
        'r1/000105': {...}, ...,                    # "r1"  : "looking at"
        'v025/000270': {...}                        # "v025": "throwing",       "v026": "tidying",
    }
    :param split: 'train' or 'test'
    :return: None
    """
    # split = 'test'
    scene_graphs = pickle.load(open(ROOT_DIR + f'AGQA_scene_graphs/AGQA_{split}_stsgs.pkl', mode='rb'))
    key = next(iter(scene_graphs.keys()))
    value = next(iter(scene_graphs.values()))
    # print(type(value['000105']['id'])) # str
    # print(value['000105']['objects'].keys())

    for k in value['000105'].keys():
        print('{:<10s}({:<20s}) : '.format(k, str(type(value['000105'][k]))))

    for i, (key, value) in enumerate(scene_graphs.items()):
        if i > 0:
            break
        pprint(key)
        print(type(value))
        # pprint(value) circular reference 때문에 무조건 print함수의 memoryError로 터진다.

    from collections import Counter
    counter = Counter()
    for vid in scene_graphs.keys():
        for k in scene_graphs[vid].keys():
            if 'c' in k:
                counter[k[-1]] += 1
                # print(vid, k)
    print(counter)

def view_AGQA2_AGQA_balanced(split: str = 'train'):
    balanced_qa = json.load(open(ROOT_DIR + f'AGQA2/AGQA_balanced/{split}_balanced.txt', mode='r', encoding='utf8'))
    key = next(iter(balanced_qa.keys()))
    value = next(iter(balanced_qa.values()))
    print('key:', key)
    print('value:', value)


def view_AGQA2_CSV_formatted_questions_for_evaluation_csvs():
    """
    train : 1,600,894
    test  :   669,207
    total : 2,270,101
    
    이유는 모르겠지만 train, test, total 전부 description이 'none'임
    그리고 train은 id가 1441052+159842개가 나누어져 있음
    example:
            ,key,       question,                                                   answer,vid_id,gif_name,description
    0,00607-10552,Which object were they tidying before taking the thing they put down from somewhere?,table,00607,00607,none
    1,00607-11002,Did the person interact with the thing they held before or after tidying up the first thing they went in front of?,after,00607,00607,none
    ...
    669206,ZZ4GP-1266,What was the person tidying before holding the thing they took?,floor,ZZ4GP,ZZ4GP,none

    :return: None
    """


if __name__ == '__main__':
    # view_scene_graphs('test')
    view_AGQA2_AGQA_balanced('test')
