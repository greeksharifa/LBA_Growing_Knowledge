import os
import pickle
from pprint import pprint


def load_scene_graphs(split: str = 'train'):
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
    example)
    key(frame id) = 'YSKX3'
    value: dict{
        '000105'(=frame_num): {
            id        (<class 'str'>       ) : '000105',
            secs      (<class 'float'>     ) : 6.3,
            type      (<class 'str'>       ) : 'frame',
            metadata  (<class 'str'>       ) : 'test',
            objects   (<class 'dict'>      ) : {
                'names'(<class 'list'>     ) : ['o4'], (=blanket)
                'vertices'(<class 'list'>  ) : len 1 [
                    dict_keys(['id', 'type', 'class', 'attention', 'contact', 'spatial', 'verb', 'visible', 'bbox', 'metadata', 'frame_num', 'secs', 'next', 'prev'])
                ]
            }
            attention (<class 'dict'>      ) : similar to objects. names: ['r1']    'looking at'
            contact   (<class 'dict'>      ) : similar to objects. names: ['r22']   'touching'
            spatial   (<class 'dict'>      ) : similar to objects. names: ['r7']    'behind'
            verb      (<class 'dict'>      ) : similar to objects. names: ['v026']  'tidying'
            actions   (<class 'dict'>      ) : similar to objects. names: ['c075']  'tidying up a blanket'
            next      (<class 'dict'>      ) : similar to objects. names: ['r1']    'looking at'
            prev      (<class 'NoneType'> or 'dict'  ) : None
                dict: dict_keys(['id', 'type', 'class', 'objects', 'metadata', 'frame', 'secs', 'next', 'prev'])
        },
        '000134': {...},
        '000163': {...},
        ...,
        'c077/1': {...}, ...,
        'o4/000105': {...}, ...,
        'v025/000270': {...}
    }
    :param split: 'train' or 'test'
    :return: None
    """
    # scene_graphs = pickle.load(open(f'D:/AGQA/AGQA_scene_graphs/AGQA_{split}_stsgs.pkl', mode='rb'))
    # split = 'test'
    scene_graphs = pickle.load(open(f'~/data/AGQA/AGQA_scene_graphs/AGQA_{split}_stsgs.pkl', mode='rb'))
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
        # pprint(value)


if __name__ == '__main__':
    load_scene_graphs('test')
