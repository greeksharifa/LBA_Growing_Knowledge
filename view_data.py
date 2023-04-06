import os
import pickle
from pprint import pprint


def load_scene_graphs(split: str = 'train'):
    """
    test key: 1814, dict_keys(['YSKX3', 'T5ECU', 'AAH6R', '015XE', ...])
    :param split:
    :return:
    """
    scene_graphs = pickle.load(open(f'D:/AGQA/AGQA_scene_graphs/AGQA_{split}_stsgs.pkl', mode='rb'))

    for i, (key, value) in enumerate(scene_graphs.items()):
        if i >= 2:
            break
        pprint(key)
        pprint(value)


if __name__ == '__main__':
    load_scene_graphs('test')
