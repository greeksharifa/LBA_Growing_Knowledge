import os
from typing import List

import cv2
from PIL import Image


def get_grounded_frame(scene_graph: dict, qa: dict, charades_frame_path: str,
                       question_id: str, uncertain_phrases: List[((int, int), str)],
                       only_uncertain_frame: bool = False,
                       verbose:bool=False) -> dict:
    """
    This function returns the grounded frame for a given question id.
    :param scene_graph:
    :param qa:
    :param charades_frame_path:
    :param question_id:
    :param uncertain_phrases:
    :param only_uncertain_frame:
    :param verbose:
    :return:
    """
    question_data = qa[question_id]
    video_id = question_data['video_id']
    sg_grounding = question_data['sg_grounding']
    frame_id_list = []
    if only_uncertain_frame:
        pass
    else:
        for value in sg_grounding.values():
            for v in value:
                if len(v) == 6 and '/' not in v:
                    frame_id_list.append(v)

    result = dict()
    for frame_id in frame_id_list:
        frame_path = os.path.join(charades_frame_path, video_id, f'{video_id}-{frame_id}.jpg')
        # frame = cv2.imread(frame_path)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_image = Image.open(frame_path).convert('RGB')
        result[frame_id] = raw_image

    return result

