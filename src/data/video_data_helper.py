import logging
import pdb
import random
import torch
import numpy as np
import pickle
import glob
logger = logging.getLogger(__name__)


SCORE_THRESHOLD = 0.8


def load_video_data(in_file_name, args):
    videos = {}
    if '@@@' in in_file_name:
        file_names = in_file_name.split('@@@')
    else:
        file_names = [in_file_name]
    for file_name in file_names:
        if len(file_name) == 0:
            continue

        is_gt = 'train.csv' in file_name
        logger.info('Loading from %s (is GT? %s)' % (file_name, str(is_gt)))
        with open(file_name, 'r') as f:
            for line in f:
                items = line.split(',')
                assert len(items) == 11
                if not is_gt:
                    score = float(items[7])
                    assert score <= 1.0 and score >= 0.0, score
                    if score < SCORE_THRESHOLD:
                        continue

                video_name = items[0]
                if is_gt:
                    video_name = 'GT_' + video_name
                scene_id = int(items[-2])
                link_id = int(items[-3])
                box = ','.join(items[2:6])
                if items[6] == '':
                    action = -1
                else:
                    action = int(items[6]) - 1

                sec = int(items[1])

                if video_name not in videos:
                    videos[video_name] = {}
                    prev_scene_id = -1

                assert scene_id >= prev_scene_id, (scene_id, prev_scene_id)
                prev_scene_id = scene_id

                if sec not in videos[video_name]:
                    videos[video_name][sec] = {}

                if box in videos[video_name][sec]:
                    assert videos[video_name][sec][box][:2] == (scene_id, link_id)
                    videos[video_name][sec][box][2].append(action)
                else:
                    videos[video_name][sec][box] = (scene_id, link_id, [action])


    logger.info('\t{} videos loaded.'.format(len(videos)))
    return videos


def load_mc_video_data(args, evaluate):
    with open(f'data/instance_meta/instance_meta_{args.long_term_task_name}.pkl', 'rb') as fin:
        videos = pickle.load(fin)

    if not args.train_long_term:
        for video_id in videos:
            for sec in videos[video_id]:
                for box in videos[video_id][sec]:
                    videos[video_id][sec][box].append([-1])
        return videos, set(), set()

    val_set = set()
    test_set = set()
    videos_new = {}
    for split in (['val', 'test'] if evaluate else ['train']):
        with open(f'data/lvu_1.0/{args.long_term_task_name}/{split}.csv', 'r') as f:
            f.readline()
            for line in f:
                if args.long_term_task_name == 'view_count':
                    label = float(np.log(float(line.split()[0])))

                    # make zero-mean
                    label -= 11.76425435683139
                elif args.long_term_task_name == 'like_ratio':
                    items = line.split()
                    like, dislike = float(items[0]), float(items[1])
                    label = like / (like + dislike) * 10.0

                    # make zero-mean
                    label -= 9.138220535629456
                else:
                    label = int(line.split()[0])


                video_id = line.split()[-2].strip()
                for sec in videos[video_id]:
                    for box in videos[video_id][sec]:
                        videos[video_id][sec][box].append(label)

                videos_new[video_id] = videos[video_id]
                if split == 'test':
                    test_set.add(video_id)
                elif split == 'val':
                    val_set.add(video_id)
    return videos_new, val_set, test_set



def load_features(in_file_name, args):

    if '@@@' in in_file_name:
        file_names = in_file_name.split('@@@')
    else:
        file_names = [in_file_name]

    features = {}
    feature_count = 0
    for file_name in file_names:
        is_gt = 'train_features' in file_name
        logger.info('Loading features from {} (is GT? {})'.format(file_name, is_gt))
        if file_name == '':
            continue
        with open(file_name, 'rb') as f:
            X, boxes, meta = pickle.load(f)

        for i in range(X.shape[0]):
            box = ','.join(['%.03f' % x for x in boxes[i][1:]])
            video_name = meta[i][0]
            if is_gt:
                video_name = 'GT_' + video_name
            sec = meta[i][1]

            if video_name not in features:
                features[video_name] = {}

            if sec not in features[video_name]:
                features[video_name][sec] = {}

            if box in features[video_name][sec]:
                logger.info((video_name, sec, box,
                             features[video_name][sec][box], X[i], i))
            features[video_name][sec][box] = X[i]
        feature_count += X.shape[0]

    logger.info('{} features of {} videos loaded.'.format(
        feature_count, len(features)))
    return features



def binarize(indices, num_classes=None):
    if num_classes is None:
        num_classes = 80

    vec = np.zeros((num_classes, ))
    for idx in indices:
        if idx == -1:
            continue

        vec[idx] = 1
    return vec
