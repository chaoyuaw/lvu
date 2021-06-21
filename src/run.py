# coding=utf-8

# Based on:
# HuggingFace Transformers
# See https://github.com/huggingface/transformers/LICENSE for details.
#################################################
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import pdb
from typing import Dict, List, Tuple
import time
import hashlib

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score


from models import (
    WEIGHTS_NAME,
    AdamW,
    PreTrainedModel,
    RobertaConfig,
    RobertaForMaskedLM,
    get_linear_schedule_with_warmup,
)
from data import video_data_helper
from data.video_data_helper import binarize
from utils.ava_eval_helper import evaluate_ava


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "roberta": (RobertaConfig, RobertaForMaskedLM),
}

ZERO_FEAT2304 = np.zeros((2304,))

EVAL_START_SEC = 902 # inclusive
EVAL_END_SEC = 1799 # not inclusive


with open('data/ava/slowfast_baseline_outputs/ava_eval_data.pkl', 'rb') as f:
    (excluded_keys,
     class_whitelist,
     categories,
     groundtruth,
     video_idx_to_name) = pickle.load(f)
with open('data/ava/slowfast_baseline_outputs/predictions-29.4.pkl', 'rb') as f:
    (all_preds,
     all_ori_boxes,
     all_metadata) = pickle.load(f)
video_name_to_idx = {video_idx_to_name[key] : key for key in range(len(video_idx_to_name))}
logger.info(video_name_to_idx)
logger.info(video_idx_to_name)

proj_W = None
proj_b = None


def proj(x):
    return torch.matmul(x, proj_W) + proj_b


class VideoDataset(Dataset):
    def __init__(self, args, evaluate):

        self.evaluate = evaluate
        self.secs_per_example = args.secs_per_example


        self.all_features = video_data_helper.load_features(
            args.eval_feature_file if evaluate else args.train_feature_file,
            args,
        )

        if args.action_recognition:
            self.videos = video_data_helper.load_video_data(
                args.eval_data_file if evaluate else args.train_data_file,
                args,
            )
        elif args.train_long_term:
            self.videos, self.val_set, self.test_set = video_data_helper.load_mc_video_data(args, evaluate)
        else:
            if evaluate:
                self.videos = video_data_helper.load_video_data(
                    args.eval_data_file if evaluate else args.train_data_file,
                    args,
                )
            else:
                self.videos, self.val_set, self.test_set = video_data_helper.load_mc_video_data(args, evaluate)
        self.args = args

        self.spans = []
        if args.action_recognition:
            for video_name in self.videos.keys():
                v = self.videos[video_name]
                # for action recognition only, both train and test use 15 min only.
                for center_sec in range(EVAL_START_SEC, EVAL_END_SEC):
                    if sum(
                            [sec in v.keys()
                             for sec in range(center_sec - self.secs_per_example // 2,
                                              center_sec + self.secs_per_example // 2)
                            ]) > 0:
                        self.spans.append((video_name, center_sec, None))
            if evaluate:
                self.spans = self.spans * args.eval_sample_x

        if args.same_movie:
            self.spans = {}
        for video_name in self.videos.keys():


            v = self.videos[video_name]

            if args.same_movie:
                positive_id = video_name

            # complete spans
            range_start = min(v.keys()) + self.secs_per_example - 1
            range_end = max(v.keys()) + 1
            gap = 60 if (self.evaluate and not args.is_end_task) else 1

            found_long_span = False
            for tail_sec in range(range_start, range_end, gap):
                if sum(
                        [sec in v.keys()
                         for sec in range(tail_sec + 1 - self.secs_per_example,
                                          tail_sec + 1)
                        ]) > 0:
                    if args.same_movie:
                        if positive_id not in self.spans:
                            self.spans[positive_id] = []
                        self.spans[positive_id].append((video_name, None, tail_sec))
                    else:
                        self.spans.append((video_name, None, tail_sec))
                        found_long_span = True
            if not found_long_span and args.train_long_term:
                self.spans.append((video_name, None, range_end - 1))

        self.force_len = None
        print(len(set([x[0] for x in self.spans])), 'videos in spans in total')
        print(len(self.videos), 'video data loaded in total')

    def __len__(self):
        if self.args.is_end_task and not self.evaluate:
            return len(set([x[0] for x in self.spans])) * int(self.args.num_train_epochs)
        if self.force_len is not None:
            return self.force_len
        if self.args.same_movie:
            return sum([len(x) for x in self.spans.values()])

        return len(self.spans)

    def __getitem__(self, item):

        if self.args.same_movie:
            if self.evaluate:
                positive_id = list(self.spans.keys())[item % len(self.spans.keys())]
            else:
                positive_id = random.choice(list(self.spans.keys()))
            selected = [random.choice(self.spans[positive_id]) for _ in range(2)]
        else:
            if self.evaluate:
                selected = [self.spans[item % len(self.spans)]]
            else:
                selected = [random.choice(self.spans)]

        ret = []
        construct_func = self.construct_example

        for video_name, center_start, tail_start in selected:
            for _ in range(100):
                one_ex = construct_func(
                    video_name,
                    center_start=center_start,
                    tail_start=tail_start
                )
                if one_ex is not None:
                    break
                v = self.videos[video_name]
                tail_start = random.choice(range(min(v.keys()), max(v.keys()) + 1))

            ret.append(one_ex + [video_name])
        return ret



    def construct_example(self, video_name, center_start=None, tail_start=None):

        def get_spatial_encoding(box, perturb=0.0):
            box = [float(x) for x in box.split(',')]
            if perturb > 0 and not self.evaluate:
                p0 = (box[2] - box[0]) * perturb
                p1 = (box[3] - box[1]) * perturb
                box = [
                    box[0] + p0 * random.uniform(-1.0, 1.0),
                    box[1] + p1 * random.uniform(-1.0, 1.0),
                    box[2] + p0 * random.uniform(-1.0, 1.0),
                    box[3] + p1 * random.uniform(-1.0, 1.0),
                ]
            box.append((box[2] - box[0]) * (box[3] - box[1]))
            return np.array(box)
        args = self.args

        is_pretrain = (not args.action_recognition) and (not args.train_long_term)
        is_mc = not args.action_recognition and not (is_pretrain and self.evaluate)

        video = self.videos[video_name]

        if is_mc:
            video_features = np.load(
                os.path.join(args.mc_train_feature_file, video_name + '.npz'),
                allow_pickle=True,
            )['a'].item()
        else:
            video_features = self.all_features[video_name] if (
                self.all_features is not None) else None

        ex_link_ids = []
        ex_scene_ids = []
        ex_boxes = []
        ex_secs = []
        ex_actions = []
        ex_long_term = []
        ex_features = []
        ex_spatial = []

        all_tube_exs = {}
        for shift_idx, sec_shift in enumerate(range(self.secs_per_example)):

            if center_start is not None:
                if sec_shift % 2 == 0:
                    sec = center_start + (sec_shift + 1) // 2
                    auged_sec = center_start + (shift_idx + 1) // 2
                else:
                    sec = center_start - (sec_shift + 1) // 2
                    auged_sec = center_start - (shift_idx + 1) // 2
            if tail_start is not None:
                sec = tail_start - sec_shift
                auged_sec = tail_start - shift_idx
            if sec in video:
                for box, (scene_id, link_id, actions) in video[sec].items():

                    if len(ex_link_ids) < args.max_position_embeddings - 4:
                        ex_link_ids.append(link_id)
                        ex_secs.append(auged_sec)
                        ex_scene_ids.append(scene_id)
                        ex_boxes.append(box)
                        if args.action_recognition:
                            ex_actions.append(binarize(actions))
                        if args.train_long_term:
                            before_action = actions

                            ex_long_term.append(actions)

                        cur_feat = video_features[sec][box]
                        cur_mc_feat_ava = None
                        if is_mc:
                            cur_mc_feat_ava = cur_feat

                        ex_features.append(cur_feat)

                        ex_spatial.append(get_spatial_encoding(box, 0.2))


        if len(ex_secs) == 0:
            return None

        original_ex_secs = ex_secs
        assert (max(ex_secs) - min(ex_secs)) < args.secs_per_example

        halfway = args.max_position_embeddings // 2

        if tail_start is None:
            tail_start = max(ex_secs)
        if center_start is None:
            center_start = (max(ex_secs) + min(ex_secs)) // 2

        increasing_pos_ids = [x - min(ex_secs) for x in ex_secs]
        decreasing_pos_ids = [max(ex_secs) - x for x in ex_secs]
        center_pos_ids = [max(0, x - center_start + halfway) for x in ex_secs]

        increasing_scene_ids = [x - min(ex_scene_ids) for x in ex_scene_ids]
        decreasing_scene_ids = [max(ex_scene_ids) - x for x in ex_scene_ids]

        dists = [abs(x - center_start) for x in ex_secs]
        for dist, tmp_scene_id in zip(dists, ex_scene_ids):
            if dist == min(dists):
                center_scene_id = tmp_scene_id

        center_scene_ids = [max(0, x - center_scene_id + halfway) for x in ex_scene_ids]


        n_links = len(set(ex_link_ids))
        rand_link_ids = dict(zip(
            list(set(ex_link_ids)),
            random.sample(range(n_links), n_links),
        ))
        ex_link_ids = [rand_link_ids[x] + 2 for x in ex_link_ids]


        if args.action_recognition:
            ex_actions = [binarize([])] + ex_actions + [binarize([])]
        else:
            ex_actions = []

        if args.train_long_term:
            ex_long_term = [-1] + ex_long_term + [-1]
        else:
            ex_long_term = []

        ex_link_ids = [0] + ex_link_ids + [1] # end doens't belong to a link

        increasing_pos_ids = [0] + [x + 2 for x in increasing_pos_ids] + [1] # end can have a new pos
        decreasing_pos_ids = [0] + [x + 2 for x in decreasing_pos_ids] + [1] # end can have a new pos
        center_pos_ids = [0] + [x + 2 for x in center_pos_ids] + [1] # end can have a new pos

        increasing_scene_ids = [0] + [x + 2 for x in increasing_scene_ids] + [1]
        decreasing_scene_ids = [0] + [x + 2 for x in decreasing_scene_ids] + [1]
        center_scene_ids = [0] + [x + 2 for x in center_scene_ids] + [1]

        ex_features = [ZERO_FEAT2304] + ex_features + [ZERO_FEAT2304]

        ex_spatial = [ex_spatial[0] * 0.0] + ex_spatial + [ex_spatial[0] * 0.0]


        return [torch.tensor(ex_link_ids) + 2,
                torch.tensor(increasing_pos_ids) + 2,
                torch.tensor(decreasing_pos_ids) + 2,
                torch.tensor(center_pos_ids) + 2,
                torch.tensor(increasing_scene_ids) + 2,
                torch.tensor(decreasing_scene_ids) + 2,
                torch.tensor(center_scene_ids) + 2,
                torch.tensor(ex_actions),
                torch.tensor(ex_long_term),
                torch.from_numpy(np.ascontiguousarray(ex_features)),
                torch.tensor(ex_spatial),
                ex_secs,
                ex_boxes,
               ]

def set_seed(args):
    seed = args.seed + args.local_rank + 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    # if args.save_total_limit <= 0:
    #     return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def get_mask_indices(x_batch, masked_indices, features=None):
    use_pos = None
    if features is not None:
        use_pos = (features.sum(axis=2) != 0)

    for i in range(x_batch.shape[0]):
        if use_pos is not None:
            cur_x = x_batch[i][use_pos[i]]
        else:
            cur_x = x_batch[i]

        x_ids = set(cur_x.tolist()) - set([1, 2, 3]) # remove padding, start, and end
        assert 0 not in x_ids

        if len(x_ids) == 0:
            if features is not None and features.shape[-1] != 1024:
                logger.info('warning: no masked elements in example')
            continue

        group_mask = {}
        while sum(group_mask.values()) < 1:
            group_mask = {x_id: int(np.random.choice([0, 1], p=[0.85, 0.15])) for x_id in x_ids}

        for x_id in x_ids:
            if group_mask[x_id] == 1:
                assert x_id > 3
                masked_indices[i, x_batch[i] == x_id] = 1
        assert masked_indices[i].sum() > 0

    return masked_indices



def perform_masking(masked_indices, inputs_embed_batch, contents):
    mask_dim = inputs_embed_batch.shape[2]

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(
        torch.full(masked_indices.shape, 0.8)
    ).bool() & masked_indices

    feat_mask = indices_replaced.view((
        indices_replaced.shape[0],
        indices_replaced.shape[1],
        1)).expand(-1, -1, mask_dim)

    inputs_embed_batch[feat_mask] = -10

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(masked_indices.shape, 0.5)).bool() & masked_indices & ~indices_replaced

    not_replaced = inputs_embed_batch[(~indices_replaced & contents)].reshape((-1, mask_dim))

    num_to_sample = int(indices_random.sum())
    num_features = not_replaced.shape[0]
    if num_to_sample > 0 and num_features > 0:
        random_indices = np.random.choice(num_features, num_to_sample)
        inputs_embed_batch[indices_random[:, :, None].repeat(1, 1, mask_dim)] = not_replaced[random_indices].reshape((-1,))
    return inputs_embed_batch


def mask_tokens(link_batch: torch.Tensor,
                inc_scene_batch: torch.Tensor,
                action_batch: torch.Tensor,
                soft_label_batch: torch.Tensor,
                inputs_embed_batch: torch.Tensor,
                center_pos_batch: torch.Tensor,
                args,
                is_eval=False,
                dec_pos_batch=None) -> Tuple[torch.Tensor, torch.Tensor]:

    ################################################################################
    masked_indices = torch.zeros(link_batch.shape, dtype=torch.int64)
    scene_masked_indices = None

    masked_indices = get_mask_indices(link_batch, masked_indices, inputs_embed_batch)


    masked_indices = masked_indices.bool()

    ################################################################################
    ################################################################################
    contents = (center_pos_batch != 1) & (center_pos_batch != 2) & (center_pos_batch != 3)

    out_masked_indices = masked_indices.clone().detach()


    if args.action_recognition:
        action_batch[~out_masked_indices[:, :, None].expand(-1, -1, args.num_action_classes)] = -100  # We only compute loss on masked tokens

    if args.use_soft_labels:
        soft_label_batch[~out_masked_indices[:, :, None].expand(-1, -1, args.soft_label_dim)] = -100

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(link_batch.shape, 0.8)).bool() & masked_indices

    if args.mask_sep:
        if not args.mask_sep_no_mask:

            start = 0
            for cur_feat_dim in args.all_feat_dims:
                if cur_feat_dim == args.action_feat_dim:
                    cur_masked_indices = masked_indices
                else:
                    assert False

                inputs_embed_batch[:, :, start:start + cur_feat_dim] = perform_masking(
                    cur_masked_indices,
                    inputs_embed_batch[:, :, start:start + cur_feat_dim], contents)

                start += cur_feat_dim


    inputs_embed_batch[:, :, :args.action_feat_dim] = perform_masking(
        masked_indices, inputs_embed_batch[:, :, :args.action_feat_dim], contents)
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return (action_batch, inputs_embed_batch, soft_label_batch, masked_indices)


def shared_collate(all_examples: List[torch.Tensor]):
    if len(all_examples[0]) == 1:
        all_examples = [x[0] for x in all_examples]
    elif len(all_examples[0]) == 2:
        all_examples = [x[0] for x in all_examples] + [x[1] for x in all_examples]

    assert len(all_examples[0]) == 14
    zipped = list(zip(*all_examples))

    meta = [list(examples) for examples in zipped[9:]]


    padding_value = 1
    padding_values = [padding_value] * 7 + [-100] * 2

    return [pad_sequence(list(examples), batch_first=True, padding_value=padding_values[i])
            for i, examples in enumerate(zipped[:9])] + meta


def prepare_model_input(link_batch,
                        inc_pos_batch, dec_pos_batch, center_pos_batch,
                        inc_scene_batch, dec_scene_batch, center_scene_batch,
                        action_batch,  
                        feature_batch, spatial_batch, sec_batch,
                        args, is_eval=False):


    inputs_embed_batch = pad_feature_batch(feature_batch, args.device)
    soft_label_batch = proj(inputs_embed_batch) if args.use_soft_labels else None

    spatial_batch = pad_feature_batch(spatial_batch, args.device)

    if args.action_recognition:
        outputs_embed_batch = inputs_embed_batch.clone().detach()
    else:
        outputs_embed_batch = None

    if args.mask_sep:
        start = 0
        for cur_feat_dim in args.all_feat_dims:
            if cur_feat_dim == 2304:
                pass
            else:
                assert False
            start += cur_feat_dim

    (action_batch, inputs_embed_batch, soft_label_batch,
     target_locations) = mask_tokens(
        link_batch,
        inc_scene_batch,
        action_batch,
        soft_label_batch,
        inputs_embed_batch,
        center_pos_batch,
        args,
        is_eval=is_eval,
        dec_pos_batch=dec_pos_batch)

    if action_batch is not None:
        action_batch = action_batch.to(args.device)

    target_locations = target_locations.to(args.device)


    link_batch = link_batch.to(args.device)

    inc_pos_batch = inc_pos_batch.to(args.device)
    dec_pos_batch = dec_pos_batch.to(args.device)
    center_pos_batch = center_pos_batch.to(args.device)

    inc_scene_batch = inc_scene_batch.to(args.device)
    dec_scene_batch = dec_scene_batch.to(args.device)
    center_scene_batch = center_scene_batch.to(args.device)


    return (action_batch, link_batch,
            inc_pos_batch, dec_pos_batch, center_pos_batch,
            inc_scene_batch, dec_scene_batch, center_scene_batch,
            inputs_embed_batch, outputs_embed_batch, spatial_batch,
            soft_label_batch, 
            target_locations)


def freeze(mod):
    count = 0
    for p in mod.parameters():
        p.requires_grad = False
        count += 1
    logger.info('freeze {} ({} params)'.format(mod, count))


def pad_feature_batch(feature_batch, device):
    batch_size = len(feature_batch)
    max_len = max([len(x) for x in feature_batch])
    dim = feature_batch[0][0].shape[0]

    batch = torch.zeros((batch_size, max_len, dim), device=device)
    for i in range(batch_size):
        batch[i, :len(feature_batch[i])] = feature_batch[i].to(device)
    return batch


def train(args, train_dataset, model: PreTrainedModel) -> Tuple[int, float]:
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    def collate(all_examples: List[torch.Tensor]):
        return shared_collate(all_examples)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        if args.is_end_task:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    tmp_model = model.module if hasattr(model, "module") else model
    if args.action_recognition:
        logger.warn('Initializing final W/b')
        state_dict = torch.load(
            args.short_term_model_weights,
            map_location="cpu",
        )
        tmp_model.action_lm_head.decoder_feat.weight = nn.Parameter(state_dict['model_state']['head.projection.weight'])
        tmp_model.action_lm_head.decoder_feat.bias = nn.Parameter(state_dict['model_state']['head.projection.bias'])

        pretrained_state_dict = torch.load(args.force_load_checkpoint, map_location="cpu")

        tmp_weight = pretrained_state_dict['action_lm_head.decoder.weight']
        if tmp_model.action_lm_head.decoder.weight.shape == tmp_weight.shape:
            logger.warn('init pretrained weights')
            tmp_model.action_lm_head.decoder.weight = nn.Parameter(tmp_weight)
            tmp_bias = pretrained_state_dict['action_lm_head.decoder.bias']
            tmp_model.action_lm_head.bias = nn.Parameter(tmp_bias)
            tmp_model.action_lm_head.decoder.bias = tmp_model.action_lm_head.bias
        else:
            logger.warn('Not init pretrained weights {} {} not match'.format(
                tmp_model.action_lm_head.decoder.weight.shape,
                tmp_weight.shape
            ))

    if args.action_recognition:
        freeze(tmp_model.roberta)
        if hasattr(tmp_model, 'lm_head'):
            freeze(tmp_model.lm_head.dense)
        if hasattr(tmp_model, 'action_lm_head'):
            freeze(tmp_model.action_lm_head.dense)
        if hasattr(tmp_model, 'lm_head'):
            freeze(tmp_model.lm_head.layer_norm)
        if hasattr(tmp_model, 'action_lm_head'):
            freeze(tmp_model.action_lm_head.layer_norm)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    rbt_no_d = []
    final_no_d = []
    rbt_d = []
    final_d = []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            if 'roberta' in n:
                rbt_no_d.append(p)
            else:
                final_no_d.append(p)
        else:
            if 'roberta' in n:
                rbt_d.append(p)
            else:
                final_d.append(p)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(t_total * 0.1), num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
        logger.info("loading optimizer and scheduler from {}".format(args.model_name_or_path))

    if (
        args.force_load_checkpoint_opt
        and os.path.isfile(os.path.join(args.force_load_checkpoint_opt, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.force_load_checkpoint_opt, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.force_load_checkpoint_opt, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.force_load_checkpoint_opt, "scheduler.pt")))
        logger.info("loading optimizer and scheduler from {}".format(args.force_load_checkpoint_opt))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_remain_in_current_epoch = -1
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    if args.force_load_checkpoint_opt:
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.force_load_checkpoint_opt.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_remain_in_current_epoch = (len(train_dataloader) // args.gradient_accumulation_steps) - steps_trained_in_current_epoch
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            # logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            logger.info("  Will train only %d steps in the first epoch", steps_remain_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    lm_action_loss, same_movie_loss = 0.0, 0.0


    model = model.to(args.device)

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, 1 if args.is_end_task else int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility

    logger.info(model)

    is_first_epoch = True
    for cur_epoch in train_iterator:
        if steps_remain_in_current_epoch > -1:
            tr_d = train_dataloader.dataset
            if is_first_epoch:
                original_dataset_len = len(tr_d)
                tr_d.force_len = steps_remain_in_current_epoch * args.train_batch_size
                is_first_epoch = False
            else:
                tr_d.force_len = original_dataset_len

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, (link_batch,
                   inc_pos_batch, dec_pos_batch, center_pos_batch,
                   inc_scene_batch, dec_scene_batch, center_scene_batch,
                   action_batch, long_term_batch,
                   feature_batch, spatial_batch, sec_batch, box_batch,
                   video_name_batch) in enumerate(epoch_iterator):

            (action_batch, link_batch,
             inc_pos_batch, dec_pos_batch, center_pos_batch,
             inc_scene_batch, dec_scene_batch, center_scene_batch,
             inputs_embed_batch, outputs_embed_batch, spatial_batch,
             soft_label_batch, 
             target_locations) = prepare_model_input(
                    link_batch,
                    inc_pos_batch, dec_pos_batch, center_pos_batch,
                    inc_scene_batch, dec_scene_batch, center_scene_batch,
                    action_batch, 
                    feature_batch, spatial_batch, sec_batch, args)

            model.train()

            outputs = model(
                link_ids=None if args.no_link_ids else link_batch,
                inc_scene_ids=None if args.no_scene_ids else inc_scene_batch,
                dec_scene_ids=None if args.no_scene_ids else dec_scene_batch,
                center_scene_ids=None if args.no_scene_ids else center_scene_batch,
                inc_position_ids=None if args.no_pos_ids else inc_pos_batch,
                dec_position_ids=None if args.no_pos_ids else dec_pos_batch,
                center_position_ids=None if args.no_pos_ids else center_pos_batch,
                action_labels=action_batch,####
                long_term_labels=long_term_batch,
                inputs_embeds=inputs_embed_batch,
                outputs_embeds=outputs_embed_batch,
                spatial_codes=spatial_batch,
                soft_labels=soft_label_batch,
                target_locations=target_locations,
                secs=sec_batch,
                boxes=box_batch,
                args=args)
            losses = outputs[0]  # model outputs are always tuple in transformers (see doc)



            if step == 0:
                logger.info(losses)


            if args.n_gpu > 1:
                loss = sum([loss.mean() for loss in losses.values()])  # mean() to average on multi-gpu parallel training
            else:
                loss = sum(losses.values())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if 'lm_action' in losses:
                lm_action_loss += losses['lm_action'].mean().item()
            if 'same_movie' in losses:
                same_movie_loss += losses['same_movie'].mean().item()


            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if len(args.eval_epochs) == 0:
                    do_eval = (step == len(train_dataloader) - 1 and args.is_end_task) \
                            or (args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0)
                else:
                    epoch_len = len(train_dataloader) // int(args.num_train_epochs)
                    if (step + 1) % epoch_len == 0:
                        do_eval = (step + 1) in [int(x) * epoch_len for x in args.eval_epochs.strip().split(',')]
                    else:
                        do_eval = False
                if do_eval:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training and not args.is_end_task
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model)

                    logger.info(("lr", scheduler.get_lr()[0], global_step))
                    logger.info(
                        (
                            'training loss',
                            (tr_loss - logging_loss) / args.logging_steps,
                            global_step,
                        )
                    )

                    logger.info(
                        (
                            'same_movie_loss',
                            same_movie_loss / args.logging_steps,
                        )
                    )
                    logger.info(
                        (
                            'lm_action_loss',
                            lm_action_loss / args.logging_steps,
                        )
                    )
                    same_movie_loss = 0.0
                    lm_action_loss = 0.0

                    logging_loss = tr_loss

                if args.save_steps == -1:
                    do_save = do_eval
                else:
                    do_save = (args.local_rank in [-1, 0]) and (args.save_steps > 0) and (global_step % args.save_steps == 0)
                if do_save:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break


    return global_step, tr_loss / global_step


def evaluate_action_recognition(bert_all_preds, args):

    logger.info('bert output to dict')
    bert_preds = {}
    for pred_batch, video_name_batch, sec_batch, box_batch, is_center in bert_all_preds:
        pred_batch = torch.sigmoid(pred_batch)

        for i in range(len(video_name_batch)):
            video_idx = video_name_to_idx[video_name_batch[i]]

            secs = sec_batch[i]
            boxes = box_batch[i]
            for j, (ex_sec, ex_box) in enumerate(zip(secs, boxes)):

                if not is_center[i, j + 1]:
                    continue
                if video_idx not in bert_preds:
                    bert_preds[video_idx] = {}

                if isinstance(ex_sec, int):
                    sec_list = [ex_sec]
                    box_list = [ex_box]
                else:
                    sec_list = ex_sec
                    box_list = ex_box
                for sec, box in zip(sec_list, box_list):
                    if sec not in bert_preds[video_idx]:
                        bert_preds[video_idx][sec] = {}

                    if box in bert_preds[video_idx][sec]:
                        #### WTF it should be j + 1.
                        bert_preds[video_idx][sec][box].append(pred_batch[i, j + 1])
                    else:
                        bert_preds[video_idx][sec][box] = [pred_batch[i, j + 1]]

    logger.info('set all_preds to bert')
    used_count = 0
    all_preds[:, :] = 0.0
    for i in range(all_preds.shape[0]):
        video_idx = int(all_metadata[i][0])
        sec = int(all_metadata[i][1])
        box = ','.join(['%.03f' % x for x in all_ori_boxes[i][1:]])
        if video_idx in bert_preds \
                and sec in bert_preds[video_idx] \
                and box in bert_preds[video_idx][sec]:
            pred_list = bert_preds[video_idx][sec][box]
            all_preds[i, :] = sum(pred_list) / len(pred_list)
            used_count += 1

    logger.info('%d predictions used' % used_count)
    logger.info('%d predictions in total' % all_preds.shape[0])

    mean_ap = evaluate_ava(
        all_preds,
        all_ori_boxes,
        all_metadata.tolist(),
        excluded_keys,
        class_whitelist,
        categories,
        groundtruth=groundtruth,
        video_idx_to_name=video_idx_to_name,
    )
    return mean_ap * 100.0


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


def evaluate(args, model: PreTrainedModel, prefix="") -> Dict:

    logger.info(model)

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = VideoDataset(args, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(all_examples: List[torch.Tensor]):
        return shared_collate(all_examples)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate,
        num_workers=args.num_workers_eval,
        pin_memory=True,
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    all_eval_loss = 0.0


    long_term_top1 = 0.0
    long_term_count = 0

    nb_eval_steps = 0
    eval_example_count = 0
    model.eval()

    all_preds = []
    all_states = []
    for (link_batch,
            inc_pos_batch, dec_pos_batch, center_pos_batch,
            inc_scene_batch, dec_scene_batch, center_scene_batch,
            action_batch, long_term_batch,
            feature_batch, spatial_batch, sec_batch, box_batch,
            video_name_batch) in tqdm(eval_dataloader, desc="Evaluating"):

        (action_batch, link_batch,
         inc_pos_batch, dec_pos_batch, center_pos_batch,
         inc_scene_batch, dec_scene_batch, center_scene_batch,
         inputs_embed_batch, outputs_embed_batch, spatial_batch,
         soft_label_batch, 
         target_locations) = prepare_model_input(
                link_batch,
                inc_pos_batch, dec_pos_batch, center_pos_batch,
                inc_scene_batch, dec_scene_batch, center_scene_batch,
                action_batch, 
                feature_batch, spatial_batch, sec_batch, args, is_eval=True)

        with torch.no_grad():
            outputs = model(
                link_ids=None if args.no_link_ids else link_batch,
                inc_scene_ids=None if args.no_scene_ids else inc_scene_batch,
                dec_scene_ids=None if args.no_scene_ids else dec_scene_batch,
                center_scene_ids=None if args.no_scene_ids else center_scene_batch,
                inc_position_ids=None if args.no_pos_ids else inc_pos_batch,
                dec_position_ids=None if args.no_pos_ids else dec_pos_batch,
                center_position_ids=None if args.no_pos_ids else center_pos_batch,
                action_labels=action_batch,
                long_term_labels=long_term_batch,
                inputs_embeds=inputs_embed_batch,
                outputs_embeds=outputs_embed_batch,
                spatial_codes=spatial_batch,
                soft_labels=soft_label_batch,
                target_locations=target_locations,
                secs=sec_batch,
                boxes=box_batch,
                args=args)
            losses = outputs[0]

            if args.action_recognition:
                all_preds.append((
                    outputs[1]['pred'].cpu(), video_name_batch, sec_batch, box_batch,
                    (action_batch[:, :, 0] != -100).cpu()
                ))


            if args.train_long_term:
                lt_pred = outputs[1]['long_term_logits'].cpu()
                lt_labels = long_term_batch[:, 1]

                if args.num_long_term_classes == -1:
                    lt_pred = lt_pred[:, 0]

                all_preds.append((video_name_batch, lt_pred, lt_labels))

                if args.num_long_term_classes > 0:
                    lt_pred = outputs[1]['long_term_logits'].argmax(dim=1).cpu()
                    lt_labels = long_term_batch[:, 1]
                    long_term_top1 += (lt_pred == lt_labels).sum()
                    long_term_count += lt_labels.shape[0]

            if args.mask_sep:
                eval_loss += losses['lm_action'].mean().item()
                all_eval_loss += sum([loss.mean() for loss in losses.values()]).item()
            else:
                eval_loss += sum([loss.mean() for loss in losses.values()]).item()

            eval_example_count += inc_pos_batch.shape[0]

        nb_eval_steps += 1

    mean_ap = 0.0
    if args.action_recognition:
        start_eval = time.time()
        mean_ap = evaluate_action_recognition(all_preds, args)
        logger.info('eval done in {} secs'.format(time.time() - start_eval))

    clip_mse = []
    split_result = {}
    if args.train_long_term:
        pred_agg = {}
        video_label = {}

        for video_name_batch, pred_batch, label_batch in all_preds:
            for i in range(len(video_name_batch)):
                v_name = video_name_batch[i]
                if v_name not in pred_agg:
                    if args.num_long_term_classes > 0:
                        pred_agg[v_name] = softmax(pred_batch[i])
                    else:
                        pred_agg[v_name] = [pred_batch[i]]
                    video_label[v_name] = label_batch[i]
                else:
                    if args.num_long_term_classes > 0:
                        pred_agg[v_name] += softmax(pred_batch[i])
                    else:
                        pred_agg[v_name].append(pred_batch[i])

                    assert video_label[v_name] == label_batch[i]

                if args.num_long_term_classes == -1:
                    clip_mse.append(
                        (pred_batch[i] - label_batch[i]) ** 2.0
                    )

        for split in (['val', 'test'] if args.three_split else ['val']):
            agg_sm_correct, agg_count = 0.0, 0.0
            mse = []

            for v_name in pred_agg.keys():
                if args.three_split and split == 'val':
                    if v_name not in eval_dataset.val_set:
                        continue

                if args.three_split and split == 'test':
                    if v_name not in eval_dataset.test_set:
                        continue

                if args.num_long_term_classes > 0:
                    if pred_agg[v_name].argmax() == video_label[v_name]:
                        agg_sm_correct += 1
                else:
                    mse.append(
                        (np.mean(pred_agg[v_name]) - video_label[v_name]) ** 2.0
                    )
                agg_count += 1
            if args.num_long_term_classes > 0:
                acc = 100.0 * agg_sm_correct / agg_count
                split_result[split] = f'{acc} {agg_sm_correct} {agg_count}'
            else:
                split_result[split] = f'{np.mean(mse)} {len(mse)}'


    eval_loss = eval_loss / nb_eval_steps
    all_eval_loss = all_eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    total_perplexity = torch.exp(torch.tensor(all_eval_loss))


    if long_term_count > 0:
        long_term_top1 = float(long_term_top1) / float(long_term_count)

    result = {"perplexity": perplexity,
              "all_eval_loss": all_eval_loss,
              "total_perplexity": total_perplexity,
              "map": mean_ap,
              "clip_mse": np.mean(clip_mse),
              "long_term_top1": long_term_top1,
             }
    for split in split_result.keys():
        result['agg_' + split] = split_result[split]

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} ({} examples) *****".format(prefix, eval_example_count))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=4000, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=4000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--max_iter", type=int, default=-1, help="")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--secs_per_example", type=int, default=60, help="Number of secs per example.")
    parser.add_argument("--get_mc_states_name", type=str, default="binary_task", help="")

    parser.add_argument("--same_movie", action="store_true", help="")
    parser.add_argument("--same_movie_temperature", type=float, default=0.2, help="")
    parser.add_argument("--same_movie_weight", type=float, default=1.0, help="")

    parser.add_argument("--train_long_term", action="store_true", help="")
    parser.add_argument("--train_long_term_linear", action="store_true", help="")
    parser.add_argument("--train_long_term_dropout", action="store_true", help="")

    parser.add_argument("--long_term_task_name", type=str, default="relationship", help="")
    parser.add_argument("--num_long_term_classes", type=int, default=-1, help="")

    parser.add_argument("--eval_epochs", default="", type=str, help="")

    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers.")
    parser.add_argument("--num_workers_eval", type=int, default=2, help="Number of DataLoader workers.")
    parser.add_argument("--force_load_checkpoint", type=str, default="", help="Force-load checkpoint path.")
    parser.add_argument("--force_load_checkpoint_opt", type=str, default=None, help="Force-load checkpoint path.")

    parser.add_argument("--init_final", action="store_true", help="")

    parser.add_argument("--train_feature_file", default=None, type=str, required=True, help="")
    parser.add_argument("--mc_train_feature_file", default=None, type=str, help="")
    parser.add_argument("--eval_feature_file", default=None, type=str, required=True, help="")

    parser.add_argument("--exp", default='', type=str, required=True, help="")
    parser.add_argument("--num_action_classes", type=int, default=80, help="")
    parser.add_argument("--max_position_embeddings", type=int, default=258,  help="")
    parser.add_argument("--action_recognition", action="store_true", help="")
    parser.add_argument("--num_hidden_layers", type=int, default=3,  help="")
    parser.add_argument("--num_attention_heads", type=int, default=12,  help="")

    parser.add_argument("--action_feat_dim", type=int, default=2304,  help="")
    parser.add_argument("--feat_dim", type=int, default=2304,  help="")
    parser.add_argument("--action_loss_weight", default=1.0, type=float, help="")

    parser.add_argument("--no_link_ids", action="store_true", help="")
    parser.add_argument("--no_scene_ids", action="store_true", help="")
    parser.add_argument("--no_pos_ids", action="store_true", help="")

    parser.add_argument("--use_soft_labels", action="store_true", help="")
    parser.add_argument("--mask_sep", action="store_true", help="")
    parser.add_argument("--mask_sep_no_mask", action="store_true", help="")

    parser.add_argument("--temperature", default=1.0, type=float, help="")
    parser.add_argument("--eval_sample_x", default=10, type=int, help="")

    parser.add_argument("--three_split", action="store_true", help="")

    parser.add_argument("--short_term_model_weights", default='data/ava/SLOWFAST_32x2_R101_50_50.pkl', type=str, help="")

    parser.add_argument("--debug", action="store_true", help="")
    parser.add_argument("--use_good_quality", action="store_true", help="")

    args = parser.parse_args()


    args.is_end_task = args.train_long_term or args.action_recognition

    args.all_feat_dims = [2304]


    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            args=args,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    global proj_W
    global proj_b

    tmp_state_dict = torch.load(
        args.short_term_model_weights,
        map_location="cpu",
    )
    proj_W = torch.tensor(tmp_state_dict['model_state']['head.projection.weight'].numpy()).float().T # 2304, 80
    proj_b = torch.tensor(tmp_state_dict['model_state']['head.projection.bias'].numpy()).float() # 80

    args.soft_label_dim = 80

    proj_W = proj_W.to(args.device)
    proj_b = proj_b.to(args.device)


    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = VideoDataset(args, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.is_end_task and args.local_rank in [-1, 0]:
        evaluate(args, model)


if __name__ == "__main__":
    main()
