import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings
import cv2 as cv

class CDTB(BaseVideoDataset):
    """ CDTB dataset

        return
            - [depth, depth, depth]
            - colormap from depth
            - rgb
            - rgb + depth
            - rgb + colormap
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, dtype='depth', split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().cdtb_dir if root is None else root
        super().__init__('CDTB', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the CDTB root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'cdtb_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'cdtb_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            self.sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        elif seq_ids is not None:
            self.sequence_list = [self.sequence_list[i] for i in seq_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        # self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

        self.dtype = dtype

    def get_name(self):
        return 'cdtb_depth'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _build_seq_per_class(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('_')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        '''
            CDTB provide 80 sequences
        '''
        # with open(os.path.join(self.root, 'list.txt')) as f:
        #     dir_list = list(csv.reader(f))
        # dir_list = [dir_name[0] for dir_name in dir_list]
        dir_list = pandas.read_csv(os.path.join(self.root, 'list.txt'), header=None, squeeze=True).values.tolist()
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        # gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        with open(bb_anno_file, 'r') as fp:
            lines = fp.readlines()
        lines = [line.strip() for line in lines]
        gt = []
        for line in lines:
            gt.append([float(b) for b in line.split(',')])
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "full-occlusion.tag")
        out_of_view_file = os.path.join(seq_path, "out-of-frame.tag")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion & ~out_of_view

        return target_visible

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 5) & (bbox[:, 3] > 5)
        visible = self._read_target_visible(seq_path)
        visible = visible & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # if self.dtype in ['raw_depth', 'normalized_depth', 'colormap']:
        #     frame_path = os.path.join(seq_path, 'depth/{:08}.png'.format(frame_id+1))  # frames start from 1
        # elif self.dtype == 'color':
        #     frame_path = os.path.join(seq_path, 'color/{:08}.jpg'.format(frame_id+1))
        # elif self.dtype == 'rgbd':
        #     color_frame_path = os.path.join(seq_path, 'color/{:08}.jpg'.format(frame_id+1))
        #     depth_frame_path = os.path.join(seq_path, 'depth/{:08}.png'.format(frame_id+1))
        #     frame_path = [color_frame_path, depth_frame_path]
        # else:
        #     frame_path = os.path.join(seq_path, 'depth/{:08}.png'.format(frame_id+1))

        return os.path.join(seq_path, 'color', '{:08}.jpg'.format(frame_id+1)) , os.path.join(seq_path, 'depth', '{:08}.png'.format(frame_id+1))

    def _get_frame(self, seq_path, frame_id, depth_threshold=None):

        color_path, depth_path = self._get_frame_path(seq_path, frame_id)

        rgb = cv.imread(color_path)
        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
        dp = cv.imread(depth_path, -1)

        if self.dtype == 'color':
            # img = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
            img = rgb

        elif self.dtype == 'rgbcolormap':

            colormap = cv.normalize(dp, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            colormap = np.asarray(colormap, dtype=np.uint8)
            colormap = cv.applyColorMap(colormap, cv.COLORMAP_JET)

            img = cv.merge((rgb, colormap))

        elif self.dtype == 'centered_colormap':
            if bbox is None:
                print('Error !!! require bbox for centered_colormap')
                return
            target_depth = get_target_depth(dp, bbox)
            img = get_layered_image_by_depth(dp, target_depth, dtype=self.dtype)

        elif self.dtype == 'colormap':

            dp = cv.normalize(dp, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            img = cv.applyColorMap(dp, cv.COLORMAP_JET)

        elif self.dtype == 'colormap_normalizeddepth':
            '''
            Colormap + depth
            '''
            dp = cv.normalize(dp, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)

            colormap = cv.applyColorMap(dp, cv.COLORMAP_JET)
            r, g, b = cv.split(colormap)
            img = cv.merge((r, g, b, dp))

        elif self.dtype == 'raw_depth':
            # No normalization here !!!!
            image = cv.merge((dp, dp, dp))

        elif self.dtype == 'normalized_depth':
            dp = cv.normalize(dp, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            img = cv.merge((dp, dp, dp)) # H * W * 3

        elif self.dtype == 'rgbd':
            r, g, b = cv.split(rgb)
            dp = cv.normalize(dp, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            img = cv.merge((r, g, b, dp))
        else:
            print('no such dtype ... : %s'%self.dtype)
            img = None

        return img

    # def get_class_name(self, seq_id):
    #     obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
    #
    #     return obj_meta['object_class_name']

    def _get_class(self, seq_path):
        raw_class = seq_path.split('_')[0]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None, depth_threshold=8000):
        seq_path = self._get_sequence_path(seq_id)
        # obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        obj_class = self._get_class(seq_path)

        frame_list = [self._get_frame(seq_path, f_id, depth_threshold=depth_threshold) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        obj_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, obj_meta
