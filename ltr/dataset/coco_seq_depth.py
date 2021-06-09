import os
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
import torch
import random
from pycocotools.coco import COCO
from collections import OrderedDict
from ltr.admin.environment import env_settings
import numpy as np
import cv2
from ltr.dataset.depth_utils import get_target_depth, get_layered_image_by_depth

class MSCOCOSeq_depth(BaseVideoDataset):
    """ The COCO dataset. COCO is an image dataset. Thus, we treat each image as a sequence of length 1.

    Publication:
        Microsoft COCO: Common Objects in Context.
        Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona,
        Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick
        ECCV, 2014
        https://arxiv.org/pdf/1405.0312.pdf

    Download the images along with annotations from http://cocodataset.org/#download. The root folder should be
    organized as follows.
        - coco_root
            - annotations
                - instances_train2014.json
                - instances_train2017.json
            - images
                - train2014
                - train2017

    Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", version="2014", dtype='depth'):
        """
        args:
            root - path to the coco dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            data_fraction (None) - Fraction of images to be used. The images are selected randomly. If None, all the
                                  images  will be used
            split - 'train' or 'val'.
            version - version of coco dataset (2014 or 2017)
        """
        root = env_settings().cocodepth_dir if root is None else root
        super().__init__('COCO_depth', root, image_loader)

        # self.img_pth = os.path.join(root, 'images/{}{}/'.format(split, version))
        self.img_pth = os.path.join(root, '{}{}/'.format(split, version))
        self.anno_path = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, version))

        self.dtype = dtype
        # Load the COCO set.
        self.coco_set = COCO(self.anno_path)

        self.cats = self.coco_set.cats

        self.class_list = self.get_class_list()

        self.sequence_list = self._get_sequence_list()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))
        self.seq_per_class = self._build_seq_per_class()

    def _get_sequence_list(self):
        ann_list = list(self.coco_set.anns.keys())
        seq_list = [a for a in ann_list if self.coco_set.anns[a]['iscrowd'] == 0]

        return seq_list

    def is_video_sequence(self):
        return False

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'coco_depth'

    def has_class_info(self):
        return True

    def get_class_list(self):
        class_list = []
        for cat_id in self.cats.keys():
            class_list.append(self.cats[cat_id]['name'])
        return class_list

    def has_segmentation_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _build_seq_per_class(self):
        seq_per_class = {}
        for i, seq in enumerate(self.sequence_list):
            class_name = self.cats[self.coco_set.anns[seq]['category_id']]['name']
            if class_name not in seq_per_class:
                seq_per_class[class_name] = [i]
            else:
                seq_per_class[class_name].append(i)

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)

        bbox = torch.Tensor(anno['bbox']).view(1, 4)

        mask = torch.Tensor(self.coco_set.annToMask(anno)).unsqueeze(dim=0)

        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid = (bbox[:, 2] > 5.0) & (bbox[:, 3] > 5.0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'mask': mask, 'valid': valid, 'visible': visible}

    def _get_anno(self, seq_id):
        anno = self.coco_set.anns[self.sequence_list[seq_id]]

        return anno

    def _get_frames(self, seq_id, depth_threshold=None, bbox=None):

        rgb_path = self.coco_set.loadImgs([self.coco_set.anns[self.sequence_list[seq_id]]['image_id']])[0]['file_name']

        depth_path = rgb_path[:-4] + '.png'

        rgb = cv2.imread(os.path.join(self.img_pth, 'color', rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        dp = cv2.imread(os.path.join(self.img_pth, 'depth', depth_path), -1)

        max_depth = min(np.max(dp), 10000)
        dp[dp > max_depth] = max_depth

        if self.dtype == 'centered_colormap':
            if bbox is None:
                print('Error !!!  centered_colormap requires BBox ')
                return
            # bbox is repeated
            target_depth = get_target_depth(dp, bbox[0])
            img = get_layered_image_by_depth(dp, target_depth, dtype=self.dtype)

        elif self.dtype == 'colormap':
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            img = cv2.applyColorMap(dp, cv2.COLORMAP_JET)

        elif self.dtype == 'colormap_depth':
            '''
            Colormap + depth
            '''
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
            r, g, b = cv2.split(colormap)
            img = cv2.merge((r, g, b, dp))

        elif self.dtype == 'depth_gray':
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            img = cv2.merge((dp, dp, dp)) # H * W * 3

        elif self.dtype == 'color':
            img = rgb

        elif self.dtype == 'rgbcolormap':
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
            img = cv2.merge((rgb, colormap))

        elif self.dtype == 'rgb3d':
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            dp = cv2.merge((dp, dp, dp))
            img = cv2.merge((rgb, dp))

        return img

    def get_meta_info(self, seq_id):
        try:
            cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[seq_id]]['category_id']]
            object_meta = OrderedDict({'object_class_name': cat_dict_current['name'],
                                       'motion_class': None,
                                       'major_class': cat_dict_current['supercategory'],
                                       'root_class': None,
                                       'motion_adverb': None})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta


    def get_class_name(self, seq_id):
        cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[seq_id]]['category_id']]
        return cat_dict_current['name']

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...] for _ in frame_ids]

        frame = self._get_frames(seq_id, bbox=anno_frames['bbox'])

        frame_list = [frame.copy() for _ in frame_ids]

        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, object_meta
