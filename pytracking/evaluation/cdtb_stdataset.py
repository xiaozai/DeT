import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os

class CDTBD_STDataset(BaseDataset):
    """
    CDTB, RGB dataset, Depth dataset, Colormap dataset, RGB+depth
    """
    def __init__(self, dtype='colormap'):
        super().__init__()
        self.base_path = self.env_settings.cdtb_st_path
        self.sequence_list = self._get_sequence_list()
        self.dtype = dtype

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        start_frame = 1

        if self.dtype == 'color':
            ext = 'jpg'
        elif self.dtype in ['rgbd', 'rgbcolormap', 'rgb3d']:
            ext = ['jpg', 'png'] # Song not implemented yet
        else:
            ext = 'png'

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        if self.dtype in ['colormap', 'normalized_depth', 'raw_depth', 'centered_colormap', 'centered_normalized_depth', 'centered_raw_depth']:
            group = 'depth'
        elif self.dtype == 'color':
            group = self.dtype
        else:
            group = self.dtype

        if self.dtype in ['rgbd', 'rgbcolormap', 'rgb3d']:
            frames = [{'color': '{base_path}/{sequence_path}/color/{frame:0{nz}}.jpg'.format(base_path=self.base_path,sequence_path=sequence_path, frame=frame_num, nz=nz),
                       'depth': '{base_path}/{sequence_path}/depth/{frame:0{nz}}.png'.format(base_path=self.base_path,sequence_path=sequence_path, frame=frame_num, nz=nz)
                       }for frame_num in range(start_frame, end_frame+1)]

        else:
            frames = ['{base_path}/{sequence_path}/{group}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                      sequence_path=sequence_path, group=group, frame=frame_num, nz=nz, ext=ext)
                      for frame_num in range(start_frame, end_frame+1)]

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)

        return Sequence(sequence_name, frames, 'cdtb_st', ground_truth_rect, dtype=self.dtype)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):

        sequence_list = os.listdir(self.base_path)
        try:
            sequence_list.remove('list.txt')
        except:
            pass

        # sequence_list= ['backpack_blue',
        #                 'backpack_robotarm_lab_occ',
        #                 'backpack_room_noocc_1',
        #                 'bag_outside',
        #                 'bicycle2_outside',
        #                 'bicycle_outside',
        #                 'bottle_box',
        #                 'bottle_room_noocc_1',
        #                 'bottle_room_occ_1',
        #                 'box1_outside',
        #                 'box_darkroom_noocc_1',
        #                 'box_darkroom_noocc_10',
        #                 'box_darkroom_noocc_2',
        #                 'box_darkroom_noocc_3',
        #                 'box_darkroom_noocc_4',
        #                 'box_darkroom_noocc_5',
        #                 'box_darkroom_noocc_6',
        #                 'box_darkroom_noocc_7',
        #                 'box_darkroom_noocc_8',
        #                 'box_darkroom_noocc_9',
        #                 'box_humans_room_occ_1',
        #                 'box_room_noocc_1',
        #                 'box_room_noocc_2',
        #                 'box_room_noocc_3',
        #                 'box_room_noocc_4',
        #                 'box_room_noocc_5',
        #                 'box_room_noocc_6',
        #                 'box_room_noocc_7',
        #                 'box_room_noocc_8',
        #                 'box_room_noocc_9',
        #                 'box_room_occ_1',
        #                 'box_room_occ_2',
        #                 'boxes_backpack_room_occ_1',
        #                 'boxes_humans_room_occ_1',
        #                 'boxes_office_occ_1',
        #                 'boxes_room_occ_1',
        #                 'cart_room_occ_1',
        #                 'cartman',
        #                 'cartman_robotarm_lab_noocc',
        #                 'case',
        #                 'container_room_noocc_1',
        #                 'dog_outside',
        #                 'human_entry_occ_1',
        #                 'human_entry_occ_2',
        #                 'humans_corridor_occ_1',
        #                 'humans_corridor_occ_2_A',
        #                 'humans_corridor_occ_2_B',
        #                 'humans_longcorridor_staricase_occ_1',
        #                 'humans_shirts_room_occ_1_A',
        #                 'humans_shirts_room_occ_1_B',
        #                 'jug',
        #                 'mug_ankara',
        #                 'mug_gs',
        #                 'paperpunch',
        #                 'person_outside',
        #                 'robot_corridor_noocc_1',
        #                 'robot_corridor_occ_1',
        #                 'robot_human_corridor_noocc_1_A',
        #                 'robot_human_corridor_noocc_1_B',
        #                 'robot_human_corridor_noocc_2',
        #                 'robot_human_corridor_noocc_3_A',
        #                 'robot_human_corridor_noocc_3_B',
        #                 'robot_lab_occ',
        #                 'teapot',
        #                 'tennis_ball',
        #                 'thermos_office_noocc_1',
        #                 'thermos_office_occ_1',
        #                 'toy_office_noocc_1',
        #                 'toy_office_occ_1',
        #                 'trashcan_room_occ_1',
        #                 'trashcans_room_occ_1_A',
        #                 'trashcans_room_occ_1_B',
        #                 'trendNetBag_outside',
        #                 'trendNet_outside',
        #                 'trophy_outside',
        #                 'trophy_room_noocc_1',
        #                 'trophy_room_occ_1',
        #                 'two_mugs',
        #                 'two_tennis_balls',
        #                 'XMG_outside']

        return sequence_list
