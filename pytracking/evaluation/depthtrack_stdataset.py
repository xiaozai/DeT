import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os

class DepthTrack_ST_Dataset(BaseDataset):
    """
    CDTB, RGB dataset, Depth dataset, Colormap dataset, RGB+depth
    """
    def __init__(self, dtype='colormap'):
        super().__init__()
        self.base_path = self.env_settings.depthtrack_st_path
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
        elif self.dtype == 'rgbd':
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
            depth_frames = ['{base_path}/{sequence_path}/depth/{frame:0{nz}}.png'.format(base_path=self.base_path,
                            sequence_path=sequence_path, frame=frame_num, nz=nz)
                            for frame_num in range(start_frame, end_frame+1)]
            color_frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.jpg'.format(base_path=self.base_path,
                            sequence_path=sequence_path, frame=frame_num, nz=nz)
                            for frame_num in range(start_frame, end_frame+1)]
            # frames = {'color': color_frames, 'depth': depth_frames}
            frames = []
            for c_path, d_path in zip(color_frames, depth_frames):
                frames.append({'color': c_path, 'depth': d_path})

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

        return Sequence(sequence_name, frames, 'depthtrack', ground_truth_rect, dtype=self.dtype)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = os.listdir(self.base_path)
        try:
            sequence_list.remove('list.txt')
        except:
            pass

        return sequence_list
