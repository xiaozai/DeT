import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from pytracking.evaluation import Sequence, Tracker
from ltr.data.image_loader import imwrite_indexed
import cv2
from matplotlib import pyplot as plt

def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict, run_id=None):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        os.makedirs(tracker.results_dir)

    base_results_path = os.path.join(tracker.results_dir, seq.name)
    if not os.path.exists(base_results_path):
        os.makedirs(base_results_path)

    segmentation_path = os.path.join(tracker.segmentation_dir, seq.name)

    if type(seq.frames[0]) is dict:
        frame_names = [os.path.splitext(os.path.basename(f['color']))[0] for f in seq.frames]
    else:
        frame_names = [os.path.splitext(os.path.basename(f))[0] for f in seq.frames]

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        # np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')
        np.savetxt(file, tracked_bb, delimiter=',', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_confidence(file, data):
        confidence = np.array(data).astype(float)
        np.savetxt(file, confidence, delimiter='\t', fmt='%f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    # bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    if run_id:
                        bbox_file = '{}/{}_{}_{:03}.txt'.format(base_results_path, seq.name, obj_id,run_id)
                    else:
                        bbox_file = '{}/{}_{}_001.txt'.format(base_results_path, seq.name, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                # bbox_file = '{}.txt'.format(base_results_path)
                if run_id:
                    bbox_file = '{}/{}_{:03}.txt'.format(base_results_path, seq.name, run_id)
                else:
                    bbox_file = '{}/{}_001.txt'.format(base_results_path, seq.name)
                save_bb(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    # timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    if run_id:
                        timings_file = '{}/{}_{}_{:03}_time.value'.format(base_results_path, seq.name, obj_id, run_id)
                    else:
                        timings_file = '{}/{}_{}_001_time.value'.format(base_results_path, seq.name, obj_id)
                    save_time(timings_file, d)
            else:
                # timings_file = '{}_time.txt'.format(base_results_path)
                if run_id:
                    timings_file = '{}/{}_{:03}_time.value'.format(base_results_path, seq.name, run_id)
                else:
                    timings_file = '{}/{}_001_time.value'.format(base_results_path, seq.name)
                save_time(timings_file, data)

        elif key == 'confidence':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    # confidence_file = '{}_{}_confidence.txt'.format(base_results_path, obj_id)
                    if run_id:
                        confidence_file = '{}/{}_{}_{:03}_confidence.value'.format(base_results_path, seq.name, obj_id, run_id)
                    else:
                        confidence_file = '{}/{}_{}_001_confidence.value'.format(base_results_path, seq.name, obj_id)
                    save_time(confidence_file, d)
            else:
                # confidence_file = '{}_confidence.txt'.format(base_results_path)
                if run_id:
                    confidence_file = '{}/{}_{:03}_confidence.value'.format(base_results_path, seq.name, run_id)
                else:
                    confidence_file = '{}/{}_001_confidence.value'.format(base_results_path, seq.name)
                save_time(confidence_file, data)

        elif key == 'segmentation':
            assert len(frame_names) == len(data)
            if not os.path.exists(segmentation_path):
                os.makedirs(segmentation_path)
            for frame_name, frame_seg in zip(frame_names, data):
                imwrite_indexed(os.path.join(segmentation_path, '{}.png'.format(frame_name)), frame_seg)

def run_sequence(seq: Sequence, tracker: Tracker, debug=False, visdom_info=None, run_id=None):
    """Runs a tracker on a sequence."""

    def _results_exist():
        if seq.object_ids is None:
            if run_id is None:
                bbox_file = '{}/{}/{}_001.txt'.format(tracker.results_dir, seq.name, seq.name)
            else:
                bbox_file = '{}/{}/{}_{:03}.txt'.format(tracker.results_dir, seq.name, seq.name, run_id)
            return os.path.isfile(bbox_file)
        else:
            if run_id is None:
                bbox_files = ['{}/{}/{}_{}_001.txt'.format(tracker.results_dir, seq.name, seq.name, obj_id) for obj_id in seq.object_ids]
            else:
                bbox_files = ['{}/{}/{}_{}_{:03}.txt'.format(tracker.results_dir, seq.name, seq.name, obj_id, run_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

            # bbox_file = '{}/{}/{}_{:03}.txt'.format(tracker.results_dir, seq.name, seq.name, run_id)
            # return os.path.isfile(bbox_file)

    visdom_info = {} if visdom_info is None else visdom_info

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
    else:
        try:
            output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output, run_id=run_id)


def run_dataset(dataset, trackers, debug=False, threads=0, visdom_info=None, run_id=None):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
        visdom_info: Dict containing information about the server for visdom

        run_id : Song added it, for bbox_file saving, seqname_{03}.txt!!!!
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    visdom_info = {} if visdom_info is None else visdom_info

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, visdom_info=visdom_info, run_id=run_id)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, visdom_info, run_id) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')
