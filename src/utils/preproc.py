"""Description
   -----------
This script includes functions that are related to datasets preparation and preprocessing
"""

import argparse
import csv
import multiprocessing
import ntpath
import os
import shutil
import time
import glob
import cv2
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=' ')
parser.add_argument("--data_path",
                    default="/mnt/4bb99913-1249-4f5e-834f-38ea9a27b11f/datasets/cholec80")
parser.add_argument('--cholec80', action='store_true')
parser.add_argument('--heichole', action='store_true')
parser.add_argument('--heichole_tool_annotation_suffix',
                    help="Tool Annotation file suffix. Please check the suffix in the current dataset."
                         "Available options: _annotation_instrument.csv, "
                         "_annotation_instrument_detailed.csv")
parser.add_argument('--tools', action='store_true', help="If set, the tool annotation is processed")
parser.add_argument('--phases', action='store_true', help="If set, the phase annotation is processed")
parser.add_argument('--images', action='store_true', help="If set, videos are split in images")
parser.add_argument('--binary_class', action='store_true',
                    help="If set, data is prepared for binary tool classification.")
parser.add_argument('--multilabel_class', action='store_true',
                    help="If set, data is prepared for multi label tool classification.")
parser.add_argument('--split_sort', action='store_true', help="If set, videos are split in images. The images are "
                                                              "sorted in folders. ")
parser.add_argument("--sort_class_wise",
                    default=True, help="frames are sorted classwise, be careful if it is true set video_wise to false")
parser.add_argument("--sort_video_wise",
                    default=False, help="frames are sorted videowise, be careful if it is true set sort_wise to false")
parser.add_argument("--nthframe", type=int,
                    default=25, help="specify the nth frame to be stored")
parser.add_argument("--number_of_cores_not_use", type=int,
                    default=5)
parser.add_argument("--video_ending",
                    default=".mp4")
parser.add_argument('--use1fps', action='store_true', help="video is split in 1 frame per second")

args = parser.parse_args()
annotation_tool_path = os.path.join(args.data_path, "tool_annotations")
annotation_phase_path = os.path.join(args.data_path, "phase_annotations")
video_data_path = os.path.join(args.data_path, "videos")
path_to_store_frames = os.path.join(args.data_path, "img+gt")
path_to_store_train_frames = os.path.join(args.data_path, "frames_tools_train_2")

# -------------------------------- classes
phases = ["00_Preparation", "01_CalotTriangleDissection", "02_ClippingCutting", "03_GallbladderDissection",
          "04_GallbladderPackaging", "05_CleaningCoagulation", "06_GallbladderRetraction"]
tools_cholec80 = ["00_Grasper", "01_Bipolar", "02_Hook", "03_Scissors", "04_Clipper",
                  "05_Irrigator", "06_SpecimenBag"]
detailed_tools_heichole = ["00_Curved atraumatic grasper", "01_Toothed grasper", "02_Fenestrated toothed grasper",
                           "03_Atraumatic grasper",
                           "04_Overholt", "05_LigaSure", "06_Electric hook", "07_Scissors", "08_Clip-applier (metal)",
                           "09_Clip-applier (Hem-O-Lok)", "10_Swab grasper", "11_Argon beamer", "12_Suction-irrigation",
                           "13_Specimen bag",
                           "14_Tiger mouth forceps", "15_Claw forceps", "16_Atraumatic grasper short",
                           "17_Crocodile forceps", "18_Flat grasper",
                           "19_Pointed forceps", "20_Stapler", "21_Reserved", "22_Reserved", "23_Reserved",
                           "24_Reserved", "25_Reserved", "26_Reserved", "27_Reserved", "28_Reserved", "29_Reserved",
                           "30_Undefined instrument shaft"]
tools_category_heichole = ["00_Grasper", "01_Clipper", "02_CoagulationInstruments", "03_Scissors",
                           "04_SuctionIrrigation", "05_SpecimenBag", "06_Stapler"]


def read_phase_cholec80(path):
    """
       Reads phase annotation file and returns content as dictionary
       :param path: path to annotation file
       :return: data, dict containing annotation information
       """
    data = pd.read_csv(path, sep='\t')
    data = data.transpose().to_dict()
    return data


def read_tool_cholec80(path):
    """
        Reads tool annotation file and returns content as dictionary
        :param path: path to annotation file
        :return: data, dict containing annotation information
        """
    data = pd.read_csv(path, sep='\t')
    data = data.to_dict()
    return data


def split_and_sort_videos_by_phases_cholec80(video_path):
    """
         Splits video into single frames based on opencv frame count and sorts frames either class_wise or video_wise
        :param video_path path to video
        :return:
         """
    time_start = time.time()
    video_name_without_ending = ntpath.basename(video_path).split(".mp4")[0]
    video_number = video_name_without_ending.split("video")[1]
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    data = read_phase_cholec80(os.path.join(annotation_phase_path, video_name_without_ending + "-phase.txt"))
    if args.sort_class_wise:
        create_sub_folders_for_classes(path_to_store_frames, phases)
    new_frame_path = ''
    if args.sort_class_wise:
        new_frame_path = os.path.join(path_to_store_frames, '%s', "c080." + f"{int(video_number):03d}" +
                                      '.%s.png')

    if args.sort_video_wise:
        if not os.path.exists(os.path.join(path_to_store_frames, video_name_without_ending)):
            os.makedirs(os.path.join(path_to_store_frames, video_name_without_ending))
        new_frame_path = os.path.join(path_to_store_frames, video_name_without_ending,
                                      "c080." + f"{int(video_number):03d}" + '.%s.png')

    counter = 0
    print("Number of frames in video number: " + video_name_without_ending, frame_count)
    success = True
    while success:
        success, frame = video_capture.read()
        if args.use1fps:
            if counter % fps == 0:
                frame_name = ''
                if args.sort_class_wise:
                    phase = data[counter]["Phase"]
                    for j in range(len(phases)):
                        if phase == phases[j][3:]:
                            frame_name = new_frame_path % (phases[j], f"{counter:06d}")
                            break
                if args.sort_video_wise:
                    frame_name = new_frame_path % f"{counter:06d}"

                cv2.imwrite(frame_name, frame)
        counter += 1
        if counter >= frame_count:
            time_end = time.time()
            video_capture.release()
            print(video_name_without_ending + " took %d minutes for conversion." % ((time_end - time_start) / 60))
            break


def split_video_in_frames(video_path):
    """
    Splits video into single frames based on opencv frame count
    :param video_path path to video
     :return:
    """
    time_start = time.time()
    video_name_without_ending = ntpath.basename(video_path).split(args.video_ending)[0]
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    if not os.path.exists(os.path.join(path_to_store_frames, video_name_without_ending)):
        os.makedirs(os.path.join(path_to_store_frames, video_name_without_ending))

    if args.cholec80:
        video_number = video_name_without_ending.split("video")[1]
        new_frame_path = os.path.join(path_to_store_frames, video_name_without_ending,
                                      "c080." + f"{int(video_number):03d}" + '.%s.png')
    elif args.heichole:
        video_number = video_name_without_ending.split("hei-chole")[1]
        new_frame_path = os.path.join(path_to_store_frames, video_name_without_ending,
                                      "h19_c" + f"{int(video_number):02d}" + '_%s.png')
    else:
        new_frame_path = os.path.join(path_to_store_frames, video_name_without_ending,
                                      video_name_without_ending + "." + '%s.png')
    counter = 0
    print("Number of frames in video number: " + video_name_without_ending, frame_count)
    success = True
    while success:
        success, frame = video_capture.read()
        if args.use1fps:
            if counter % fps == 0:
                cv2.imwrite(new_frame_path % f"{counter:06d}", frame)
        else:
            cv2.imwrite(new_frame_path % f"{counter:06d}", frame)
        counter += 1
        if counter >= frame_count:
            time_end = time.time()
            video_capture.release()
            print(video_name_without_ending + " took %d minutes for conversion." % ((time_end - time_start) / 60))
            break


def split_and_sort_frames_for_tools_cholec80(video_path):
    """
    Splits video into single frames based on opencv frame count and sorts frames, if multiple
    tools are annotated frames are stored in both folders
    :param video_path path to video
    :return:
    """

    time_start = time.time()
    video_name_without_ending = ntpath.basename(video_path).split(".mp4")[0]
    video_number = video_name_without_ending.split("video")[1]
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    data = read_tool_cholec80(os.path.join(annotation_tool_path, video_name_without_ending + "-tool.txt"))
    num_of_annotated_frames = int(sum(map(len, data.values())) / 8)

    if args.binary_class:
        if not os.path.exists(os.path.join(path_to_store_frames, "01_Tool/")):
            os.makedirs(os.path.join(path_to_store_frames, "01_Tool/"))

        if not os.path.exists(os.path.join(path_to_store_frames, "02_NoTool/")):
            os.makedirs(os.path.join(path_to_store_frames, "02_NoTool/"))
    else:
        create_sub_folders_for_classes(path_to_store_frames, tools_cholec80)
        if not os.path.exists(os.path.join(path_to_store_frames, "07_NoTool/")):
            os.makedirs(os.path.join(path_to_store_frames, "07_NoTool/"))

    new_frame_path = os.path.join(path_to_store_frames, '%s', "c080." + f"{int(video_number):03d}" +
                                  '.%s.png')

    counter = 0
    print("Number of frames in video number: " + video_name_without_ending, frame_count)
    success = True
    while success:
        success, frame = video_capture.read()
        # tool annotations only available for 1 FPS
        if counter % fps == 0:
            data_index = int(counter / 25)
            # check if data_index is existing in the annotation files and make sure it is the right data index
            if data_index < num_of_annotated_frames:
                if data["Frame"][data_index] == counter:
                    for j in range(len(tools_cholec80)):
                        if data[tools_cholec80[j][3:]][data_index] == 1:
                            if args.binary_class:
                                frame_name = new_frame_path % ("01_Tool", f"{counter:06d}")
                            else:
                                frame_name = new_frame_path % (tools_cholec80[j], f"{counter:06d}")
                            # since a frame could contain multiple instruments write it immediately down if it is found
                            cv2.imwrite(frame_name, frame)
                    if 0 in {
                        data["Grasper"][data_index] or data["Bipolar"][data_index] or data["Hook"][data_index] or
                        data["Scissors"][data_index] or
                        data["Clipper"][
                            data_index] or data["Irrigator"][data_index] or data["SpecimenBag"][data_index]}:
                        if args.binary_class:
                            frame_name = new_frame_path % ("02_NoTool", f"{counter:06d}")
                        else:
                            frame_name = new_frame_path % ("07_NoTool", f"{counter:06d}")
                        cv2.imwrite(frame_name, frame)
        counter += 1
        if counter >= frame_count:
            time_end = time.time()
            video_capture.release()
            print(video_name_without_ending + " took %d minutes for conversion." % ((time_end - time_start) / 60))
            break


def split_videos_by_tools_cholec80_for_multi_label_class(video_path):
    """
        Splits video into single frames based on opencv frame count and creates txt for file for each frame containing
        the annotation information to use for multi label classification
        :param video_path path to video
        :return:
    """
    time_start = time.time()
    video_name_without_ending = ntpath.basename(video_path).split(".mp4")[0]
    video_number = video_name_without_ending.split("video")[1]
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    data = read_tool_cholec80(os.path.join(annotation_tool_path, video_name_without_ending + "-tool.txt"))
    num_of_annotated_frames = int(sum(map(len, data.values())) / 8)

    new_frame_path = os.path.join(path_to_store_frames,
                                  "c080." + f"{int(video_number):03d}" + '.%s.png')
    new_annotation_file_path = os.path.join(path_to_store_frames,
                                            "c080." + f"{int(video_number):03d}" + '.%s.txt')
    counter = 0
    print("Number of frames in video number: " + video_name_without_ending, frame_count)
    success = True
    while success:
        success, frame = video_capture.read()
        if args.use1fps:
            if counter % fps == 0:
                data_index = int(counter / 25)
                # check if data_index is existing in the annotation files and make sure it is the right data index
                if data_index < num_of_annotated_frames:
                    if data["Frame"][data_index] == counter:
                        line = ""
                        for j in range(len(tools_cholec80)):
                            line += str(data[tools_cholec80[j][3:]][data_index]) + ","
                        line = line[:-1]
                        annotation_txt_file_name = new_annotation_file_path % f"{counter:06d}"
                        annotation_txt_file = open(annotation_txt_file_name, "w")
                        annotation_txt_file.write(line)
                        annotation_txt_file.close()
                        cv2.imwrite(new_frame_path % f"{counter:06d}", frame)
        counter += 1
        if counter >= frame_count:
            time_end = time.time()
            video_capture.release()
            print(video_name_without_ending + " took %d minutes for conversion." % ((time_end - time_start) / 60))
            break


def create_sub_folders_for_classes(classpath, classes):
    """
           Creates subfolder for each class in classes under given class path
           :param classpath path where the subfolder should be created
           :param classes, list of classes
           :return:
       """
    for item in classes:
        classpath = os.path.abspath(classpath)
        if not os.path.exists(classpath):
            try:
                os.makedirs(classpath)
            except OSError as exception:
                print("Could not create folder: \n" + str(classpath) + " because of \n" + str(exception))
        if not os.path.exists(os.path.join(classpath, item)):
            os.makedirs(os.path.join(classpath, item))


def read_phase_heichole(path):
    """
          Reads phase annotation file and returns content as dictionary
          :param path: path to annotation file
          :return: data, dict containing annotation information
    """
    data = pd.read_csv(path, sep=',', header=None)
    data = data.to_dict()
    return data


def split_and_sort_videos_by_phases_heichole(video_path):
    """
        Splits video into single frames based on opencv frame count and sorts frames either class_wise or video_wise
        (working only for phase annotation)
        :param video_path path to video
        :return:
        """
    time_start = time.time()
    video_name_without_ending = ntpath.basename(video_path).split(args.video_ending)[0]
    video_number = video_name_without_ending.split("hei-chole")[1]
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    data = read_phase_heichole(
        os.path.join(annotation_phase_path, video_name_without_ending + "_annotation_phase.csv"))
    index_dict_data_frame = 1
    if args.sort_class_wise:
        create_sub_folders_for_classes(path_to_store_frames, phases)
    new_frame_path = ''
    if args.sort_class_wise:
        new_frame_path = os.path.join(path_to_store_frames, '%s', "h19_c" + f"{int(video_number):02d}" +
                                      '_%s.png')

    if args.sort_video_wise:
        if not os.path.exists(os.path.join(path_to_store_frames, video_name_without_ending)):
            os.makedirs(os.path.join(path_to_store_frames, video_name_without_ending))
        new_frame_path = os.path.join(path_to_store_frames, video_name_without_ending,
                                      "h19_c" + f"{int(video_number):02d}" + '_%s.png')

    counter = 0
    print("Number of frames in video number: " + video_name_without_ending, frame_count)
    success = True
    while success:
        success, frame = video_capture.read()
        if args.use1fps:
            if counter % fps == 0:
                frame_name = ''
                if args.sort_class_wise:
                    phase = data[index_dict_data_frame][counter]
                    for j in range(len(phases)):
                        if phase == int((phases[j].split('_')[0])):
                            frame_name = new_frame_path % (phases[j], f"{counter:08d}")
                            break
                if args.sort_video_wise:
                    frame_name = new_frame_path % f"{counter:08d}"
                cv2.imwrite(frame_name, frame)
        counter += 1
        if counter >= frame_count:
            time_end = time.time()
            video_capture.release()
            print(video_name_without_ending + " took %d minutes for conversion." % ((time_end - time_start) / 60))
            break


def read_tool_heichole(path):
    """
    Reads tool annotation file and returns content as dictionary
      :param path: path to annotation file
      :return: data, dict containing annotation information
    """
    data = pd.read_csv(path, sep=',', header=None)
    data = data.transpose()
    data = data.to_dict()
    return data


def read_tool_heichole_as_list(summary_path):
    """ Reads tool annotation file and returns content as lists
        :param summary_path: path to summary file
        :return: labels, frames
      """
    labels = []
    frames = []
    with open(summary_path, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            frames.append(row[0])
            labels.append(list(map(int, row[1].split(','))))

    return labels, frames


def generate_dataset_summary(path):
    """ Generate a summary.csv for multi-label-classification based on images and annotation files in path.
          :param path: path to data
          :return:
        """
    data_path = os.path.join(path)
    gt_paths = sorted(glob.glob(data_path + '/*.txt'))
    gts, filenames = [], []
    for i in range(len(gt_paths)):
        print('\r%d / %d' % (i + 1, len(gt_paths)), end="")
        with open(gt_paths[i]) as f:
            line = f.readlines()
        f.close()
        gts.append(line[0])
        file_name = ntpath.basename(gt_paths[i])
        filenames.append(file_name)

    file = open(path + "/summary.csv", 'w+', newline='')
    with file:
        write = csv.writer(file, delimiter=',')
        write.writerows(zip(filenames, gts))


def create_balanced_train_dataset(summary_path):
    """ Generates a balanced dataset for multi-label-classification for cholec80 or heichole use case.
        :param summary_path: path to summary file
        :return:
    """
    labels, frames = read_tool_heichole_as_list(summary_path)

    def sample_no_tool_frames(N_keep, _labels, _frames):
        target_labels, target_frames = [], []
        for i in range(len(_labels)):
            if np.sum(_labels[i]) == 0:
                target_labels.append(_labels[i])
                target_frames.append(_frames[i])
        print("\n number of empty frames: ", len(target_frames))
        idx_spacing = np.round(np.linspace(0, len(target_frames) - 1, N_keep)).astype(int)
        keep_frames = [target_frames[i] for i in idx_spacing]
        counter = 0
        for i in reversed(range(len(_frames))):
            if _frames[i] in target_frames and _frames[i] not in keep_frames:
                _frames.pop(i)
                _labels.pop(i)
                counter += 1
                print('\r Empty frames removed %d' % (counter), end="")
        print("\nRemaining empty frames: ", count_empty_frames(_labels))
        return _labels, _frames

    def get_tool_combi(tool_index, _labels, _frames):
        combinations_labels, combinations_frames = [], []
        i = 0
        for label, frame in zip(_labels, _frames):
            print('\r processing label combinations %d / %d for tool %d' % (i + 1, len(_labels), tool_index), end="")
            if np.sum(label) > 1 and label[tool_index] == 1:
                combinations_labels.append(label)
                combinations_frames.append(frame)
            i += 1
        print('\ntotal combinations for tool %d = %d\t' % (tool_index, len(combinations_labels)))
        print(np.sum(combinations_labels, axis=0))
        return combinations_labels, combinations_frames

    def get_tool_single(tool_index, _labels, _frames):
        single_labels, single_frames = [], []
        i = 0
        for label, frame in zip(_labels, _frames):
            print('\r processing single labels %d / %d for tool %d' % (i + 1, len(_labels), tool_index), end="")
            if np.sum(label) == 1 and label[tool_index] == 1:
                single_labels.append(label)
                single_frames.append(frame)
            i += 1
        print('\ntotal single labels for tool %d = %d\n' % (tool_index, len(single_labels)))
        return single_labels, single_frames

    def copy_frames_from_source(_frames, data_path, dest_path):
        """ Copies frames of given data path to train data path.
            :param data_path, path to data (frames and annotation txt files)
            :param dest_path, path to store training data
            :return
        """

        dest_path = os.path.join(dest_path, "img+gt")
        # data_path = os.path.join(data_path, "img+gt")
        if not os.path.exists(os.path.join(dest_path)):
            os.makedirs(os.path.join(dest_path))
        print("num of all frame names to copy : " + str(len(_frames)))
        for i in range(len(_frames)):
            print('\r Copying frame %d / %d' % (i + 1, len(_frames)), end="")
            shutil.copy(os.path.join(data_path, _frames[i][:-4] + ".txt"),
                        os.path.join(dest_path, _frames[i][:-4] + ".txt"))
            shutil.copy(os.path.join(data_path, _frames[i]), os.path.join(dest_path, _frames[i]))

    def count_single_label(label_index, _labels):
        counter = 0
        for label in _labels:
            if label[label_index] == 1 and np.sum(label) == 1:
                counter += 1
        return counter

    def count_label_combi(label_index, _labels, target=None):
        if target is None:
            counter = [0, 0, 0, 0, 0, 0, 0]
            for label in _labels:
                if label[label_index] == 1:
                    counter += np.asarray(label)
        else:
            counter = 0
            tmp = np.zeros(len(_labels[0]))
            tmp[label_index] = tmp[target] = 1
            for label in _labels:
                if (tmp == label).all():
                    counter += 1
        return counter

    def count_empty_frames(_labels):
        counter = 0
        for label in _labels:
            if np.sum(label) == 0:
                counter += 1
        return counter

    def remove_single_labels_from_summary_list(N_keep, tool_index, _labels, _frames, target_frames):
        idx_spacing = np.round(np.linspace(0, len(target_frames) - 1, N_keep)).astype(int)
        tmp_frames = [target_frames[i] for i in idx_spacing]
        counter = 0

        for i in reversed(range(len(_frames))):
            if _frames[i] in target_frames and _frames[i] not in tmp_frames:
                _frames.pop(i)
                _labels.pop(i)
                counter += 1
                print('\r frames removed containing single label %d = %d' % (tool_index, counter), end="")
        print('\nremaining frames of single lable %d = %d\n' % (tool_index, count_single_label(tool_index, _labels)))

        return _labels, _frames

    def remove_combi_labels_from_summary_list(N_keep, tool_index, _labels, _frames, target_frames, target_labels,
                                              comb_index):
        # exclude frames where the target comb_index is combined with other labels
        tmp = np.zeros(len(_labels[0]))
        tmp[tool_index] = tmp[comb_index] = int(1)
        for i in reversed(range(len(target_labels))):
            if not (tmp == target_labels[i]).all():
                target_labels.pop(i)
                target_frames.pop(i)
        idx_spacing = np.round(np.linspace(0, len(target_frames) - 1, N_keep)).astype(int)
        keep_frames = [target_frames[i] for i in idx_spacing]
        counter = 0
        for i in reversed(range(len(_frames))):
            if _frames[i] in target_frames and _frames[i] not in keep_frames:
                _frames.pop(i)
                _labels.pop(i)
                counter += 1
                print('\r frames removed containing labels %d and %d = %d' % (tool_index, comb_index, counter), end="")
        print('\nremaining combinations for tool ', tool_index, ':   ', count_label_combi(tool_index, _labels))
        return _labels, _frames

    print("Labels: 00_Grasper | 01_Bipolar | 02_Hook | 03_Scissors | 04_Clipper | 05_Irrigator | 06_SpecimenBag")
    print("-------------\nStage_0\nCurrent dataset labels: ", np.sum(labels, axis=0), "\n Number of frames: ",
          len(frames),
          "\nNumber of empty frames", count_empty_frames(labels), "\n-------------\n")
    # optimize hook
    single_hooks_labels, single_hooks_frames = get_tool_single(2, labels, frames)
    hook_combinations_labels, hook_combinations_frames = get_tool_combi(2, labels, frames)
    labels, frames = remove_single_labels_from_summary_list(1000, 2, labels, frames, single_hooks_frames)
    labels, frames = remove_combi_labels_from_summary_list(1000, 2, labels, frames, hook_combinations_frames,
                                                           hook_combinations_labels, 0)

    print("-------------\nStage_1\nCurrent dataset labels: ", np.sum(labels, axis=0), "\n Number of frames: ",
          len(frames), "\n-------------\n")

    # optimize grasper
    graspers_combinations_labels, graspers_combinations_frames = get_tool_combi(0, labels, frames)
    single_grasper_labels, single_graspers_frames = get_tool_single(0, labels, frames)
    labels, frames = remove_single_labels_from_summary_list(1000, 0, labels, frames, single_graspers_frames)
    labels, frames = remove_combi_labels_from_summary_list(1000, 0, labels, frames, graspers_combinations_frames,
                                                           graspers_combinations_labels, 6)
    graspers_combinations_labels, graspers_combinations_frames = get_tool_combi(0, labels, frames)
    labels, frames = remove_combi_labels_from_summary_list(1000, 0, labels, frames, graspers_combinations_frames,
                                                           graspers_combinations_labels, 5)
    graspers_combinations_labels, graspers_combinations_frames = get_tool_combi(0, labels, frames)
    labels, frames = remove_combi_labels_from_summary_list(1000, 0, labels, frames, graspers_combinations_frames,
                                                           graspers_combinations_labels, 1)
    print("-------------\nStage_2\nCurrent dataset labels: ", np.sum(labels, axis=0), "\n Number of frames: ",
          len(frames), "\n-------------\n")

    labels, frames = sample_no_tool_frames(1000, labels, frames)
    print("\n-------------\nStage_3\nCurrent dataset labels: ", np.sum(labels, axis=0), "\n Number of frames: ",
          len(frames), "\n-------------\n")
    # copy
    copy_frames_from_source(frames, path_to_store_frames, path_to_store_train_frames)


def split_and_sort_videos_by_tools_heichole(video_path):
    """ Splits video into single frames based on opencv frame count and sorts frames. A frame with multiple instrument
    annotation, will be sorted in multiple folders.
    :param video_path path to video
    :return
    """
    time_start = time.time()
    video_name_without_ending = ntpath.basename(video_path).split(args.video_ending)[0]
    video_number = video_name_without_ending.split("hei-chole")[1]
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    data = pd.read_csv(
        os.path.join(annotation_tool_path, video_name_without_ending + args.heichole_tool_annotation_suffix), sep=',',
        header=None)
    data = data.to_dict()
    labels_to_use = []
    if args.heichole_tool_annotation_suffix == "_annotation_instrument_detailed.csv":
        labels_to_use = detailed_tools_heichole
        if not os.path.exists(os.path.join(path_to_store_frames, "31_NoTool")):
            os.makedirs(os.path.join(path_to_store_frames, "31_NoTool"))
    if args.heichole_tool_annotation_suffix == "_annotation_instrument.csv":
        labels_to_use = tools_category_heichole
        if not os.path.exists(os.path.join(path_to_store_frames, "07_NoTool")):
            os.makedirs(os.path.join(path_to_store_frames, "07_NoTool"))

    create_sub_folders_for_classes(path_to_store_frames, labels_to_use)
    new_frame_path = os.path.join(path_to_store_frames, '%s', "h19_c" + f"{int(video_number):02d}" +
                                  '_%s.png')
    counter = 0
    print("Number of frames in video number: " + video_name_without_ending, frame_count)
    success = True
    while success:
        success, frame = video_capture.read()
        if args.use1fps:
            if counter % fps == 0:
                frame_names_to_write = []
                found_tools = 0
                for j in range(len(labels_to_use)):
                    tool = data[j + 1][counter]
                    if tool == 1:
                        found_tools += 1
                        frame_name = new_frame_path % (labels_to_use[j], f"{counter:08d}")
                        frame_names_to_write.append(frame_name)
                if found_tools == 0:
                    if args.heichole_tool_annotation_suffix == "_annotation_instrument_detailed.csv":
                        frame_name = new_frame_path % ("31_NoTool", f"{counter:08d}")
                    else:
                        frame_name = new_frame_path % ("07_NoTool", f"{counter:08d}")
                    frame_names_to_write.append(frame_name)

                [cv2.imwrite(f, frame) for f in frame_names_to_write]
        counter += 1
        if counter >= frame_count:
            time_end = time.time()
            video_capture.release()
            print(video_name_without_ending + " took %d minutes for conversion." % ((time_end - time_start) / 60))
            break


def split_videos_by_tools_heichole_for_multi_label_class(video_path):
    """ Splits video into single frames based on opencv frame count and generate txt file containing annotations for
    each frame.
    :param video_path path to video
    :return
    """
    time_start = time.time()
    video_name_without_ending = ntpath.basename(video_path).split(args.video_ending)[0]
    video_number = video_name_without_ending.split("hei-chole")[1]
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    data = read_tool_heichole(
        os.path.join(annotation_tool_path, video_name_without_ending + args.heichole_tool_annotation_suffix))
    if not os.path.exists(os.path.join(path_to_store_frames)):
        os.makedirs(os.path.join(path_to_store_frames))
    new_frame_path = os.path.join(path_to_store_frames,
                                  "h19_c" + f"{int(video_number):02d}" + '_%s.png')
    new_annotation_file_path = os.path.join(path_to_store_frames,
                                            "h19_c" + f"{int(video_number):02d}" + '_%s.txt')

    counter = 0
    print("Number of frames in video number: " + video_name_without_ending, frame_count)
    success = True
    while success:
        success, frame = video_capture.read()
        if args.use1fps:
            if counter % fps == 0:
                line = ""
                if args.heichole_tool_annotation_suffix == "_annotation_instrument_detailed.csv":
                    for i in range(1, len(data[counter]), 1):
                        line += str(data[counter][i]) + ","
                if args.heichole_tool_annotation_suffix == "_annotation_instrument.csv":
                    for i in range(1, len(tools_category_heichole) + 1, 1):
                        line += str(data[counter][i]) + ","
                line = line[:-1]
                frame_name = new_frame_path % f"{counter:08d}"
                cv2.imwrite(frame_name, frame)
                annotation_txt_file_name = new_annotation_file_path % f"{counter:08d}"
                annotation_txt_file = open(annotation_txt_file_name, "w")
                annotation_txt_file.write(line)
                annotation_txt_file.close()
        counter += 1
        if counter >= frame_count:
            time_end = time.time()
            video_capture.release()
            print(video_name_without_ending + " took %d minutes for conversion." % ((time_end - time_start) / 60))
            break


def split_videos_by_toolcategory_heichole_for_multi_label_class_grasper_coag(video_path):
    """ Splits video into single frames based on opencv frame count and generate txt file containing tool category
        annotation for each frame. Only the categories grasper and coagulation instruments are considerd.
      :param video_path path to video
      :return:
      """
    time_start = time.time()
    video_name_without_ending = ntpath.basename(video_path).split(args.video_ending)[0]
    video_number = video_name_without_ending.split("hei-chole")[1]
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    data = read_tool_heichole(
        os.path.join(annotation_tool_path, video_name_without_ending + "_annotation_instrument.csv"))

    if not os.path.exists(os.path.join(path_to_store_frames, video_name_without_ending)):
        os.makedirs(os.path.join(path_to_store_frames, video_name_without_ending))
    new_frame_path = os.path.join(path_to_store_frames, video_name_without_ending,
                                  "h19_c" + f"{int(video_number):02d}" + '_%s.png')
    new_annotation_file_path = os.path.join(path_to_store_frames, video_name_without_ending,
                                            "h19_c" + f"{int(video_number):02d}" + '_%s.txt')
    grasper_data_index = 1
    coagulation_instruments_data_index = 3
    counter = 0
    print("Number of frames in video number: " + video_name_without_ending, frame_count)
    success = True
    while success:
        success, frame = video_capture.read()
        if args.use1fps:
            if counter % fps == 0:
                line = ""
                if data[counter][1] == 1 or data[counter][3] == 1:
                    all_tool_annot = []
                    for i in range(1, len(tools_category_heichole) + 1, 1):
                        all_tool_annot.append(int(data[counter][i]))
                    # take only frames which contains grasper or coagulation instruments, or both
                    if np.sum(all_tool_annot) == 1 or np.sum(all_tool_annot) == 2:
                        line += str(data[counter][grasper_data_index]) + ","
                        line += str(data[counter][coagulation_instruments_data_index]) + ","
                        line = line[:-1]
                        frame_name = new_frame_path % f"{counter:08d}"
                        cv2.imwrite(frame_name, frame)
                        annotation_txt_file_name = new_annotation_file_path % f"{counter:08d}"
                        annotation_txt_file = open(annotation_txt_file_name, "w")
                        annotation_txt_file.write(line)
                        annotation_txt_file.close()
        counter += 1
        if counter >= frame_count:
            time_end = time.time()
            video_capture.release()
            print(video_name_without_ending + " took %d minutes for conversion." % ((time_end - time_start) / 60))
            break


if __name__ == '__main__':
    cpu_count = multiprocessing.cpu_count()
    if cpu_count > args.number_of_cores_not_use:
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
        video_list = sorted([f for f in os.listdir(video_data_path) if f.endswith(args.video_ending)])
        video_list = sorted([os.path.join(video_data_path, v) for v in video_list])
        if args.cholec80:
            if args.phases:
                if args.split_sort:
                    pool.map(split_and_sort_videos_by_phases_cholec80, video_list)
            if args.tools:

                if args.multilabel_class:
                    pool.map(split_videos_by_tools_cholec80_for_multi_label_class, video_list)
                    generate_dataset_summary(path_to_store_frames)
                    create_balanced_train_dataset(summary_path=path_to_store_frames + '/summary.csv')
                if args.split_sort:
                    pool.map(split_and_sort_frames_for_tools_cholec80, video_list)

            if args.images:
                pool.map(split_video_in_frames, video_list)
            print("All finished...")

        elif args.heichole:
            if args.phases:
                if args.split_sort:
                    pool.map(split_and_sort_videos_by_phases_heichole, video_list)

            if args.tools:
                if args.split_sort:
                    pool.map(split_and_sort_videos_by_tools_heichole, video_list)
                if args.multilabel_class:
                    pool.map(split_videos_by_tools_heichole_for_multi_label_class, video_list)
                    generate_dataset_summary(path_to_store_frames)
                    create_balanced_train_dataset(summary_path=path_to_store_frames + '/summary.csv')

            print("All finished...")

        else:
            if args.image:
                pool.map(split_video_in_frames, video_list)
            print("All finished...")
    else:
        print("Argument \"number_of_cores_not_use\" is too high. Please modify.")
