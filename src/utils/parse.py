import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from ntpath import basename, dirname

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageColor

INSTRUMENTS_LABELS = list(np.arange(1))
fontscale, thickness = 1, 1


def load_xml(path_to_xml):
    """
    extract annotation attributes and tags from .xml file(s)
    :param path_to_xml: list containing path to single .xml file, multiple .xml files or single directory of .xml files
    :return: annotations, header, exclude_list, multi_xml_flag, file_names
    """
    header = ['task_id', 'frame_id', 'frame', 'smoke_level', 'lens_status', 's_mode', 'icg', 'instruments',
              'out_of_body']

    if len(path_to_xml) > 1:  # multiple .xmls passed
        multi_xml_flag = 1
        num_files = len(path_to_xml)
        annotations = np.zeros(num_files, dtype='O')
        exclude_list = [[]] * num_files
        for index, file in enumerate(path_to_xml):
            annotations[index], exclude_list[index] = parse_smoke_xml(os.path.join(file))
        file_names = [basename(p) for p in path_to_xml]

    else:
        if os.path.isdir(path_to_xml[0]):  # one directory
            multi_xml_flag = 1
            num_files = len(os.listdir(path_to_xml[0]))
            annotations = np.zeros(num_files, dtype='O')
            exclude_list = [[]] * num_files
            for index, file in enumerate(os.listdir(path_to_xml[0])):
                annotations[index], exclude_list[index] = parse_smoke_xml(os.path.join(path_to_xml[0], file))
            file_names = os.listdir(path_to_xml[0])
        else:  # one .xml
            multi_xml_flag = 0
            annotations, exclude_list = parse_smoke_xml(path_to_xml[0])
            annotations, exclude_list = [annotations], [exclude_list]
            file_names = basename(path_to_xml[0])

    return annotations, header, exclude_list, multi_xml_flag, file_names


def parse_inout_xml(path_to_xml):
    tree = ET.parse(path_to_xml).getroot()
    annotations_numpy = np.full((int(tree[1][0][2].text), 4), -1, dtype='O')
    exclude_list = []
    outside_list = []
    task_id = tree[1][0][0].text
    idx = 0

    for sample in list(tree.findall('image')):
        label_flag = 0
        annotations_numpy[idx][0] = int(task_id)
        annotations_numpy[idx][1] = int(sample.attrib.get('id'))
        annotations_numpy[idx][2] = sample.attrib.get('name')
        for tag in sample.findall('tag'):
            if tag.attrib['label'] == 'Inside':
                annotations_numpy[idx][3] = 1
                label_flag += 1
            if tag.attrib['label'] == 'Outside':
                annotations_numpy[idx][3] = 0
                label_flag += 1

        # check missing labels
        if label_flag == 0:
            print('Sample %s, ID %s is missing in/out label' % (sample.attrib.get('name'), sample.attrib.get('id')))
            exclude_list.append(int(sample.attrib.get('id')))

        # check duplicated labels within one tag
        if label_flag > 1:
            print('Sample %s, ID %s contains duplicates' % (sample.attrib.get('name'), sample.attrib.get('id')))
            if sample.attrib.get('id') not in exclude_list:
                exclude_list.append(int(sample.attrib.get('id')))

        idx += 1

    # exclude annotations
    if exclude_list:
        for item in exclude_list:
            annotations_numpy = np.delete(annotations_numpy, np.where(annotations_numpy[:, 1] == item), axis=0)
            # print('Sample %s excluded' % item)


def parse_smoke_xml(path_to_xml):
    """
    convert .xml tree to a numpy object
    detect and exclude multiple or missing annotations within same frame
    assert that no frames have been annotated twice within the same task
    :param path_to_xml: path to .xml file
    :return: - numpy object summary
             - header for the numpy matrix
             - list of excluded files due to missing annotations or duplications
    """

    tree = ET.parse(path_to_xml).getroot()
    annotations_numpy = np.zeros((int(tree[1][0][2].text), 9), dtype='O')
    exclude_list = []
    task_id = tree[1][0][0].text
    idx = 0

    for sample in list(tree.findall('image')):
        label_flags = np.asarray([0, 0, 0, 0, 0, 0])
        annotations_numpy[idx][0] = int(task_id)
        annotations_numpy[idx][1] = int(sample.attrib.get('id'))
        annotations_numpy[idx][2] = sample.attrib.get('name')
        for tag in sample.findall('tag'):
            if tag.attrib['label'] == 'Smoke':
                annotations_numpy[idx][3] = tag.findall('attribute')[0].text
                label_flags[0] += 1
            elif tag.attrib['label'] == 'Meta-Data':
                for meta in tag.findall('attribute'):
                    if meta.attrib.get('name') == 'Lens Status':
                        annotations_numpy[idx][4] = meta.text
                        label_flags[1] += 1
                    elif meta.attrib.get('name') == 'S-Mode':
                        annotations_numpy[idx][5] = meta.text
                        label_flags[2] += 1
                    elif meta.attrib.get('name') == 'ICG':
                        annotations_numpy[idx][6] = meta.text
                        label_flags[3] += 1
                    elif meta.attrib.get('name') == 'Instruments':
                        annotations_numpy[idx][7] = int(meta.text)
                        label_flags[4] += 1
                    elif meta.attrib.get('name') == 'Out of body':
                        annotations_numpy[idx][8] = meta.text
                        label_flags[5] += 1

        # check missing labels
        if np.where(label_flags == 0)[0].size > 0:
            if 0 in np.where(label_flags == 0)[0]:
                print(
                    'Sample %s, ID %s is missing smoke level label' % (
                        sample.attrib.get('name'), sample.attrib.get('id')))
            else:
                print('Sample %s, ID %s is missing meta data' % (sample.attrib.get('name'), sample.attrib.get('id')))
            exclude_list.append(sample.attrib.get('id'))

        # check duplicated labels within one tag
        if np.count_nonzero(label_flags > 1):
            print('Sample %s, ID %s contains duplicates' % (sample.attrib.get('name'), sample.attrib.get('id')))
            if sample.attrib.get('id') not in exclude_list:
                exclude_list.append(sample.attrib.get('id'))

        idx += 1
    # exclude annotations
    if exclude_list:
        for item in exclude_list:
            annotations_numpy = np.delete(annotations_numpy, np.where(annotations_numpy[:, 1] == int(item)), axis=0)
            print('Sample %s excluded' % item)
    # sort according to ID
    annotations_numpy = annotations_numpy[annotations_numpy[:, 1].argsort()]

    # reassign instrument label
    global INSTRUMENTS_LABELS
    if np.max(annotations_numpy[:, 7]) > np.max(INSTRUMENTS_LABELS):
        INSTRUMENTS_LABELS = list(np.arange(np.max(annotations_numpy[:, 7]) + 1))

    # check if frames with identical name have been annotated twice
    frame_names = np.asarray([basename(name) for name in annotations_numpy[:, 2]])
    assert len(np.unique(frame_names)) == len(frame_names), "%s contains annotation duplicates" % path_to_xml

    return annotations_numpy, exclude_list


def parse_segmentation_xml(path_to_xml, overlay=0.0):
    """
    Create the gt masks according to the polygon annotation in the xml file

    :param path_to_xml: path to cvat-xml file containing polygon-annotation
    :param overlay: if >0, create directory with images overlay with gt
    :return:
    """

    def to_point_sequence(point_str):
        point_str_list = [xy_str.split(',') for xy_str in point_str.split(';')]
        point_list = [tuple(float(xy) for xy in xy_str) for xy_str in point_str_list]
        return point_list

    def remove_mask_border(image, mask):
        grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, border_mask = cv2.threshold(grey_img, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((9, 9), np.uint8)
        border_mask = cv2.morphologyEx(border_mask, cv2.MORPH_CLOSE, kernel)
        border_mask = cv2.morphologyEx(border_mask, cv2.MORPH_OPEN, kernel)

        if not border_mask.shape == mask.shape:
            border_mask = np.stack([border_mask, border_mask, border_mask], axis=2)

        result = mask.copy()
        result[border_mask == 0] = 0

        return result

    def save_mask(savedir, file, gt, img):
        mask = np.array(gt)
        if np.any(mask):
            mask = remove_mask_border(img, mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(savedir, file), mask)
        return mask

    def check_create_dir(parent_dir, dir_name):
        new_dir = os.path.join(parent_dir, dir_name)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        return new_dir

    print(path_to_xml)
    tree = ET.parse(path_to_xml).getroot()
    task_meta = tree[1][0]
    task_labels = task_meta.find('labels').findall('label')
    num_images = int(task_meta.find('size').text)

    class_to_id = {'background': 0, 'pTa': 1, 'pT1': 2, 'pT2': -1, 'pT3': -1, 'pT4': -1, 'pCIS': -1, 'NED': 3}

    task_colors = {}
    for _, label in enumerate(task_labels):  # masks with colors as in xml
        task_colors[label.find('name').text] = [ImageColor.getrgb(label.find('color').text)]
        task_colors[label.find('name').text].append(class_to_id[label.find('name').text])
    print(task_colors)

    gt_cvat_dir = check_create_dir(dirname(path_to_xml), 'gt_cvat')
    gt_train_dir = check_create_dir(dirname(path_to_xml), 'gt_train')
    path_to_imgs = os.path.join(dirname(path_to_xml), 'images')
    if overlay:
        gt_dir_overlay = check_create_dir(dirname(path_to_xml), 'overlay')

    dataset_meta = {'Label': [], 'Art': [], 'Bild': [], 'Kombinationen': []}
    for i, sample in enumerate(tree.findall('image')):
        print('\rProcessing image %d / %d' % (i + 1, num_images), end="")
        label = None
        filename = sample.attrib.get('name').split('.')[0] + '.jpg'
        height = int(sample.attrib.get('height'))
        width = int(sample.attrib.get('width'))

        frame_name = sample.attrib.get('name') if sample.attrib.get('name').endswith(
            tuple(['.jpg', '.JPG', '.png', '.PNG'])) else sample.attrib.get('name') + '.PNG'
        frame = cv2.imread(os.path.join(path_to_imgs, frame_name))

        img_cvat = Image.new('RGB', (width, height), 0)
        img_train = Image.new('L', (width, height), 0)
        polygon_meta = {'Label': [], 'Id': [], 'Art': [], 'Bild': [], 'Center': []}
        num_polygons = 0
        for polygon in sample.findall('polygon'):
            num_polygons += 1
            label = polygon.attrib.get('label')
            polygon_meta['Label'].append(label)
            dataset_meta['Label'].append(label)
            points = to_point_sequence(polygon.attrib.get('points'))
            ImageDraw.Draw(img_cvat).polygon(points, outline=task_colors[label][0], fill=task_colors[label][0])
            ImageDraw.Draw(img_train).polygon(points, outline=task_colors[label][1], fill=task_colors[label][1])
            polygon_meta['Center'].append(np.array(points).mean(axis=0).astype(int))
            for attr in polygon.findall('attribute'):
                if attr.attrib['name'] == 'Art':
                    polygon_meta['Art'].append(attr.text)
                    dataset_meta['Art'].append(attr.text)
                elif attr.attrib['name'] == 'Bild':
                    polygon_meta['Bild'].append(attr.text)
                    dataset_meta['Bild'].append(attr.text)
                    dataset_meta['Kombinationen'].append('-'.join([label, attr.text]))
                elif attr.attrib['name'] == 'Lesion':
                    polygon_meta['Id'].append(attr.text)
            if not polygon_meta['Id']:
                polygon_meta['Id'].append('1')

        if overlay and label is not None:
            gt_mask = save_mask(gt_cvat_dir, filename, img_cvat, frame)
            _ = save_mask(gt_train_dir, filename, img_train, frame)
            overlay_img = cv2.addWeighted(frame, 1, gt_mask, overlay, 0)
            for i in range(num_polygons):
                text = "Polygon %d: %s, %s, %s" % (int(polygon_meta['Id'][i]), polygon_meta['Label'][i],
                                                   polygon_meta['Art'][i], polygon_meta['Bild'][i])
                (_, dy), _ = cv2.getTextSize('text', cv2.FONT_HERSHEY_DUPLEX, fontscale, thickness)
                y = int(20 + (i + 1) * dy + i * 0.5 * dy)
                cv2.putText(overlay_img, text, (20, y), 2, fontscale, (255, 255, 255), thickness)
                if polygon_meta['Label'][i] != 'background':
                    cv2.putText(overlay_img, polygon_meta['Id'][i], tuple(polygon_meta['Center'][i]), 0, fontscale,
                                (0, 0, 0), thickness * 2)
                # print polygon class?
            cv2.imwrite(os.path.join(gt_dir_overlay, filename), overlay_img)

    dataset_counts = {'Label': dict(zip(*np.unique(dataset_meta['Label'], return_counts=True))),
                      'Art': dict(zip(*np.unique(dataset_meta['Art'], return_counts=True))),
                      'Bild': dict(zip(*np.unique(dataset_meta['Bild'], return_counts=True))),
                      'Kombinationen': dict(zip(*np.unique(dataset_meta['Kombinationen'], return_counts=True)))}
    print('\n', dataset_counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", help="path to .xml file. Can either be a single path, multiple, paths "
                                           "or a directory containing multiple .xmls", required=True, nargs='+')
    parser.add_argument("--frame_path", help="path to directory containing dataset frames ")  # , required=True)
    parser.add_argument("--save_path", help="path to directory to copy dataset to")  # , required=True)
    # for segmentation-xml
    parser.add_argument("--segmentation", action='store_true',
                        help="The images as in args.xml_path are expected "
                             "to be in the same directory as args.xml_path in /images")
    parser.add_argument("--overlay", default=0.0, type=float,
                        help="if >0, images with gt overlay are created. "
                             "The images are expected to be in the same directory as args.xml_path in /images")
    args = parser.parse_args()

    # frame_path = r"C:\Users\atstern\Documents\Daten\MKT-Videos\smoke\1fps"
    # path_to_xml = r"C:\Users\atstern\Documents\Daten\MKT-Videos\smoke\1fps_annotation\annotations_mkt1.xml"

    if args.segmentation:
        parse_segmentation_xml(args.xml_path[0], overlay=args.overlay)

    else:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        annotations_all, _, _, _, _ = load_xml(path_to_xml=args.xml_path)

        i = 1
        class_names = {'Smoke-0': '00_Smoke-0', 'Smoke-1': '01_Smoke-1', 'Smoke-2': '02_Smoke-2',
                       'Smoke-3': '03_Smoke-3', 'Smoke-4': '04_Smoke-4'}
        for annotation_np in annotations_all:
            print('\rAnnotation file %d / %d' % (i, len(annotations_all)), end="")
            # remove outside frames from numpy
            print("\nRemove %d outside frames" % len(np.where(annotation_np[:, 8] == 'true')[0]))
            annotation_cleaned = np.delete(annotation_np, np.where(annotation_np[:, 8] == 'true'), axis=0)
            # create dirs according to np.unique(annotations[0][:, 3]) in savepath
            for class_name in np.unique(annotation_cleaned[:, 3]):
                if not os.path.exists(os.path.join(args.save_path, class_names[class_name])):
                    os.makedirs(os.path.join(args.save_path, class_names[class_name]))
            # copy files from frame_path to dirs in savepath according to annotation
            for frame in annotation_cleaned:
                shutil.copy(os.path.join(args.frame_path, basename(frame[2])),
                            os.path.join(args.save_path, class_names[frame[3]], basename(frame[2])))
            i += 1

    print("\ndone")
