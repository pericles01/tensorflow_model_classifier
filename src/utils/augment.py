"""Description
   -----------
This script includes functions and transforms related to image augmentation using albumentations package.
Uncomment desired augmentation transforms in order to use them in training.
"""

import albumentations as A
import matplotlib.pyplot as plt
import cv2
from os import path, listdir, mkdir
from datetime import datetime
import numpy as np
import shutil
import ntpath
import argparse



def generate_augmentation(args, trials=None):

    augment=[]
    if trials is None:
        if len(args.augment) != 0:    
            # ## -- Spatial-level transforms -- #
            if "RandomRotate90" in args.augment:
                augment.append(A.RandomRotate90(args.augment["RandomRotate90"]))
            if "Rotate" in args.augment:
                augment.append(A.Rotate(args.augment["Rotate"]))
            if "IAAPerspective" in args.augment:
                augment.append(A.IAAPerspective(args.augment["IAAPerspective"]))
            if "RandomGridShuffle" in args.augment:
                augment.append(A.RandomGridShuffle(args.augment["RandomGridShuffle"]))
            if "RandomResizedCrop" in args.augment:
                augment.append(A.RandomResizedCrop(args.augment["RandomResizedCrop"]))
            if "Flip" in args.augment:
                augment.append(A.Flip(args.augment["Flip"]))
            if "GridDistortion" in args.augment:
                augment.append(A.GridDistortion(args.augment["GridDistortion"]))
            
            # ## -- Pixel-level transforms --#
            if "GaussianBlur" in args.augment:
                augment.append(A.Blur(args.augment["GaussianBlur"]))
            if "Blur" in args.augment:
                augment.append(A.Blur(args.augment["Blur"]))
            if "MotionBlur" in args.augment:
                augment.append(A.Blur(args.augment["MotionBlur"]))
            if "GlassBlur" in args.augment:
                augment.append(A.Blur(args.augment["GlassBlur"]))
            if "RandomFog" in args.augment:
                augment.append(A.Blur(args.augment["RandomFog"]))
            if "MedianBlur" in args.augment:
                augment.append(A.MedianBlur(args.augment["MedianBlur"]))

            # ## Sharpening transforms ##
            if "IAASharpen" in args.augment:
                augment.append(A.IAASharpen(args.augment["IAASharpen"]))
            if "IAAEmboss" in args.augment:
                augment.append(A.Blur(args.augment["IAAEmboss"]))
            if "Downscale" in args.augment:
                augment.append(A.Blur(args.augment["Downscale"]))
            
            # ## RGB trabsforms ##
            if "RandomBrightnessContrast" in args.augment:
                augment.append(A.Blur(args.augment["RandomBrightnessContrast"]))
            if "ChannelDropout" in args.augment:
                augment.append(A.Blur(args.augment["ChannelDropout"]))
            if "ChannelShuffle" in args.augment:
                augment.append(A.Blur(args.augment["ChannelShuffle"]))
            if "ColorJitter" in args.augment:
                augment.append(A.ChannelDropout(args.augment["ColorJitter"]))
            if "HueSaturationValue" in args.augment:
                augment.append(A.ChannelDropout(args.augment["HueSaturationValue"]))
            if "RGBShift" in args.augment:
                augment.append(A.ChannelDropout(args.augment["RGBShift"]))
            
            # ##  Binary transforms ##
            if "ToGray" in args.augment:
                augment.append(A.Blur(args.augment["ToGray"]))
            if "ToSepia" in args.augment:
                augment.append(A.Blur(args.augment["ToSepia"]))

    else:
        augment.extend([
            A.Rotate(limit=90, p=trials),
            A.Flip(p=trials),
            A.RandomResizedCrop(224, 224, p=trials),
            A.MotionBlur(p=trials),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=trials),
            A.ColorJitter(p=trials)
            ])
    transforms = A.Compose(augment,)
    
    return transforms


# # Define the set of transformations to be applied on training data
# transforms = A.Compose([
#     # ## -- Spatial-level transforms -- #
#     #A.RandomRotate90(p=0.5),
#     #A.Rotate(limit=90, p=0.5),
#     #A.Flip(p=0.5),
#     #A.IAAPerspective(p=0.1), # Doesn't support detection
#     #A.GridDistortion(p=0.5),  # Doesn't support detection
#     #A.RandomGridShuffle(grid=(2, 2), p=0.5),  # Doesn't support detection
#     #A.RandomResizedCrop(args.dim[0], args.dim[1], p=0.1),  # --resize can be set to None
#     #A.RandomResizedCrop(224, 224, p=0.3),
#     
#     # ## -- Pixel-level transforms --#
#     # ## Blur transforms ##
#     #A.GaussianBlur(p=0.5),
#     #A.Blur(p=0.3),
#     #A.MotionBlur(p=0.3),
#     #A.MedianBlur(blur_limit=5, p=0.5),
#     #A.GlassBlur(sigma=0.9, max_delta=1, iterations=1, p=0.1),
#     #A.RandomFog(p=0.1),
#     # ## Sharpening transforms ##
#     #A.IAASharpen(p=0.5),
#     #A.IAAEmboss(p=0.1),
#     #A.Downscale(scale_min=0.75, scale_max=0.9, p=0.5),
#     # ## RGB trabsforms ##
#     #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
#     #A.ChannelDropout(p=0.5),
#     #A.ChannelShuffle(p=0.5),
#     #A.ColorJitter(p=0.5),
#     #A.HueSaturationValue(p=0.5),
#     #A.RGBShift(p=0.1),
#     # ##  Binary transforms ##
#     #A.ToGray(p=0.05),
#     #A.ToSepia(p=0.05)
# ],
#     # bbox_params=A.BboxParams(format='pascal_voc') # uncomment for detection
# )


# def generate_augmentation_samples(image_path, transform):
#     """
#     Helper function to generate sample outputs of transforms from albumentations augmentations package
#     transforms can be found here: https://albumentations.ai/docs/api_reference/augmentations/transforms/
#     :param image_path
#     :param transform: desired transform from albumentations package, e.g.A.MedianBlur(blur_limit=5, p=1).
#     Make sure that p=1 and that the transform accepts float32 input to be integrated in the project
#     """

#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

#     fig, axs = plt.subplots(nrows=3, ncols=3, squeeze=False, figsize=(12, 12))
#     for i in range(3):
#         for j in range(3):
#             out = transform(image=img)
#             axs[i, j].imshow(out['image'])
#             axs[i, j].axis('off')
#     fig.tight_layout()
#     plt.show()


# def generate_augmentated_dataset(data_path, epochs):
#     from utils import data  # no clue why it give me an error when I put this on top
#     time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     mkdir(path.join("./results", time_stamp))
#     if path.isfile(data_path):
#         task = 'classification'
#     else:
#         if 'img' and 'gt' in listdir(data_path):
#             task = 'segmentation'
#         elif 'img+bb' in listdir(data_path):
#             task = 'detection'
#         else:
#             task = 'classification'

#     if task == 'classification':
#         image_paths, gt_paths, class_names = data.get_classification_data_paths(data_path=data_path, args=args)
#         if args.num_labels > 0:
#             mkdir(path.join("./results", time_stamp, 'img+gt'))
#         else:
#             for i in range(len(class_names)):
#                 mkdir(path.join("./results", time_stamp, class_names[i]))

#         for ep in range(epochs):
#             print('\r Processing epoch %d / %d' % (ep + 1, epochs), end="")
#             for idx in range(len(image_paths)):
#                 img = cv2.imread(image_paths[idx])
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
#                 trans_img = transforms(image=img)['image']
#                 trans_img = cv2.cvtColor(trans_img, cv2.COLOR_BGR2RGB)

#                 if args.num_labels > 0:
#                     cv2.imwrite(path.join("./results", time_stamp, 'img+gt',
#                                           'epoch_' + str(ep) + '_' + ntpath.basename(image_paths[idx])), trans_img)
#                     with open(path.join("./results", time_stamp, 'img+gt', 'epoch_' + str(ep) + '_' +
#                                                                            ntpath.basename(image_paths[idx])[
#                                                                            :-4]) + '.txt', 'w') as f:
#                         f.write(','.join(map(str, gt_paths[idx])))

#                 else:
#                     cv2.imwrite(path.join("./results", time_stamp, class_names[gt_paths[idx]],
#                                           'epoch_' + str(ep) + '_' + ntpath.basename(image_paths[idx])), trans_img)


#     elif task == 'segmentation':
#         image_paths, gt_paths = data.get_segmentation_data_paths(data_path=data_path, args=args)
#         mkdir(path.join("./results/", time_stamp, 'img'))
#         mkdir(path.join("./results/", time_stamp, 'gt'))

#         for ep in range(epochs):
#             print('\r Processing epoch %d / %d' % (ep + 1, epochs), end="")
#             for idx in range(len(image_paths)):
#                 img = cv2.imread(image_paths[idx])
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 mask = cv2.imread(gt_paths[idx], cv2.IMREAD_GRAYSCALE)
#                 img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
#                 mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

#                 data = {"image": img, "mask": mask}
#                 aug_data = transforms(**data)
#                 trans_img = aug_data["image"]
#                 trans_mask = aug_data["mask"]

#                 trans_img = cv2.cvtColor(trans_img, cv2.COLOR_BGR2RGB)
#                 cv2.imwrite(path.join("./results", time_stamp, 'img',
#                                       'epoch_' + str(ep) + '_' + ntpath.basename(image_paths[idx])), trans_img)
#                 cv2.imwrite(path.join("./results", time_stamp, 'gt',
#                                       'epoch_' + str(ep) + '_' + ntpath.basename(gt_paths[idx])), trans_mask)

#     else:  # Detection
#         def get_bbox_params(bbox):
#             return int(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]) / 2, float(bbox[4]) / 2

#         image_paths, gt_paths = data.get_detection_data_paths(data_path=data_path, args=args)
#         mkdir(path.join("./results/", time_stamp, 'img+bb'))
#         shutil.copy(path.join(data_path, 'class_names.txt'), path.join("./results/", time_stamp, 'class_names.txt'))
#         for ep in range(epochs):
#             print('\r Processing epoch %d / %d' % (ep + 1, epochs), end="")
#             for idx in range(len(image_paths)):
#                 img = cv2.imread(image_paths[idx])
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 with open(gt_paths[idx]) as f:
#                     gt = f.readlines()
#                 f.close()
#                 bboxes = []
#                 # coco to pascal
#                 for box in gt:
#                     box = box.strip()
#                     box = box.split()
#                     class_num, center_x, center_y, half_width, half_height = get_bbox_params(box)
#                     bboxes.append([center_x - half_width, center_y - half_height, center_x + half_width,
#                                    center_y + half_height, class_num])

#                 # pad resize image and bboxes
#                 ih, iw = args.dim
#                 h, w, _ = img.shape
#                 scale = min(iw / w, ih / h)
#                 nw, nh = int(scale * w), int(scale * h)
#                 img = cv2.resize(img, (nw, nh))

#                 image_padded = np.full(shape=[ih, iw, 3], fill_value=0).astype(np.uint8)
#                 dw, dh = (iw - nw) // 2, (ih - nh) // 2
#                 image_padded[dh:nh + dh, dw:nw + dw, :] = img
#                 eshe = image_padded
#                 img = eshe
#                 bboxes = bboxes * np.array([w, h, w, h, 1])
#                 bboxes = bboxes.astype(np.int64)
#                 bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + dw
#                 bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + dh

#                 # apply augmentation
#                 transformed = transforms(image=img, bboxes=bboxes)
#                 trans_img = transformed['image']
#                 trans_bboxes = transformed['bboxes']
#                 trans_bboxes = np.asarray(trans_bboxes)

#                 # pascal to coco
#                 trans_bboxes[:, 2] = trans_bboxes[:, 2] - trans_bboxes[:, 0]
#                 trans_bboxes[:, 3] = trans_bboxes[:, 3] - trans_bboxes[:, 1]
#                 trans_bboxes[:, 0] = trans_bboxes[:, 0] + (trans_bboxes[:, 2] / 2)
#                 trans_bboxes[:, 1] = trans_bboxes[:, 1] + (trans_bboxes[:, 3] / 2)
#                 trans_bboxes = trans_bboxes * np.array([1 / iw, 1 / ih, 1 / iw, 1 / ih, 1])
#                 trans_bboxes = trans_bboxes[:, [4, 0, 1, 2, 3]]
#                 # save
#                 trans_img = cv2.cvtColor(trans_img, cv2.COLOR_BGR2RGB)
#                 cv2.imwrite(path.join("./results", time_stamp, 'img+bb',
#                                       'epoch_' + str(ep) + '_' + ntpath.basename(image_paths[idx])), trans_img)
#                 with open(path.join("./results", time_stamp, 'img+bb',
#                                     'epoch_' + str(ep) + '_' + ntpath.basename(image_paths[idx])[:-4]) + '.txt',
#                           'w') as f:
#                     for box in trans_bboxes:
#                         f.write("%s %s %s %s %s\n" % (int(box[0]), box[1], box[2], box[3], box[4]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help="input data directory")
    parser.add_argument('--outdir', type=str, help="output data directory")
    parser.add_argument('--ckpt', type=str, default=None, help="path to model weight file (for pretrained weights)")
    parser.add_argument('--config', type=str, default='run_config.yml', help="path to config file of the experiment")
    args = parser.parse_args()
    
    from experiment_config import ExperimentConfig
    conf = ExperimentConfig(args.config)
    conf.add(args)
    
    from main_experiment import Exp
    args = Exp(conf)

    #generate_augmentated_dataset(args.indir, args.epochs)
