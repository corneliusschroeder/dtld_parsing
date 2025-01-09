from __future__ import print_function
import warnings
import argparse
import os
import numpy as np
import csv
import cv2
from dtld_parsing.driveu_dataset import DriveuDatabase

np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_file_dir", help="directory with DTLD label files (.json)", type=str, required=True)
    parser.add_argument(
        "--data_base_dir",
        default="",
        help="base directory of the original data",
        type=str,
    )
    parser.add_argument(
        "--export_dir", help="directory for exported dataset", type=str, required=True
    )
    return parser.parse_args()


def main(args):
    export_dir = args.export_dir
    train_dir = os.path.join(export_dir, 'train')
    test_dir = os.path.join(export_dir, 'test')
    val_dir = os.path.join(export_dir, 'validation')
    for dir in [train_dir, test_dir, val_dir]:
        os.makedirs(dir)

    keys = ["ImageID", "Source", "LabelName", "Confidence", "XMin",
            "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated",
            "IsGroupOf", "IsDepiction", "IsInside", "id", "ClassName"]
    train_data = {k: [] for k in keys}
    test_data = {k: [] for k in keys}
    val_data = {k: [] for k in keys}

    for label_file in os.listdir(args.label_file_dir):
        # Load database
        label_file = os.path.join(args.label_file_dir, label_file)
        database = DriveuDatabase(label_file)
        if not database.open(args.data_base_dir):
            continue

        for idx_d, img in enumerate(database.images):
            status, full_image = img.get_image()
            if not status:
                continue
            height, width, _ = full_image.shape
            split = np.random.choice(
                ['train', 'val', 'test'], p=[0.7, 0.15, 0.15])

            if status:
                for o in img.objects:
                    x_min = (o.x - 0.05 * o.width) / width
                    x_max = (o.x + 1.05 * o.width) / width
                    y_min = (o.y - 0.05 * o.width) / height
                    y_max = (o.y + 1.05 * o.height) / height

                    if o.attributes['relevance'] == 'relevant' and o.attributes['direction'] == 'front':
                        match split:
                            case 'train':
                                train_data['ImageID'].append(idx_d)
                                train_data['Source'].append("dtld")
                                train_data['LabelName'].append("/m/015qff")
                                train_data['Confidence'].append(1)
                                train_data['XMin'].append(x_min)
                                train_data['XMax'].append(x_max)
                                train_data['YMin'].append(y_min)
                                train_data['YMax'].append(y_max)
                                if o.attributes['occlusion'] == 'not_occluded':
                                    train_data['IsOccluded'].append(0)
                                else:
                                    train_data['IsOccluded'].append(1)
                                train_data['IsTruncated'].append(0)
                                train_data['IsGroupOf'].append(0)
                                train_data['IsDepiction'].append(0)
                                train_data['IsInside'].append(0)
                                train_data['id'].append("/m/015qff")
                                train_data['ClassName'].append("Traffic light")
                            case 'val':
                                val_data['ImageID'].append(idx_d)
                                val_data['Source'].append("dtld")
                                val_data['LabelName'].append("/m/015qff")
                                val_data['Confidence'].append(1)
                                val_data['XMin'].append(x_min)
                                val_data['XMax'].append(x_max)
                                val_data['YMin'].append(y_min)
                                val_data['YMax'].append(y_max)
                                if o.attributes['occlusion'] == 'not_occluded':
                                    val_data['IsOccluded'].append(0)
                                else:
                                    val_data['IsOccluded'].append(1)
                                val_data['IsTruncated'].append(0)
                                val_data['IsGroupOf'].append(0)
                                val_data['IsDepiction'].append(0)
                                val_data['IsInside'].append(0)
                                val_data['id'].append("/m/015qff")
                                val_data['ClassName'].append("Traffic light")
                            case 'test':
                                test_data['ImageID'].append(idx_d)
                                test_data['Source'].append("dtld")
                                test_data['LabelName'].append("/m/015qff")
                                test_data['Confidence'].append(1)
                                test_data['XMin'].append(x_min)
                                test_data['XMax'].append(x_max)
                                test_data['YMin'].append(y_min)
                                test_data['YMax'].append(y_max)
                                if o.attributes['occlusion'] == 'not_occluded':
                                    test_data['IsOccluded'].append(0)
                                else:
                                    test_data['IsOccluded'].append(1)
                                test_data['IsTruncated'].append(0)
                                test_data['IsGroupOf'].append(0)
                                test_data['IsDepiction'].append(0)
                                test_data['IsInside'].append(0)
                                test_data['id'].append("/m/015qff")
                                test_data['ClassName'].append("Traffic light")

                match split:
                    case 'train':
                        cv2.imwrite(os.path.join(
                            train_dir, f"{idx_d}.jpg"), full_image)
                    case 'val':
                        cv2.imwrite(os.path.join(
                            val_dir, f"{idx_d}.jpg"), full_image)
                    case 'test':
                        cv2.imwrite(os.path.join(
                            test_dir, f"{idx_d}.jpg"), full_image)
            if idx_d == 100:
                break

    with open(os.path.join(export_dir, "train_annotations.csv"), "w") as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(*[train_data[key] for key in keys]))

    with open(os.path.join(export_dir, "test_annotations.csv"), "w") as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(*[test_data[key] for key in keys]))

    with open(os.path.join(export_dir, "val_annotations.csv"), "w") as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(*[val_data[key] for key in keys]))


if __name__ == "__main__":
    main(parse_args())
