"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from yolo.generators.common import Generator
from yolo.utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from
import csv
import sys
import os.path
from collections import OrderedDict


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        # raise_from 第二个参数为 None, 表示不显示 try 语句中异常的内容
        # 那么我想目的就是用自定义的错误信息来替换掉系统默认的错误信息
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """
    Parse the classes file given by csv_reader.
    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_annotations(csv_reader, classes):
    """
    Read annotations from the csv_reader.
    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        # UNCLEAR: 你保留一个没有 annotation 的 img_file 有何用?
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result


def _open_for_csv(path):
    """
    Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb', for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        # 这里的 newline='' 与不指定 newline 的区别是
        # 如果不指定 newline, 默认原文件中的 \r,\n,\r\n 会被转换成 \n 返回
        # 如果指定了 newline='', 那么不进行这样的转换
        return open(path, 'r', newline='')


class CSVGenerator(Generator):
    """
    Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
            self,
            csv_data_file,
            csv_class_file,
            base_dir=None,
            **kwargs
    ):
        """
        Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        # 如果只是文件名, 那么会到 base_dir 下面寻找
        # 也可以是图片的绝对路径
        self.image_names = []
        # 存放从 annotation csv 中读取的数据
        self.image_data = {}
        self.base_dir = base_dir

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            # dirname 得到的是 csv_data_file 去掉最后一个斜杠之后的内容, 不管 csv_data_file 是相对路径还是绝对路径
            self.base_dir = os.path.dirname(csv_data_file)

        # parse the provided class file
        try:
            with _open_for_csv(csv_class_file) as file:
                # class_name --> class_id
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}
        # class_id --> class_name
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with _open_for_csv(csv_data_file) as file:
                # {'img_path1':[{'x1':xx,'y1':xx,'x2':xx,'y2':xx,'class':xx}...],...}
                # 这里的 class 是 class_name
                self.image_data = _read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        # 可以是 self.base_dir 下面的图片的文件名
        # 也可以是绝对路径,  osp.join('/home/adam', '/root/workspace'), 可以看到 join 时, 如果后者是绝对路径, 那么前者可以是任何值
        # 没有任何影响
        self.image_names = list(self.image_data.keys())

        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        # self.labels 是 class_id --> class_name 的 dict
        return label in self.labels

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """
        Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """
        Returns the image path for image_index.
        """
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        path = self.image_names[image_index]
        annotations = {'labels': np.empty((0,), dtype=np.int32), 'bboxes': np.empty((0, 4))}

        for idx, annot in enumerate(self.image_data[path]):
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(annot['x1']),
                float(annot['y1']),
                float(annot['x2']),
                float(annot['y2']),
            ]]))

        return annotations


if __name__ == '__main__':
    import cv2
    csv_generator = CSVGenerator(
        csv_data_file='/home/adam/workspace/github/keras-retinanet_vat/val_gray_annotations_20190615_1255_127.csv',
        csv_class_file='/home/adam/workspace/github/keras-retinanet_vat/vat_classes.csv'
    )
    for image_group, annotation_group, targets in csv_generator:
        locations = targets[0]
        batch_regr_targets = targets[1]
        batch_cls_targets = targets[2]
        batch_centerness_targets = targets[3]
        for image, annotation, regr_targets, cls_targets, centerness_targets in zip(image_group, annotation_group, batch_regr_targets, batch_cls_targets, batch_centerness_targets):
            gt_boxes = annotation['bboxes']
            for gt_box in gt_boxes:
                gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
                cv2.rectangle(image, (int(gt_xmin), int(gt_ymin)), (int(gt_xmax), int(gt_ymax)), (0, 255, 0), 2)
            pos_indices = np.where(centerness_targets[:, 1] == 1)[0]
            for pos_index in pos_indices:
                cx, cy = locations[pos_index]
                l, t, r, b, _ = regr_targets[pos_index]
                xmin = cx - l
                ymin = cy - t
                xmax = cx + r
                ymax = cy + b
                class_id = np.argmax(cls_targets[pos_index])
                centerness = centerness_targets[pos_index][0]
                cv2.putText(image, '{:.2f}'.format(centerness), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 255), 2)
                cv2.putText(image, str(class_id), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
                cv2.circle(image, (round(cx), round(cy)), 5, (255, 0, 0), -1)
                cv2.rectangle(image, (round(xmin), round(ymin)), (round(xmax), round(ymax)), (0, 0, 255), 2)
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.imshow('image', image)
                cv2.waitKey(0)
        pass
