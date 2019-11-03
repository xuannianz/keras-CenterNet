import cv2
import keras
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings

from utils import get_affine_transform, affine_transform, gaussian_radius, draw_gaussian


class Generator(keras.utils.Sequence):
    """
    Abstract generator class.
    """

    def __init__(
            self,
            multi_scale=False,
            multi_image_sizes=(320, 352, 384, 416, 448, 480, 512, 544, 576, 608),
            batch_size=1,
            group_method='ratio',  # one of 'none', 'random', 'ratio'
            shuffle_groups=True,
            input_size=512,
            max_objects=100
    ):
        """
        Initialize Generator object.

        Args:
            batch_size: The size of the batches to generate.
            group_method: Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
                ratio 指的是对所有图片按 aspect_ratio 进行排序
                random 指的是随机生成一种顺序
                Note: group 的概念就是把图片按照 group_method 的方式排序, 然后一个 batch 作为一个 group
            shuffle_groups: If True, shuffles the groups each epoch.
            input_size:
            max_objects:
        """
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.input_size = input_size
        self.max_objects = max_objects
        self.groups = None
        self.multi_scale = multi_scale
        self.multi_image_sizes = multi_image_sizes
        self.current_index = 0

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)
        self.current_index = 0

    def size(self):
        """
        Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def get_anchors(self):
        """
        loads the anchors from a txt file
        """
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        # (N, 2), wh
        return np.array(anchors).reshape(-1, 2)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """
        Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """
        Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """
        Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        """
        Load annotations for all images in group.
        """
        # load_annotations 返回的样式是  {'labels': np.array, 'annotations': np.array}
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert (isinstance(annotations,
                               dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(
                type(annotations))
            assert (
                    'labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert (
                    'bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                # np.array 之间的 or 可以使用 |
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] <= 0) |
                (annotations['bboxes'][:, 3] <= 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                # keys() 有两个值, 一个是 labels, 一个是 bboxes
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)
            if annotations['bboxes'].shape[0] == 0:
                warnings.warn('Image with id {} (shape {}) contains no valid boxes before transform'.format(
                    group[index],
                    image.shape,
                ))
        return image_group, annotations_group

    def clip_transformed_annotations(self, image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        filtered_image_group = []
        filtered_annotations_group = []
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            image_height = image.shape[0]
            image_width = image.shape[1]
            # x1
            annotations['bboxes'][:, 0] = np.clip(annotations['bboxes'][:, 0], 0, image_width - 2)
            # y1
            annotations['bboxes'][:, 1] = np.clip(annotations['bboxes'][:, 1], 0, image_height - 2)
            # x2
            annotations['bboxes'][:, 2] = np.clip(annotations['bboxes'][:, 2], 1, image_width - 1)
            # y2
            annotations['bboxes'][:, 3] = np.clip(annotations['bboxes'][:, 3], 1, image_height - 1)
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            small_indices = np.where(
                # np.array 之间的 or 可以使用 |
                (annotations['bboxes'][:, 2] - annotations['bboxes'][:, 0] < 10) |
                (annotations['bboxes'][:, 3] - annotations['bboxes'][:, 1] < 10)
            )[0]

            # delete invalid indices
            if len(small_indices):
                # keys() 有两个值, 一个是 labels, 一个是 bboxes
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], small_indices, axis=0)
                # import cv2
                # for invalid_index in small_indices:
                #     x1, y1, x2, y2 = annotations['bboxes'][invalid_index]
                #     label = annotations['labels'][invalid_index]
                #     class_name = self.labels[label]
                #     print('width: {}'.format(x2 - x1))
                #     print('height: {}'.format(y2 - y1))
                #     cv2.rectangle(image, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 255, 0), 2)
                #     cv2.putText(image, class_name, (int(round(x1)), int(round(y1))), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
                # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
            if annotations_group[index]['bboxes'].shape[0] != 0:
                filtered_image_group.append(image)
                filtered_annotations_group.append(annotations_group[index])
            else:
                warnings.warn('Image with id {} (shape {}) contains no valid boxes after transform'.format(
                    group[index],
                    image.shape,
                ))

        return filtered_image_group, filtered_annotations_group

    def load_image_group(self, group):
        """
        Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_visual_effect_group_entry(self, image, annotations):
        """
        Randomly transforms image and annotation.
        """
        visual_effect = next(self.visual_effect_generator)
        # apply visual effect
        image = visual_effect(image)
        return image, annotations

    def random_visual_effect_group(self, image_group, annotations_group):
        """
        Randomly apply visual effect on each image.
        """
        assert (len(image_group) == len(annotations_group))

        if self.visual_effect_generator is None:
            # do nothing
            return image_group, annotations_group

        for index in range(len(image_group)):
            # apply effect on a single group entry
            image_group[index], annotations_group[index] = self.random_visual_effect_group_entry(
                image_group[index], annotations_group[index]
            )

        return image_group, annotations_group

    def random_transform_group_entry(self, image, annotations, transform=None):
        """
        Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image,
                                                       self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations

    def random_transform_group(self, image_group, annotations_group):
        """
        Randomly transforms each image and its annotations.
        """

        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index],
                                                                                             annotations_group[index])

        return image_group, annotations_group

    def random_rotate_group_entry(self, image, annotations):
        image, bboxes = rotate(image, annotations['bboxes'])
        annotations['bboxes'] = bboxes
        return image, annotations

    def random_rotate_group(self, image_group, annotations_group):
        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_rotate_group_entry(image_group[index],
                                                                                          annotations_group[index])
        return image_group, annotations_group

    def random_crop_group_entry(self, image, annotations):
        image, bboxes = crop(image, annotations['bboxes'])
        annotations['bboxes'] = bboxes
        return image, annotations

    def random_crop_group(self, image_group, annotations_group):
        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_crop_group_entry(image_group[index],
                                                                                        annotations_group[index])

        return image_group, annotations_group

    def preprocess_group_entry(self, image, annotations):
        """
        Preprocess image and its annotations.
        """

        # preprocess the image
        image, scale, offset_h, offset_w = self.preprocess_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= scale
        annotations['bboxes'][:, [0, 2]] += offset_w
        annotations['bboxes'][:, [1, 3]] += offset_h
        # print(annotations['bboxes'][:, [2, 3]] - annotations['bboxes'][:, [0, 1]])
        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """
        Preprocess each image and its annotations in its group.
        """
        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index],
                                                                                       annotations_group[index])

        return image_group, annotations_group

    def group_images(self):
        """
        Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images

        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group, annotations_group):
        """
        Compute inputs for the network using an image_group.
        """
        # construct an image batch object
        batch_images = np.zeros((len(image_group), self.input_size, self.input_size, 3), dtype=np.float32)

        output_size = self.input_size // 4
        batch_hms = np.zeros((len(image_group), output_size, output_size, self.num_classes()), dtype=np.float32)
        batch_whs = np.zeros((len(image_group), self.max_objects, 2), dtype=np.float32)
        batch_regs = np.zeros((len(image_group), self.max_objects, 2), dtype=np.float32)
        # bbox 缩小后会计算并且判断其 h, w 是否大小 0, 此举会过滤掉那些比较小的 bbox, 也就不需要参与 loss 计算
        batch_reg_masks = np.zeros((len(image_group), self.max_objects), dtype=np.float32)
        # bbox 中心点在 hm 上的下标 cy * w + cx
        batch_indices = np.zeros((len(image_group), self.max_objects), dtype=np.float32)

        # copy all images to the upper left part of the image batch object
        for b, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
            s = max(image.shape[0], image.shape[1]) * 1.0

            # inputs
            trans_input = get_affine_transform(c, s, (self.input_size, self.input_size))
            image = cv2.warpAffine(image, trans_input, (self.input_size, self.input_size), flags=cv2.INTER_LINEAR)
            batch_images[b] = image

            # outputs
            bboxes = annotations['bboxes']
            # 按道理不该有这种情况
            assert bboxes.shape[0] != 0
            class_ids = annotations['labels']
            assert class_ids.shape[0] != 0

            trans_output = get_affine_transform(c, s, (output_size, output_size))
            for i in range(bboxes.shape[0]):
                # 不想修改原来的 annotations 中的 bboxes
                bbox = bboxes[i].copy()
                cls_id = class_ids[i]
                # (x1, y1)
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                # (x2, y2)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_size - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_size - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius_h, radius_w = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius_h = max(0, int(radius_h))
                    radius_w = max(0, int(radius_w))
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_gaussian(batch_hms[b, :, :, cls_id], ct_int, radius_h, radius_w)
                    batch_whs[b, i] = 1. * w, 1. * h
                    # 中心点所在的像素的下标
                    batch_indices[b, i] = ct_int[1] * output_size + ct_int[0]
                    # 中心点的偏移量
                    batch_regs[b, i] = ct - ct_int
                    batch_reg_masks[b, i] = 1
            print(np.sum(batch_reg_masks[b]))
            for i in range(self.num_classes()):
                plt.subplot(4, 5, i + 1)
                hm = batch_hms[b, :, :, i]
                plt.imshow(hm, cmap='gray')
                plt.axis('off')
            plt.show()
            for i in range(bboxes.shape[0]):
                x1, y1 = np.round(affine_transform(bboxes[i, :2], trans_input)).astype(np.int32)
                x2, y2 = np.round(affine_transform(bboxes[i, 2:], trans_input)).astype(np.int32)
                class_id = class_ids[i]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(image, str(class_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', image)
            cv2.waitKey()
        return [batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices]

    def compute_targets(self, image_group, annotations_group):
        """
        Compute target outputs for the network using images and their annotations.
        """
        return np.zeros((len(image_group),))

    def compute_inputs_targets(self, group):
        """
        Compute inputs and target outputs for the network.
        """

        # load images and annotations
        # list
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # image_group, annotations_group = self.random_crop_group(image_group, annotations_group)
        # image_group, annotations_group = self.random_rotate_group(image_group, annotations_group)
        #
        # # randomly apply visual effect
        # image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)
        #
        # # randomly transform data
        # image_group, annotations_group = self.random_transform_group(image_group, annotations_group)
        #
        # # perform preprocessing steps
        # image_group, annotations_group = self.preprocess_group(image_group, annotations_group)
        #
        # # check validity of annotations
        # image_group, annotations_group = self.clip_transformed_annotations(image_group, annotations_group, group)

        if len(image_group) == 0:
            return None, None

        # compute network inputs
        inputs = self.compute_inputs(image_group, annotations_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[self.current_index]
        if self.multi_scale:
            if self.current_index % 10 == 0:
                random_size_index = np.random.randint(0, len(self.multi_image_sizes))
                self.image_size = self.multi_image_sizes[random_size_index]
        inputs, targets = self.compute_inputs_targets(group)
        while inputs is None:
            current_index = self.current_index + 1
            if current_index >= len(self.groups):
                current_index = current_index % (len(self.groups))
            self.current_index = current_index
            group = self.groups[self.current_index]
            inputs, targets = self.compute_inputs_targets(group)
        current_index = self.current_index + 1
        if current_index >= len(self.groups):
            current_index = current_index % (len(self.groups))
        self.current_index = current_index
        return inputs, targets

    def preprocess_image(self, image):
        image_height, image_width = image.shape[:2]
        if image_height > image_width:
            scale = self.image_size / image_height
            resized_height = self.image_size
            resized_width = int(image_width * scale)
        else:
            scale = self.image_size / image_width
            resized_height = int(image_height * scale)
            resized_width = self.image_size
        image = cv2.resize(image, (resized_width, resized_height))
        new_image = np.ones((self.image_size, self.image_size, 3), dtype=np.float32) * 127.5
        offset_h = (self.image_size - resized_height) // 2
        offset_w = (self.image_size - resized_width) // 2
        new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image.astype(np.float32)
        new_image /= 255.
        return new_image, scale, offset_h, offset_w

    def get_transformed_group(self, group):
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)
        return image_group, annotations_group

    def get_cropped_and_rotated_group(self, group):
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly transform data
        image_group, annotations_group = self.random_crop_group(image_group, annotations_group)
        image_group, annotations_group = self.random_rotate_group(image_group, annotations_group)
        return image_group, annotations_group
