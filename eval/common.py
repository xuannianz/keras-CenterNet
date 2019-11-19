"""
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

import cv2
import numpy as np
import progressbar

from utils.compute_overlap import compute_overlap
from utils.visualization import draw_detections, draw_annotations
from generators.utils import get_affine_transform, affine_transform

assert (callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).

    Returns:
        The average precision as computed in py-faster-rcnn.

    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, visualize=False,
                    flip_test=False,
                    keep_resolution=False):
    """
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.

    Returns:
        A list of lists containing the detections for each image in the generator.

    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in
                      range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        image = generator.load_image(i)
        src_image = image.copy()

        c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
        s = max(image.shape[0], image.shape[1]) * 1.0

        if not keep_resolution:
            tgt_w = generator.input_size
            tgt_h = generator.input_size
            image = generator.preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
        else:
            tgt_w = image.shape[1] | 31 + 1
            tgt_h = image.shape[0] | 31 + 1
            image = generator.preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
        if flip_test:
            flipped_image = image[:, ::-1]
            inputs = np.stack([image, flipped_image], axis=0)
        else:
            inputs = np.expand_dims(image, axis=0)
        # run network
        detections = model.predict_on_batch(inputs)[0]
        scores = detections[:, 4]
        # select indices which have a score above the threshold
        indices = np.where(scores > score_threshold)[0]

        # select those detections
        detections = detections[indices]
        detections_copy = detections.copy()
        detections = detections.astype(np.float64)
        trans = get_affine_transform(c, s, (tgt_w // 4, tgt_h // 4), inv=1)

        for j in range(detections.shape[0]):
            detections[j, 0:2] = affine_transform(detections[j, 0:2], trans)
            detections[j, 2:4] = affine_transform(detections[j, 2:4], trans)

        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, src_image.shape[1])
        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, src_image.shape[0])

        if visualize:
            # draw_annotations(src_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(src_image, detections[:5, :4], detections[:5, 4], detections[:5, 5].astype(np.int32),
                            label_to_name=generator.label_to_name,
                            score_threshold=score_threshold)

            # cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)
            cv2.namedWindow('{}'.format(i), cv2.WINDOW_NORMAL)
            cv2.imshow('{}'.format(i), src_image)
            cv2.waitKey(0)

        # copy detections to all_detections
        for class_id in range(generator.num_classes()):
            all_detections[i][class_id] = detections[detections[:, -1] == class_id, :-1]

    return all_detections


def _get_annotations(generator):
    """
    Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_annotations[num_images][num_classes] = annotations[num_class_annotations, 5]

    Args:
        generator: The generator used to retrieve ground truth annotations.

    Returns:
        A list of lists containing the annotations for each image in the generator.

    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.01,
        max_detections=100,
        visualize=False,
        flip_test=False,
        keep_resolution=False
):
    """
    Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        visualize: Show the visualized detections or not.
        flip_test:

    Returns:
        A dict mapping class names to mAP scores.

    """
    # gather all detections and annotations
    all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections,
                                     visualize=visualize, flip_test=flip_test, keep_resolution=keep_resolution)
    all_annotations = _get_annotations(generator)
    average_precisions = {}

    # all_detections = pickle.load(open('all_detections_{}.pkl'.format(epoch + 1), 'rb'))
    # all_annotations = pickle.load(open('all_annotations_{}.pkl'.format(epoch + 1), 'rb'))
    # pickle.dump(all_detections, open('all_detections_{}.pkl'.format(epoch + 1), 'wb'))
    # pickle.dump(all_annotations, open('all_annotations_{}.pkl'.format(epoch + 1), 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue
                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


if __name__ == '__main__':
    from generators.pascal import PascalVocGenerator
    from models.resnet import centernet
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test_generator = PascalVocGenerator(
        'datasets/VOC2007',
        'test',
        shuffle_groups=False,
        skip_truncated=False,
        skip_difficult=True,
    )
    model_path = 'checkpoints/2019-11-10/pascal_81_1.5415_3.0741_0.6860_0.7057_0.7209.h5'
    num_classes = test_generator.num_classes()
    flip_test = True
    nms = True
    keep_resolution = False
    score_threshold = 0.01
    model, prediction_model, debug_model = centernet(num_classes=num_classes,
                                                     nms=nms,
                                                     flip_test=flip_test,
                                                     freeze_bn=True,
                                                     score_threshold=score_threshold)
    prediction_model.load_weights(model_path, by_name=True, skip_mismatch=True)
    # inputs, targets = test_generator.__getitem__(0)
    # y1, y2, y3 = debug_model.predict(inputs[0])
    # np.save('y1', y1)
    # np.save('y2', y2)
    # np.save('y3', y3)
    # y1 = np.load('y1.npy')
    # y2 = np.load('y2.npy')
    # y3 = np.load('y3.npy')
    # from models.resnet import decode
    # import tensorflow as tf
    # import keras.backend as K
    # detections = decode(tf.constant(y1), tf.constant(y2), tf.constant(y3))
    # print(K.eval(detections))
    average_precisions = evaluate(test_generator, prediction_model,
                                  visualize=False,
                                  flip_test=flip_test,
                                  keep_resolution=keep_resolution,
                                  score_threshold=score_threshold)
    # compute per class average precision
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations), test_generator.label_to_name(label),
              'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)
    mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
    print('mAP: {:.4f}'.format(mean_ap))
