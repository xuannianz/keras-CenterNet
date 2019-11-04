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

import argparse
from datetime import date
import keras
import keras.backend as K
from keras.optimizers import Adam, SGD
import keras.preprocessing.image
import os
import sys
import tensorflow as tf

from models.resnet import centernet


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        # 如果 path 已经存在, 会抛 FileExistsError
        os.makedirs(path)
    except OSError:
        # 如果已经存在, 且不是目录, 那么再次抛出异常
        if not os.path.isdir(path):
            raise


def get_session():
    """
    Construct a modified tf session.
    """
    config = tf.ConfigProto()
    # 根据程序需要来增加 gpu 的内存
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_callbacks(training_model, prediction_model, validation_generator, args):
    """
    Creates the callbacks to use during training.

    Args
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from eval.coco import CocoEval
            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        else:
            from eval.pascal import Evaluate
            evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        callbacks.append(evaluation)

    # save the model
    # --no-snapshots 的 dest 是 snapshots
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{dataset_type}_{{epoch:02d}}_{{loss:.4f}}_{{val_loss:.4f}}.h5'.format(dataset_type=args.dataset_type)
            ),
            verbose=1,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        callbacks.append(checkpoint)

    # callbacks.append(keras.callbacks.ReduceLROnPlateau(
    #     monitor='loss',
    #     factor=0.1,
    #     patience=2,
    #     verbose=1,
    #     mode='auto',
    #     min_delta=0.0001,
    #     cooldown=0,
    #     min_lr=0
    # ))

    return callbacks


def create_generators(args):
    """
    Create generators for training and validation.

    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': args.batch_size,
        'input_size': args.input_size,
    }

    # create random transform generator for augmenting training data
    # if args.random_transform:
    #     transform_generator = random_transform_generator(
    #         min_rotation=-0.1,
    #         max_rotation=0.1,
    #         min_translation=(-0.1, -0.1),
    #         max_translation=(0.1, 0.1),
    #         min_shear=-0.1,
    #         max_shear=0.1,
    #         min_scaling=(0.9, 0.9),
    #         max_scaling=(1.1, 1.1),
    #         flip_x_chance=0.5,
    #         flip_y_chance=0.5,
    #     )
    #     visual_effect_generator = random_visual_effect_generator(
    #         contrast_range=(0.9, 1.1),
    #         brightness_range=(-.1, .1),
    #         hue_range=(-0.05, 0.05),
    #         saturation_range=(0.95, 1.05)
    #     )
    # else:
    #     transform_generator = random_transform_generator(flip_x_chance=0.5)
    #     visual_effect_generator = None

    if args.dataset_type == 'pascal':
        from generators.pascal import PascalVocGenerator
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            skip_difficult=True,
            multi_scale=args.multi_scale,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'val',
            skip_difficult=True,
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'csv':
        from generators.csv_ import CSVGenerator
        train_generator = CSVGenerator(
            args.annotations_path,
            args.classes_path,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        if args.val_annotations_path:
            validation_generator = CSVGenerator(
                args.val_annotations_path,
                args.classes_path,
                shuffle_groups=False,
                **common_args
            )
        else:
            validation_generator = None
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.num_gpus > 1 and parsed_args.batch_size < parsed_args.num_gpus:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.num_gpus > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.num_gpus > 1 and not parsed_args.multi_gpu_force:
        raise ValueError(
            "Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    return parsed_args


def parse_args(args):
    """
    Parse the arguments.
    """
    today = str(date.today())
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations_path', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes_path', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations-path',
                            help='Path to CSV file containing annotations for validation (optional).')

    parser.add_argument('--snapshot', help='Resume training from a snapshot.', default='/home/adam/.keras/models/ResNet-50-model.keras.h5')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')

    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    # NOTE: nvidia-smi 和 tensorflow 中的 gpu id 可能不一致
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--num_gpus', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.',
                        action='store_true')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',
                        help='Path to store snapshots of models during training',
                        default='checkpoints/{}'.format(today))
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output',
                        default='logs/{}'.format(today))
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                        action='store_false')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--input-size', help='Rescale the image so the smallest side is min_side.', type=int,
                        default=512)
    parser.add_argument('--multi-scale', help='Multi-Scale training', default=False, action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss',
                        action='store_true')

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers', help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit_generator.', type=int,
                        default=10)
    print(vars(parser.parse_args(args)))
    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    K.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args)

    num_classes = train_generator.num_classes()
    model, prediction_model = centernet(num_classes=num_classes, input_size=args.input_size)

    # create the model
    print('Loading model, this may take a second...')
    model.load_weights(args.snapshot, by_name=True, skip_mismatch=True)

    # freeze layers
    if args.freeze_backbone:
        for i in range(86):
            model.layers[i].trainable = False

    # compile model
    model.compile(optimizer=Adam(lr=1e-4), loss={'centernet_loss': lambda y_true, y_pred: y_pred})
    # model.compile(optimizer=SGD(lr=1e-6, momentum=0.9, nesterov=True, decay=1e-5), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    # print model summary
    # print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None

    # start training
    return model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        initial_epoch=0,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
        validation_data=validation_generator
    )


if __name__ == '__main__':
    main()
