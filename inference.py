from generators.pascal import PascalVocGenerator
from models.resnet import centernet
import cv2
import os
import numpy as np
import time
from generators.utils import affine_transform, get_affine_transform
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
generator = PascalVocGenerator(
    'datasets/VOC2007',
    'test',
    shuffle_groups=False,
    skip_truncated=False,
    skip_difficult=True,
)
model_path = 'checkpoints/2019-11-10/pascal_81_1.5415_3.0741_0.6860_0.7057_0.7209_0.7290.h5'
num_classes = generator.num_classes()
classes = list(generator.classes.keys())
flip_test = True
nms = True
keep_resolution = False
score_threshold = 0.1
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]
model, prediction_model, debug_model = centernet(num_classes=num_classes,
                                                 nms=nms,
                                                 flip_test=flip_test,
                                                 freeze_bn=True,
                                                 score_threshold=score_threshold)
prediction_model.load_weights(model_path, by_name=True, skip_mismatch=True)
for i in range(10):
    image = generator.load_image(i)
    src_image = image.copy()

    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    s = max(image.shape[0], image.shape[1]) * 1.0

    tgt_w = generator.input_size
    tgt_h = generator.input_size
    image = generator.preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
    if flip_test:
        flipped_image = image[:, ::-1]
        inputs = np.stack([image, flipped_image], axis=0)
    else:
        inputs = np.expand_dims(image, axis=0)
    # run network
    start = time.time()
    detections = prediction_model.predict_on_batch(inputs)[0]
    print(time.time() - start)
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
    for detection in detections:
        xmin = int(round(detection[0]))
        ymin = int(round(detection[1]))
        xmax = int(round(detection[2]))
        ymax = int(round(detection[3]))
        score = '{:.4f}'.format(detection[4])
        class_id = int(detection[5])
        color = colors[class_id]
        class_name = classes[class_id]
        label = '-'.join([class_name, score])
        # ret[0] 表示包围 text 的矩形框的 width
        # ret[1] 表示包围 text 的矩形框的 height
        # baseline 表示的 text 最底下一个像素到文本 baseline 的距离
        # 文本 baseline 参考 https://blog.csdn.net/u010970514/article/details/84075776
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', src_image)
    key = cv2.waitKey(0)
    if int(key) == 121:
        image_fname = generator.image_names[i]
        cv2.imwrite('test/{}.jpg'.format(image_fname), src_image)
