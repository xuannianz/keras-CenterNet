from models.resnet import centernet
from generators.pascal import PascalVocGenerator
from keras.optimizers import Adam

from losses import loss

model, prediction_model = centernet(num_classes=20, backbone='resnet50')
train_generator = PascalVocGenerator(data_dir='datasets/VOC0712', set_name='train', skip_difficult=True)
val_generator = PascalVocGenerator(data_dir='datasets/VOC0712', set_name='val', skip_difficult=True)

model.compile(optimizer=Adam(lr=1e-3), loss=loss)
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=1,
    initial_epoch=0,
    epochs=1,
    verbose=1,
)
