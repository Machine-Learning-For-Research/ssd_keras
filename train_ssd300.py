from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from keras_ssd300 import ssd_300
from keras_ssd_loss import SSDLoss
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator

### Set up the model

# 1: Set some necessary parameters
print('Step 1:')
print('Set some necessary parameters.')

img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
n_classes = 21  # Number of classes including the background class, e.g. 21 for the Pascal VOC datasets
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
          1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets, the factors for the MS COCO dataset are smaller, namely [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
aspect_ratios = [[0.5, 1.0, 2.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [0.5, 1.0, 2.0],
                 [0.5, 1.0, 2.0]]  # The anchor box aspect ratios used in the original SSD300
two_boxes_for_ar1 = True
limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2,
             0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = True

# 2: Build the Keras model (and possibly load some trained weights)

K.clear_session()  # Clear previous models from memory.
# The output `predictor_sizes` is needed below to set up `SSDBoxEncoder`
model, predictor_sizes = ssd_300(image_size=(img_height, img_width, img_channels),
                                 n_classes=n_classes,
                                 min_scale=None,
                                 # You could pass a min scale and max scale instead of the `scales` list, but we're not doing that here
                                 max_scale=None,
                                 scales=scales,
                                 aspect_ratios_global=None,
                                 aspect_ratios_per_layer=aspect_ratios,
                                 two_boxes_for_ar1=two_boxes_for_ar1,
                                 limit_boxes=limit_boxes,
                                 variances=variances,
                                 coords=coords,
                                 normalize_coords=normalize_coords)
# model.load_weights('./ssd300_weights.h5', by_name=True) # You should load pre-trained weights for the modified VGG-16 base network here



### Set up training
print('Step 2:')
print('Set up training.')

batch_size = 32

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=0.1)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 4: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=None,
                                max_scale=None,
                                scales=scales,
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)

# 5: Create the training set batch generator

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

train_dataset = BatchGenerator(images_path='./Datasets/VOCdevkit/VOC2012/JPEGImages/',
                               include_classes='all',
                               box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

train_dataset.parse_xml(annotations_path='./Datasets/VOCdevkit/VOC2012/Annotations/',
                        image_set_path='./Datasets/VOCdevkit/VOC2012/ImageSets/Main/',
                        image_set='train.txt',
                        classes=classes,
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

train_generator = train_dataset.generate(batch_size=batch_size,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         brightness=(0.5, 2, 0.5),
                                         flip=0.5,
                                         translate=((0, 30), (0, 30), 0.5),
                                         scale=(0.75, 1.2, 0.5),
                                         random_crop=(300, 300, 1, 3),
                                         # This one is important because the Pascal VOC images vary in size
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         # While the anchor boxes are not being clipped, the ground truth boxes should be
                                         include_thresh=0.4,
                                         diagnostics=False)

n_train_samples = train_dataset.get_n_samples()  # Get the number of samples in the training dataset to compute the epoch length below

# 6: Create the validation set batch generator

val_dataset = BatchGenerator(images_path='./Datasets/VOCdevkit/VOC2012/JPEGImages/',
                             include_classes='all',
                             box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

val_dataset.parse_xml(annotations_path='./Datasets/VOCdevkit/VOC2012/Annotations/',
                      image_set_path='./Datasets/VOCdevkit/VOC2012/ImageSets/Main/',
                      image_set='val.txt',
                      classes=classes,
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     train=True,
                                     ssd_box_encoder=ssd_box_encoder,
                                     equalize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     random_crop=(300, 300, 1, 3),
                                     crop=False,
                                     resize=False,
                                     gray=False,
                                     limit_boxes=True,
                                     include_thresh=0.4,
                                     diagnostics=False)

n_val_samples = val_dataset.get_n_samples()


# 7: Define a simple learning rate schedule

def lr_schedule(epoch):
    if epoch <= 20:
        return 0.0001
    else:
        return 0.00001


### Run training
print('Step 3:')
print('Run training.')

# 7: Run trainingepoch

epochs = 10

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=ceil(n_train_samples / batch_size),
                              epochs=epochs,
                              callbacks=[ModelCheckpoint('./ssd300_0_weights_epoch{epoch:02d}_loss{loss:.4f}.h5',
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         mode='auto',
                                                         period=1),
                                         LearningRateScheduler(lr_schedule),
                                         TensorBoard(),
                                         EarlyStopping(monitor='val_loss',
                                                       min_delta=0.001,
                                                       patience=2)],
                              validation_data=val_generator,
                              validation_steps=ceil(n_val_samples / batch_size))

model_name = 'ssd300_0'
model.save('./{}.h5'.format(model_name))
model.save_weights('./{}_weights.h5'.format(model_name))

print()
print("Model saved as {}.h5".format(model_name))
print("Weights also saved separately as {}_weights.h5".format(model_name))
print()
