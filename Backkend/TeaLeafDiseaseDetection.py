#EDA

from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
     
Mounted at /content/drive

dataset_dir = "/content/drive/MyDrive/TeaDiseaseDataset"

class_counts = {}
for cls in os.listdir(dataset_dir):
    cls_path = os.path.join(dataset_dir, cls)
    if os.path.isdir(cls_path):
        class_counts[cls] = len(os.listdir(cls_path))

plt.figure(figsize=(10,5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=45)
plt.title("Number of images per class")
plt.ylabel("Count")
plt.show()

class_counts

     

{'Anthracnose': 100,
 'AlgalLeaf': 113,
 'RedLeafSpot': 143,
 'WhiteSpot': 142,
 'BirdEyeSpot': 100,
 'BrownBlight': 113,
 'GrayBlight': 100,
 'Healthy': 74}

import random
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def show_samples_raw(dataset_dir, n=4):
    classes = sorted(os.listdir(dataset_dir))
    for cls in classes:
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        imgs = random.sample(os.listdir(cls_path), n)
        plt.figure(figsize=(12,3))
        for i, img_name in enumerate(imgs):
            img_path = os.path.join(cls_path, img_name)
            img = image.load_img(img_path)
            plt.subplot(1,n,i+1)
            plt.imshow(img)
            plt.axis("off")
        plt.suptitle(f"Raw samples: {cls}")
        plt.show()

show_samples_raw(dataset_dir)

from PIL import Image
import random

sizes = []

for cls in os.listdir(dataset_dir):
    cls_path = os.path.join(dataset_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    imgs = random.sample(os.listdir(cls_path), min(20, len(os.listdir(cls_path))))
    for img_name in imgs:
        img_path = os.path.join(cls_path, img_name)
        with Image.open(img_path) as im:
            sizes.append(im.size)

widths = [s[0] for s in sizes]
heights = [s[1] for s in sizes]

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.hist(widths, bins=20)
plt.title("Original Image Width Distribution")

plt.subplot(1,2,2)
plt.hist(heights, bins=20)
plt.title("Original Image Height Distribution")
plt.show()

     


ratios = [w/h for w,h in sizes]

plt.hist(ratios, bins=20)
plt.title("Aspect Ratio Distribution (Raw Images)")
plt.show()

     


bad_files = []

for cls in os.listdir(dataset_dir):
    cls_path = os.path.join(dataset_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    for f in os.listdir(cls_path):
        path = os.path.join(cls_path, f)
        try:
            with Image.open(path) as im:
                im.verify()
        except:
            bad_files.append(path)

print("Corrupted images:", len(bad_files))

     
Corrupted images: 0

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

pixel_values = []

for cls in os.listdir(dataset_dir):
    cls_path = os.path.join(dataset_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    imgs = os.listdir(cls_path)

    # only sample few images per class
    sample_imgs = random.sample(imgs, min(5, len(imgs)))

    for img_name in sample_imgs:
        img_path = os.path.join(cls_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # sample pixels instead of all pixels
            pixel_values.extend(img.flatten()[::10])

plt.hist(pixel_values, bins=50)
plt.title("Pixel Intensity Distribution (Sampled)")
plt.xlabel("Pixel value (0-255)")
plt.ylabel("Frequency")
plt.show()

hist_total = np.zeros(256)

for cls in os.listdir(dataset_dir):
    cls_path = os.path.join(dataset_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    imgs = random.sample(os.listdir(cls_path), 5)

    for img_name in imgs:
        img = cv2.imread(os.path.join(cls_path, img_name), 0)
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        hist_total += hist.flatten()

plt.plot(hist_total)
plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.show()

     

#Pre-processing


base_output = "/content/drive/MyDrive/TeaDataset_Split"
train_dir = os.path.join(base_output, "Train")
val_dir   = os.path.join(base_output, "Valid")
test_dir  = os.path.join(base_output, "Test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

split_ratio = (0.7, 0.15, 0.15)

for cls in os.listdir(dataset_dir):
    cls_path = os.path.join(dataset_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(split_ratio[0] * total)
    val_end   = train_end + int(split_ratio[1] * total)

    splits = {
        train_dir: images[:train_end],
        val_dir: images[train_end:val_end],
        test_dir: images[val_end:]
    }

    for split_dir, img_list in splits.items():
        cls_split_dir = os.path.join(split_dir, cls)
        os.makedirs(cls_split_dir, exist_ok=True)

        for img in img_list:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(cls_split_dir, img)
            )


train_dir = "/content/drive/MyDrive/TeaDataset_Split/Train"
val_dir   = "/content/drive/MyDrive/TeaDataset_Split/Valid"
test_dir  = "/content/drive/MyDrive/TeaDataset_Split/Test"


img_height = 224
img_width  = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    #rotation_range=20,
    #zoom_range=0.15,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #horizontal_flip=True

    #rotation_range=30,
    #zoom_range=0.25,
    #width_shift_range=0.15,
    #height_shift_range=0.15,
    #brightness_range=[0.7, 1.3],
    #shear_range=0.15,
    #horizontal_flip=True,
    #fill_mode="nearest"

    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_gen.num_classes

     
Found 618 images belonging to 8 classes.
Found 130 images belonging to 8 classes.
Found 137 images belonging to 8 classes.


# balancing imbalanced training dataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Get labels from generator
classes = train_gen.classes
class_labels = np.unique(classes)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=class_labels,
    y=classes
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

     
Found 618 images belonging to 8 classes.
Class Weights: {0: np.float64(0.9778481012658228), 1: np.float64(1.1035714285714286), 2: np.float64(1.1035714285714286), 3: np.float64(0.9778481012658228), 4: np.float64(1.1035714285714286), 5: np.float64(1.5147058823529411), 6: np.float64(0.7725), 7: np.float64(0.7803030303030303)}

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

base_model.trainable = False  # freeze backbone

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)

x = Dropout(0.35)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    #loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

     
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
9406464/9406464 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (None, 224, 224,  │          0 │ -                 │
│ (InputLayer)        │ 3)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ Conv1 (Conv2D)      │ (None, 112, 112,  │        864 │ input_layer[0][0] │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_Conv1            │ (None, 112, 112,  │        128 │ Conv1[0][0]       │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ Conv1_relu (ReLU)   │ (None, 112, 112,  │          0 │ bn_Conv1[0][0]    │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ expanded_conv_dept… │ (None, 112, 112,  │        288 │ Conv1_relu[0][0]  │
│ (DepthwiseConv2D)   │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ expanded_conv_dept… │ (None, 112, 112,  │        128 │ expanded_conv_de… │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ expanded_conv_dept… │ (None, 112, 112,  │          0 │ expanded_conv_de… │
│ (ReLU)              │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ expanded_conv_proj… │ (None, 112, 112,  │        512 │ expanded_conv_de… │
│ (Conv2D)            │ 16)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ expanded_conv_proj… │ (None, 112, 112,  │         64 │ expanded_conv_pr… │
│ (BatchNormalizatio… │ 16)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_expand      │ (None, 112, 112,  │      1,536 │ expanded_conv_pr… │
│ (Conv2D)            │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_expand_BN   │ (None, 112, 112,  │        384 │ block_1_expand[0… │
│ (BatchNormalizatio… │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_expand_relu │ (None, 112, 112,  │          0 │ block_1_expand_B… │
│ (ReLU)              │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_pad         │ (None, 113, 113,  │          0 │ block_1_expand_r… │
│ (ZeroPadding2D)     │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_depthwise   │ (None, 56, 56,    │        864 │ block_1_pad[0][0] │
│ (DepthwiseConv2D)   │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_depthwise_… │ (None, 56, 56,    │        384 │ block_1_depthwis… │
│ (BatchNormalizatio… │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_depthwise_… │ (None, 56, 56,    │          0 │ block_1_depthwis… │
│ (ReLU)              │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_project     │ (None, 56, 56,    │      2,304 │ block_1_depthwis… │
│ (Conv2D)            │ 24)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_project_BN  │ (None, 56, 56,    │         96 │ block_1_project[… │
│ (BatchNormalizatio… │ 24)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_expand      │ (None, 56, 56,    │      3,456 │ block_1_project_… │
│ (Conv2D)            │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_expand_BN   │ (None, 56, 56,    │        576 │ block_2_expand[0… │
│ (BatchNormalizatio… │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_expand_relu │ (None, 56, 56,    │          0 │ block_2_expand_B… │
│ (ReLU)              │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_depthwise   │ (None, 56, 56,    │      1,296 │ block_2_expand_r… │
│ (DepthwiseConv2D)   │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_depthwise_… │ (None, 56, 56,    │        576 │ block_2_depthwis… │
│ (BatchNormalizatio… │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_depthwise_… │ (None, 56, 56,    │          0 │ block_2_depthwis… │
│ (ReLU)              │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_project     │ (None, 56, 56,    │      3,456 │ block_2_depthwis… │
│ (Conv2D)            │ 24)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_project_BN  │ (None, 56, 56,    │         96 │ block_2_project[… │
│ (BatchNormalizatio… │ 24)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_add (Add)   │ (None, 56, 56,    │          0 │ block_1_project_… │
│                     │ 24)               │            │ block_2_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_expand      │ (None, 56, 56,    │      3,456 │ block_2_add[0][0] │
│ (Conv2D)            │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_expand_BN   │ (None, 56, 56,    │        576 │ block_3_expand[0… │
│ (BatchNormalizatio… │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_expand_relu │ (None, 56, 56,    │          0 │ block_3_expand_B… │
│ (ReLU)              │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_pad         │ (None, 57, 57,    │          0 │ block_3_expand_r… │
│ (ZeroPadding2D)     │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_depthwise   │ (None, 28, 28,    │      1,296 │ block_3_pad[0][0] │
│ (DepthwiseConv2D)   │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_depthwise_… │ (None, 28, 28,    │        576 │ block_3_depthwis… │
│ (BatchNormalizatio… │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_depthwise_… │ (None, 28, 28,    │          0 │ block_3_depthwis… │
│ (ReLU)              │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_project     │ (None, 28, 28,    │      4,608 │ block_3_depthwis… │
│ (Conv2D)            │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_project_BN  │ (None, 28, 28,    │        128 │ block_3_project[… │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_expand      │ (None, 28, 28,    │      6,144 │ block_3_project_… │
│ (Conv2D)            │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_expand_BN   │ (None, 28, 28,    │        768 │ block_4_expand[0… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_expand_relu │ (None, 28, 28,    │          0 │ block_4_expand_B… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_depthwise   │ (None, 28, 28,    │      1,728 │ block_4_expand_r… │
│ (DepthwiseConv2D)   │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_depthwise_… │ (None, 28, 28,    │        768 │ block_4_depthwis… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_depthwise_… │ (None, 28, 28,    │          0 │ block_4_depthwis… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_project     │ (None, 28, 28,    │      6,144 │ block_4_depthwis… │
│ (Conv2D)            │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_project_BN  │ (None, 28, 28,    │        128 │ block_4_project[… │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_add (Add)   │ (None, 28, 28,    │          0 │ block_3_project_… │
│                     │ 32)               │            │ block_4_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_expand      │ (None, 28, 28,    │      6,144 │ block_4_add[0][0] │
│ (Conv2D)            │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_expand_BN   │ (None, 28, 28,    │        768 │ block_5_expand[0… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_expand_relu │ (None, 28, 28,    │          0 │ block_5_expand_B… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_depthwise   │ (None, 28, 28,    │      1,728 │ block_5_expand_r… │
│ (DepthwiseConv2D)   │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_depthwise_… │ (None, 28, 28,    │        768 │ block_5_depthwis… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_depthwise_… │ (None, 28, 28,    │          0 │ block_5_depthwis… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_project     │ (None, 28, 28,    │      6,144 │ block_5_depthwis… │
│ (Conv2D)            │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_project_BN  │ (None, 28, 28,    │        128 │ block_5_project[… │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_add (Add)   │ (None, 28, 28,    │          0 │ block_4_add[0][0… │
│                     │ 32)               │            │ block_5_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_expand      │ (None, 28, 28,    │      6,144 │ block_5_add[0][0] │
│ (Conv2D)            │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_expand_BN   │ (None, 28, 28,    │        768 │ block_6_expand[0… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_expand_relu │ (None, 28, 28,    │          0 │ block_6_expand_B… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_pad         │ (None, 29, 29,    │          0 │ block_6_expand_r… │
│ (ZeroPadding2D)     │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_depthwise   │ (None, 14, 14,    │      1,728 │ block_6_pad[0][0] │
│ (DepthwiseConv2D)   │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_depthwise_… │ (None, 14, 14,    │        768 │ block_6_depthwis… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_depthwise_… │ (None, 14, 14,    │          0 │ block_6_depthwis… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_project     │ (None, 14, 14,    │     12,288 │ block_6_depthwis… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_project_BN  │ (None, 14, 14,    │        256 │ block_6_project[… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_expand      │ (None, 14, 14,    │     24,576 │ block_6_project_… │
│ (Conv2D)            │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_expand_BN   │ (None, 14, 14,    │      1,536 │ block_7_expand[0… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_expand_relu │ (None, 14, 14,    │          0 │ block_7_expand_B… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_depthwise   │ (None, 14, 14,    │      3,456 │ block_7_expand_r… │
│ (DepthwiseConv2D)   │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_depthwise_… │ (None, 14, 14,    │      1,536 │ block_7_depthwis… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_depthwise_… │ (None, 14, 14,    │          0 │ block_7_depthwis… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_project     │ (None, 14, 14,    │     24,576 │ block_7_depthwis… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_project_BN  │ (None, 14, 14,    │        256 │ block_7_project[… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_add (Add)   │ (None, 14, 14,    │          0 │ block_6_project_… │
│                     │ 64)               │            │ block_7_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_expand      │ (None, 14, 14,    │     24,576 │ block_7_add[0][0] │
│ (Conv2D)            │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_expand_BN   │ (None, 14, 14,    │      1,536 │ block_8_expand[0… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_expand_relu │ (None, 14, 14,    │          0 │ block_8_expand_B… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_depthwise   │ (None, 14, 14,    │      3,456 │ block_8_expand_r… │
│ (DepthwiseConv2D)   │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_depthwise_… │ (None, 14, 14,    │      1,536 │ block_8_depthwis… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_depthwise_… │ (None, 14, 14,    │          0 │ block_8_depthwis… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_project     │ (None, 14, 14,    │     24,576 │ block_8_depthwis… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_project_BN  │ (None, 14, 14,    │        256 │ block_8_project[… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_add (Add)   │ (None, 14, 14,    │          0 │ block_7_add[0][0… │
│                     │ 64)               │            │ block_8_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_expand      │ (None, 14, 14,    │     24,576 │ block_8_add[0][0] │
│ (Conv2D)            │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_expand_BN   │ (None, 14, 14,    │      1,536 │ block_9_expand[0… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_expand_relu │ (None, 14, 14,    │          0 │ block_9_expand_B… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_depthwise   │ (None, 14, 14,    │      3,456 │ block_9_expand_r… │
│ (DepthwiseConv2D)   │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_depthwise_… │ (None, 14, 14,    │      1,536 │ block_9_depthwis… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_depthwise_… │ (None, 14, 14,    │          0 │ block_9_depthwis… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_project     │ (None, 14, 14,    │     24,576 │ block_9_depthwis… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_project_BN  │ (None, 14, 14,    │        256 │ block_9_project[… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_add (Add)   │ (None, 14, 14,    │          0 │ block_8_add[0][0… │
│                     │ 64)               │            │ block_9_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_expand     │ (None, 14, 14,    │     24,576 │ block_9_add[0][0] │
│ (Conv2D)            │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_expand_BN  │ (None, 14, 14,    │      1,536 │ block_10_expand[… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_expand_re… │ (None, 14, 14,    │          0 │ block_10_expand_… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_depthwise  │ (None, 14, 14,    │      3,456 │ block_10_expand_… │
│ (DepthwiseConv2D)   │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_depthwise… │ (None, 14, 14,    │      1,536 │ block_10_depthwi… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_depthwise… │ (None, 14, 14,    │          0 │ block_10_depthwi… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_project    │ (None, 14, 14,    │     36,864 │ block_10_depthwi… │
│ (Conv2D)            │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_project_BN │ (None, 14, 14,    │        384 │ block_10_project… │
│ (BatchNormalizatio… │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_expand     │ (None, 14, 14,    │     55,296 │ block_10_project… │
│ (Conv2D)            │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_expand_BN  │ (None, 14, 14,    │      2,304 │ block_11_expand[… │
│ (BatchNormalizatio… │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_expand_re… │ (None, 14, 14,    │          0 │ block_11_expand_… │
│ (ReLU)              │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_depthwise  │ (None, 14, 14,    │      5,184 │ block_11_expand_… │
│ (DepthwiseConv2D)   │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_depthwise… │ (None, 14, 14,    │      2,304 │ block_11_depthwi… │
│ (BatchNormalizatio… │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_depthwise… │ (None, 14, 14,    │          0 │ block_11_depthwi… │
│ (ReLU)              │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_project    │ (None, 14, 14,    │     55,296 │ block_11_depthwi… │
│ (Conv2D)            │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_project_BN │ (None, 14, 14,    │        384 │ block_11_project… │
│ (BatchNormalizatio… │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_add (Add)  │ (None, 14, 14,    │          0 │ block_10_project… │
│                     │ 96)               │            │ block_11_project… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_expand     │ (None, 14, 14,    │     55,296 │ block_11_add[0][… │
│ (Conv2D)            │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_expand_BN  │ (None, 14, 14,    │      2,304 │ block_12_expand[… │
│ (BatchNormalizatio… │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_expand_re… │ (None, 14, 14,    │          0 │ block_12_expand_… │
│ (ReLU)              │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_depthwise  │ (None, 14, 14,    │      5,184 │ block_12_expand_… │
│ (DepthwiseConv2D)   │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_depthwise… │ (None, 14, 14,    │      2,304 │ block_12_depthwi… │
│ (BatchNormalizatio… │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_depthwise… │ (None, 14, 14,    │          0 │ block_12_depthwi… │
│ (ReLU)              │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_project    │ (None, 14, 14,    │     55,296 │ block_12_depthwi… │
│ (Conv2D)            │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_project_BN │ (None, 14, 14,    │        384 │ block_12_project… │
│ (BatchNormalizatio… │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_add (Add)  │ (None, 14, 14,    │          0 │ block_11_add[0][… │
│                     │ 96)               │            │ block_12_project… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_expand     │ (None, 14, 14,    │     55,296 │ block_12_add[0][… │
│ (Conv2D)            │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_expand_BN  │ (None, 14, 14,    │      2,304 │ block_13_expand[… │
│ (BatchNormalizatio… │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_expand_re… │ (None, 14, 14,    │          0 │ block_13_expand_… │
│ (ReLU)              │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_pad        │ (None, 15, 15,    │          0 │ block_13_expand_… │
│ (ZeroPadding2D)     │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_depthwise  │ (None, 7, 7, 576) │      5,184 │ block_13_pad[0][… │
│ (DepthwiseConv2D)   │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_depthwise… │ (None, 7, 7, 576) │      2,304 │ block_13_depthwi… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_depthwise… │ (None, 7, 7, 576) │          0 │ block_13_depthwi… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_project    │ (None, 7, 7, 160) │     92,160 │ block_13_depthwi… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_project_BN │ (None, 7, 7, 160) │        640 │ block_13_project… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_expand     │ (None, 7, 7, 960) │    153,600 │ block_13_project… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_expand_BN  │ (None, 7, 7, 960) │      3,840 │ block_14_expand[… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_expand_re… │ (None, 7, 7, 960) │          0 │ block_14_expand_… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_depthwise  │ (None, 7, 7, 960) │      8,640 │ block_14_expand_… │
│ (DepthwiseConv2D)   │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_depthwise… │ (None, 7, 7, 960) │      3,840 │ block_14_depthwi… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_depthwise… │ (None, 7, 7, 960) │          0 │ block_14_depthwi… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_project    │ (None, 7, 7, 160) │    153,600 │ block_14_depthwi… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_project_BN │ (None, 7, 7, 160) │        640 │ block_14_project… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_add (Add)  │ (None, 7, 7, 160) │          0 │ block_13_project… │
│                     │                   │            │ block_14_project… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_expand     │ (None, 7, 7, 960) │    153,600 │ block_14_add[0][… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_expand_BN  │ (None, 7, 7, 960) │      3,840 │ block_15_expand[… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_expand_re… │ (None, 7, 7, 960) │          0 │ block_15_expand_… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_depthwise  │ (None, 7, 7, 960) │      8,640 │ block_15_expand_… │
│ (DepthwiseConv2D)   │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_depthwise… │ (None, 7, 7, 960) │      3,840 │ block_15_depthwi… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_depthwise… │ (None, 7, 7, 960) │          0 │ block_15_depthwi… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_project    │ (None, 7, 7, 160) │    153,600 │ block_15_depthwi… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_project_BN │ (None, 7, 7, 160) │        640 │ block_15_project… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_add (Add)  │ (None, 7, 7, 160) │          0 │ block_14_add[0][… │
│                     │                   │            │ block_15_project… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_expand     │ (None, 7, 7, 960) │    153,600 │ block_15_add[0][… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_expand_BN  │ (None, 7, 7, 960) │      3,840 │ block_16_expand[… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_expand_re… │ (None, 7, 7, 960) │          0 │ block_16_expand_… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_depthwise  │ (None, 7, 7, 960) │      8,640 │ block_16_expand_… │
│ (DepthwiseConv2D)   │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_depthwise… │ (None, 7, 7, 960) │      3,840 │ block_16_depthwi… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_depthwise… │ (None, 7, 7, 960) │          0 │ block_16_depthwi… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_project    │ (None, 7, 7, 320) │    307,200 │ block_16_depthwi… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_project_BN │ (None, 7, 7, 320) │      1,280 │ block_16_project… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ Conv_1 (Conv2D)     │ (None, 7, 7,      │    409,600 │ block_16_project… │
│                     │ 1280)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ Conv_1_bn           │ (None, 7, 7,      │      5,120 │ Conv_1[0][0]      │
│ (BatchNormalizatio… │ 1280)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ out_relu (ReLU)     │ (None, 7, 7,      │          0 │ Conv_1_bn[0][0]   │
│                     │ 1280)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (None, 1280)      │          0 │ out_relu[0][0]    │
│ (GlobalAveragePool… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout (Dropout)   │ (None, 1280)      │          0 │ global_average_p… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense (Dense)       │ (None, 128)       │    163,968 │ dropout[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_1 (Dropout) │ (None, 128)       │          0 │ dense[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (Dense)     │ (None, 8)         │      1,032 │ dropout_1[0][0]   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 2,422,984 (9.24 MB)
 Trainable params: 165,000 (644.53 KB)
 Non-trainable params: 2,257,984 (8.61 MB)


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "MobileNetV2_TeaDisease.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

epochs = 25

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)


     
/usr/local/lib/python3.12/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 11s/step - accuracy: 0.2221 - loss: 2.4626 
Epoch 1: val_accuracy improved from -inf to 0.55385, saving model to MobileNetV2_TeaDisease.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
20/20 ━━━━━━━━━━━━━━━━━━━━ 312s 15s/step - accuracy: 0.2251 - loss: 2.4464 - val_accuracy: 0.5538 - val_loss: 1.3906
Epoch 2/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.4989 - loss: 1.4816
Epoch 2: val_accuracy improved from 0.55385 to 0.70000, saving model to MobileNetV2_TeaDisease.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
20/20 ━━━━━━━━━━━━━━━━━━━━ 37s 2s/step - accuracy: 0.4988 - loss: 1.4800 - val_accuracy: 0.7000 - val_loss: 1.1507
Epoch 3/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.5667 - loss: 1.3339
Epoch 3: val_accuracy improved from 0.70000 to 0.71538, saving model to MobileNetV2_TeaDisease.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.5678 - loss: 1.3303 - val_accuracy: 0.7154 - val_loss: 1.0448
Epoch 4/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.6412 - loss: 1.2570
Epoch 4: val_accuracy improved from 0.71538 to 0.77692, saving model to MobileNetV2_TeaDisease.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
20/20 ━━━━━━━━━━━━━━━━━━━━ 37s 2s/step - accuracy: 0.6413 - loss: 1.2542 - val_accuracy: 0.7769 - val_loss: 1.0138
Epoch 5/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.6918 - loss: 1.0927
Epoch 5: val_accuracy did not improve from 0.77692
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.6915 - loss: 1.0927 - val_accuracy: 0.7538 - val_loss: 0.9907
Epoch 6/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.7097 - loss: 1.0666
Epoch 6: val_accuracy did not improve from 0.77692
20/20 ━━━━━━━━━━━━━━━━━━━━ 35s 2s/step - accuracy: 0.7107 - loss: 1.0652 - val_accuracy: 0.7692 - val_loss: 0.9467
Epoch 7/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7330 - loss: 0.9941
Epoch 7: val_accuracy did not improve from 0.77692
20/20 ━━━━━━━━━━━━━━━━━━━━ 37s 2s/step - accuracy: 0.7325 - loss: 0.9937 - val_accuracy: 0.7615 - val_loss: 0.9276
Epoch 8/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7347 - loss: 0.9781
Epoch 8: val_accuracy did not improve from 0.77692
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.7362 - loss: 0.9767 - val_accuracy: 0.7692 - val_loss: 0.9095
Epoch 9/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7676 - loss: 0.9108
Epoch 9: val_accuracy improved from 0.77692 to 0.79231, saving model to MobileNetV2_TeaDisease.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
20/20 ━━━━━━━━━━━━━━━━━━━━ 37s 2s/step - accuracy: 0.7673 - loss: 0.9127 - val_accuracy: 0.7923 - val_loss: 0.8963
Epoch 10/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.7748 - loss: 0.9582
Epoch 10: val_accuracy improved from 0.79231 to 0.80769, saving model to MobileNetV2_TeaDisease.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.7749 - loss: 0.9578 - val_accuracy: 0.8077 - val_loss: 0.9156
Epoch 11/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7525 - loss: 0.9474
Epoch 11: val_accuracy improved from 0.80769 to 0.81538, saving model to MobileNetV2_TeaDisease.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
20/20 ━━━━━━━━━━━━━━━━━━━━ 37s 2s/step - accuracy: 0.7530 - loss: 0.9468 - val_accuracy: 0.8154 - val_loss: 0.8833
Epoch 12/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7895 - loss: 0.8629
Epoch 12: val_accuracy did not improve from 0.81538
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.7897 - loss: 0.8637 - val_accuracy: 0.8154 - val_loss: 0.8611
Epoch 13/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7721 - loss: 0.9001
Epoch 13: val_accuracy improved from 0.81538 to 0.83846, saving model to MobileNetV2_TeaDisease.h5
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.7722 - loss: 0.8996 - val_accuracy: 0.8385 - val_loss: 0.8427
Epoch 14/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.7835 - loss: 0.9171
Epoch 14: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.7834 - loss: 0.9165 - val_accuracy: 0.7769 - val_loss: 0.8908
Epoch 15/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7653 - loss: 0.8989
Epoch 15: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.7665 - loss: 0.8975 - val_accuracy: 0.7923 - val_loss: 0.9031
Epoch 16/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7987 - loss: 0.8751
Epoch 16: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 37s 2s/step - accuracy: 0.7992 - loss: 0.8746 - val_accuracy: 0.8308 - val_loss: 0.8569
Epoch 17/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.8102 - loss: 0.8579
Epoch 17: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.8102 - loss: 0.8571 - val_accuracy: 0.8385 - val_loss: 0.8471
Epoch 18/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.8366 - loss: 0.7814
Epoch 18: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 34s 2s/step - accuracy: 0.8363 - loss: 0.7823 - val_accuracy: 0.8000 - val_loss: 0.8903
Epoch 19/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7972 - loss: 0.8509
Epoch 19: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.7985 - loss: 0.8494 - val_accuracy: 0.8000 - val_loss: 0.8587
Epoch 20/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.8669 - loss: 0.7445
Epoch 20: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 35s 2s/step - accuracy: 0.8665 - loss: 0.7452 - val_accuracy: 0.7846 - val_loss: 0.8676
Epoch 21/25
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.8384 - loss: 0.8082
Epoch 21: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 35s 2s/step - accuracy: 0.8380 - loss: 0.8075 - val_accuracy: 0.7846 - val_loss: 0.9220


# Unfreeze the top layers of MobileNetV2
base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_epochs = 20
history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=fine_tune_epochs,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)
     
Epoch 1/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.4679 - loss: 1.6364
Epoch 1: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 79s 3s/step - accuracy: 0.4689 - loss: 1.6312 - val_accuracy: 0.8308 - val_loss: 0.6519
Epoch 2/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.5520 - loss: 1.3806
Epoch 2: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 37s 2s/step - accuracy: 0.5519 - loss: 1.3791 - val_accuracy: 0.8308 - val_loss: 0.6494
Epoch 3/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.6006 - loss: 1.1877
Epoch 3: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 35s 2s/step - accuracy: 0.6019 - loss: 1.1855 - val_accuracy: 0.8000 - val_loss: 0.6508
Epoch 4/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.6574 - loss: 1.0679
Epoch 4: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 39s 2s/step - accuracy: 0.6567 - loss: 1.0674 - val_accuracy: 0.8077 - val_loss: 0.6538
Epoch 5/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.6733 - loss: 1.0160
Epoch 5: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.6740 - loss: 1.0131 - val_accuracy: 0.7846 - val_loss: 0.6552
Epoch 6/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7310 - loss: 0.9055
Epoch 6: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.7311 - loss: 0.9054 - val_accuracy: 0.7692 - val_loss: 0.6573
Epoch 7/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.7601 - loss: 0.7842
Epoch 7: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 35s 2s/step - accuracy: 0.7595 - loss: 0.7854 - val_accuracy: 0.7846 - val_loss: 0.6602
Epoch 8/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7352 - loss: 0.7897
Epoch 8: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 38s 2s/step - accuracy: 0.7359 - loss: 0.7881 - val_accuracy: 0.7846 - val_loss: 0.6619
Epoch 9/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.7857 - loss: 0.7109
Epoch 9: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 37s 2s/step - accuracy: 0.7852 - loss: 0.7116 - val_accuracy: 0.7692 - val_loss: 0.6655
Epoch 10/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.7481 - loss: 0.7221
Epoch 10: val_accuracy did not improve from 0.83846
20/20 ━━━━━━━━━━━━━━━━━━━━ 36s 2s/step - accuracy: 0.7496 - loss: 0.7213 - val_accuracy: 0.7769 - val_loss: 0.6652

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train (head)")
plt.plot(history.history["val_accuracy"], label="Val (head)")
plt.plot(history_fine.history["accuracy"], label="Train (fine-tune)")
plt.plot(history_fine.history["val_accuracy"], label="Val (fine-tune)")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train (head)")
plt.plot(history.history["val_loss"], label="Val (head)")
plt.plot(history_fine.history["loss"], label="Train (fine-tune)")
plt.plot(history_fine.history["val_loss"], label="Val (fine-tune)")
plt.title("Loss")
plt.legend()
plt.show()


#  Evaluate on Train Set
train_loss, train_acc = model.evaluate(train_gen, verbose=1)
print(f"Train Accuracy: {train_acc*100:.2f}%")

# Evaluate on Validation Set
val_loss, val_acc = model.evaluate(val_gen, verbose=1)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# Evaluate on Test Set
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"Test Accuracy: {test_acc*100:.2f}%")


     
20/20 ━━━━━━━━━━━━━━━━━━━━ 33s 2s/step - accuracy: 0.8901 - loss: 0.4866
Train Accuracy: 87.22%
5/5 ━━━━━━━━━━━━━━━━━━━━ 6s 1s/step - accuracy: 0.8103 - loss: 0.6422
Validation Accuracy: 83.08%
5/5 ━━━━━━━━━━━━━━━━━━━━ 50s 12s/step - accuracy: 0.7783 - loss: 0.6900
Test Accuracy: 81.02%


y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)

cm = confusion_matrix(y_true, y_pred)
class_names = list(test_gen.class_indices.keys())

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))


     
5/5 ━━━━━━━━━━━━━━━━━━━━ 13s 2s/step

              precision    recall  f1-score   support

   AlgalLeaf       0.86      0.67      0.75        18
 Anthracnose       0.91      0.67      0.77        15
 BirdEyeSpot       0.68      0.87      0.76        15
 BrownBlight       0.78      0.78      0.78        18
  GrayBlight       0.75      0.80      0.77        15
     Healthy       0.92      1.00      0.96        12
 RedLeafSpot       1.00      0.95      0.98        22
   WhiteSpot       0.68      0.77      0.72        22

    accuracy                           0.81       137
   macro avg       0.82      0.81      0.81       137
weighted avg       0.82      0.81      0.81       137


from tensorflow.keras.preprocessing import image

img_path = "/content/drive/MyDrive/DiseaseDataset/Test/Healthy/Healthy603.jpg"

img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

pred = model.predict(img_array)
pred_class = class_names[np.argmax(pred)]
confidence = np.max(pred)

print("Predicted:", pred_class)
print("Confidence:", confidence)

     
---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
/tmp/ipython-input-1934640541.py in <cell line: 0>()
----> 1 from tensorflow.keras.preprocessing import image
      2 
      3 img_path = "/content/drive/MyDrive/DiseaseDataset/Test/Healthy/Healthy603.jpg"
      4 
      5 img = image.load_img(img_path, target_size=(img_height, img_width))

/usr/local/lib/python3.12/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.12/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch
---> 11 from tensorflow._api.v2.__internal__ import distribute
     12 from tensorflow._api.v2.__internal__ import eager_context
     13 from tensorflow._api.v2.__internal__ import feature_column

/usr/local/lib/python3.12/dist-packages/tensorflow/_api/v2/__internal__/distribute/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__.distribute import combinations
      9 from tensorflow._api.v2.__internal__.distribute import interim
     10 from tensorflow._api.v2.__internal__.distribute import multi_process_runner

/usr/local/lib/python3.12/dist-packages/tensorflow/_api/v2/__internal__/distribute/combinations/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.distribute.combinations import env # line: 456
      9 from tensorflow.python.distribute.combinations import generate # line: 365
     10 from tensorflow.python.distribute.combinations import in_main_process # line: 418

/usr/local/lib/python3.12/dist-packages/tensorflow/python/distribute/combinations.py in <module>
     31 
     32 from tensorflow.python.client import session
---> 33 from tensorflow.python.distribute import collective_all_reduce_strategy
     34 from tensorflow.python.distribute import distribute_lib
     35 from tensorflow.python.distribute import multi_process_runner

/usr/local/lib/python3.12/dist-packages/tensorflow/python/distribute/collective_all_reduce_strategy.py in <module>
     31 from tensorflow.python.distribute import input_lib
     32 from tensorflow.python.distribute import input_util
---> 33 from tensorflow.python.distribute import mirrored_strategy
     34 from tensorflow.python.distribute import multi_worker_util
     35 from tensorflow.python.distribute import numpy_dataset

/usr/local/lib/python3.12/dist-packages/tensorflow/python/distribute/mirrored_strategy.py in <module>
     32 from tensorflow.python.distribute import values
     33 from tensorflow.python.distribute import values_util
---> 34 from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
     35 from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
     36 from tensorflow.python.eager import context

/usr/local/lib/python3.12/dist-packages/tensorflow/python/distribute/cluster_resolver/__init__.py in <module>
     25 from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
     26 from tensorflow.python.distribute.cluster_resolver.cluster_resolver import UnionClusterResolver
---> 27 from tensorflow.python.distribute.cluster_resolver.gce_cluster_resolver import GCEClusterResolver
     28 from tensorflow.python.distribute.cluster_resolver.kubernetes_cluster_resolver import KubernetesClusterResolver
     29 from tensorflow.python.distribute.cluster_resolver.slurm_cluster_resolver import SlurmClusterResolver

/usr/local/lib/python3.12/dist-packages/tensorflow/python/distribute/cluster_resolver/gce_cluster_resolver.py in <module>
     22 _GOOGLE_API_CLIENT_INSTALLED = True
     23 try:
---> 24   from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
     25   from oauth2client.client import GoogleCredentials  # pylint: disable=g-import-not-at-top
     26 except ImportError:

/usr/local/lib/python3.12/dist-packages/googleapiclient/discovery.py in <module>
     40 import urllib
     41 
---> 42 import google.api_core.client_options
     43 from google.auth.exceptions import MutualTLSChannelError
     44 from google.auth.transport import mtls

/usr/local/lib/python3.12/dist-packages/google/api_core/__init__.py in <module>
     38 
     39 # perform version checks against api_core, and emit warnings if needed
---> 40 check_python_version(package="google.api_core")
     41 check_dependency_versions("google.api_core")

/usr/local/lib/python3.12/dist-packages/google/api_core/_python_version_support.py in check_python_version(package, today)
    208     """
    209     today = today or datetime.date.today()
--> 210     package_label, _ = _get_distribution_and_import_packages(package)
    211 
    212     python_version = sys.version_info

/usr/local/lib/python3.12/dist-packages/google/api_core/_python_version_support.py in _get_distribution_and_import_packages(import_package)
    187 def _get_distribution_and_import_packages(import_package: str) -> Tuple[str, Any]:
    188     """Return a pretty string with distribution & import package names."""
--> 189     distribution_package = _get_pypi_package_name(import_package)
    190     dependency_distribution_and_import_packages = (
    191         f"package {distribution_package} ({import_package})"

/usr/local/lib/python3.12/dist-packages/google/api_core/_python_version_support.py in _get_pypi_package_name(module_name)
    169         try:
    170             # Get the mapping of modules to distributions
--> 171             module_to_distributions = metadata.packages_distributions()
    172 
    173             # Check if the module is found in the mapping

/usr/lib/python3.12/importlib/metadata/__init__.py in packages_distributions()
    946     for dist in distributions():
    947         for pkg in _top_level_declared(dist) or _top_level_inferred(dist):
--> 948             pkg_to_dist[pkg].append(dist.metadata['Name'])
    949     return dict(pkg_to_dist)
    950 

/usr/lib/python3.12/importlib/metadata/__init__.py in metadata(self)
    450         )
    451         text = cast(str, opt_text)
--> 452         return _adapters.Message(email.message_from_string(text))
    453 
    454     @property

/usr/lib/python3.12/importlib/metadata/_adapters.py in __new__(cls, orig)
     41     def __new__(cls, orig: email.message.Message):
     42         res = super().__new__(cls)
---> 43         vars(res).update(vars(orig))
     44         return res
     45 

KeyboardInterrupt: 

# saving full model

model.save("/content/drive/MyDrive/MobileNetV2_TeaDisease_Final.keras")


# loading saved model

from tensorflow.keras.models import load_model

model = load_model("/content/drive/MyDrive/MobileNetV2_TeaDisease_Final.keras")

# saving class nmes

import json

class_names = list(train_gen.class_indices.keys())

with open("/content/drive/MyDrive/class_names.json", "w") as f:
    json.dump(class_names, f)

# loading the saved class names

with open("/content/drive/MyDrive/class_names.json") as f:
    class_names = json.load(f)
     
After loading the saved model, testing with images

from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras.models import load_model
import json
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

Mounted at /content/drive

# Load trained model
model = load_model("/content/drive/MyDrive/MobileNetV2_TeaDisease_Final.keras")

# Load class names
with open("/content/drive/MyDrive/class_names.json") as f:
    class_names = json.load(f)

print("Model loaded successfully")
print("Classes:", class_names)

Model loaded successfully
Classes: ['AlgalLeaf', 'Anthracnose', 'BirdEyeSpot', 'BrownBlight', 'GrayBlight', 'Healthy', 'RedLeafSpot', 'WhiteSpot']

img_height = 224
img_width = 224

img_path = "/content/drive/MyDrive/CheckDisease/grayblight1.jpg"

img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)     

pred = model.predict(img_array)

pred_index = np.argmax(pred)
pred_class = class_names[pred_index]
confidence = pred[0][pred_index]

print("Predicted Disease:", pred_class)
print("Confidence:", f"{confidence*100:.2f}%")

     
1/1 ━━━━━━━━━━━━━━━━━━━━ 13s 13s/step
Predicted Disease: GrayBlight
Confidence: 53.98%
     
