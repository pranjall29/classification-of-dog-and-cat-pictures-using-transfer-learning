import tensorflow as tf
import tarfile
import urllib
import os
import numpy as np
import pets
from matplotlib import pyplot as plt
%matplotlib inline
data_dir = 'data/'
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
print('TensorFlow version:', tf.__version__)
print('Is using GPU?', tf.test.is_gpu_available())

pets.download_and_extract('data', '.')

class_to_index = {'cat': 0, 'dog': 1}
index_to_class = {0: 'cat', 1: 'dog'}
train_annot, count_train = pets.cats_vs_dogs_annotations('data/annotations/trainval.txt')
test_annot, count_test = pets.cats_vs_dogs_annotations('data/annotations/test.txt')
print('Training examples count:', count_train)
print('Test examples count:', count_test)

image_dir = r"C:\Users\admin\Desktop\coursera projects\transfer learning witth keras\data\images"
def get_random_batch(annot, batch_size=4):
    all_keys = list(annot.keys())
    total_examples = len(all_keys)
    indices = np.random.choice(range(total_examples), batch_size)
    x = np.zeros((batch_size, 128, 128, 3))
    y = np.zeros((batch_size, 1))
    images = []
    for i, index in enumerate(indices):
        image = tf.keras.preprocessing.image.load_img(os.path.join(image_dir, all_keys[index]),target_size=(128, 128))
        images.append(image)
        arr = tf.keras.preprocessing.image.img_to_array(image)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        x[i] = arr
        y[i] = class_to_index[annot[all_keys[index]]]
    return x, y, images

x, y, images = get_random_batch(train_annot, batch_size=8)
pets.display_examples(x, y, y, images, index_to_class).show()


mnet = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, input_shape=(128, 128, 3),
pooling='avg', weights='imagenet')
mnet.summary()


def create_model():
    model = tf.keras.models.Sequential([
            mnet,
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
            ])
    model.layers[0].trainable = False
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model
model = create_model()
model.summary()


def data_generator(batch_size, annot):
    while True:
        x, y, _ = get_random_batch(annot, batch_size)
        yield (x, y)


batch_size = 32
steps_per_epoch = int(len(list(train_annot.keys()))/batch_size)
validation_steps = int(len(list(test_annot.keys()))/batch_size)
print('Steps per epoch:', steps_per_epoch)
print('Validation steps:', validation_steps)

_ = model.fit_generator(
data_generator(batch_size, train_annot),
validation_data=data_generator(batch_size, test_annot),
steps_per_epoch=steps_per_epoch,
validation_steps=validation_steps,
epochs=1
)


x, y, images = get_random_batch(test_annot, batch_size=8)
preds = model.predict(x)
pets.display_examples(x, y, preds, images, index_to_class).show()
