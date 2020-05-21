import os
import argparse
from model import crnn
import tensorflow as tf
from tensorflow import keras

from dataset import Decoder


def read_image(path):
    img = tf.io.read_file(path)
    try:
        img = tf.io.decode_jpeg(img, channels=3)
    except Exception:
        print("Invalid image: {}".format(path))
        global num_invalid
        return tf.zeros((32, args.image_width, 1))
    img = tf.image.convert_image_dtype(img, tf.float32)
    if args.keep_ratio:
        width = round(32 * img.shape[1] / img.shape[0])
    else: 
        width = args.image_width
    img = tf.image.resize(img, (32, width))
    return img


parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--images", type=str, default=r'E:\tsl_file\python_project\CRNN.tf2\example\images',
#                     help="Images file path.")
# parser.add_argument("-t", "--table_path", type=str, required=True, default=r'E:\tsl_file\python_project\CRNN.tf2\example\table.txt',
#                     help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, 
                    help="Image width(>=16).")
parser.add_argument("-k", "--keep_ratio", action="store_true",
                    help="Whether keep the ratio.")
# parser.add_argument("-m", "--model", type=str, required=True,
#                     help="The saved model path.")
args = parser.parse_args()

args_images=r'E:\tsl_file\python_project\CRNN.tf2\example\images'
args_table_path=r'E:\tsl_file\python_project\CRNN.tf2\example\table.txt'

if args_images is not None:
    if os.path.isdir(args_images):
        imgs_path = os.listdir(args_images)
        img_paths = [os.path.join(args_images, img_path)
                        for img_path in imgs_path]
        imgs = list(map(read_image, img_paths))
        imgs = tf.stack(imgs)
    else:
        img_paths = [args_images]
        img = read_image(args_images)
        imgs = tf.expand_dims(img, 0)
with open(args_table_path, "r") as f:
    inv_table = [char.strip() for char in f]

# model = keras.models.load_model(args.model, compile=False)
model=crnn(63)
decoder = Decoder(inv_table)


checkpoint_dir = './checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint( model=model)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir , checkpoint_name='ckpt', max_to_keep=5)

status=checkpoint.restore(manager.latest_checkpoint)
print('status',status)


# y_pred = model.predict(imgs)
y_pred = model(imgs)
print('y_pred shape',y_pred.shape)



for path, g_pred, b_pred in zip(img_paths, 
                                decoder.decode(y_pred, method='greedy'),
                                decoder.decode(y_pred, method='beam_search')):
    print("Path: {}, greedy: {}, beam search: {}".format(path, g_pred, b_pred))