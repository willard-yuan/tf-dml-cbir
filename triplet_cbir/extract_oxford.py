
import os
import cv2
import glob
import h5py
import tensorflow as tf
import numpy as np
import nets.resnet_v1_50 as model
import heads.fc1024 as head

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Tensorflow descriptor model
tf.Graph().as_default()
sess = tf.Session()
images = tf.zeros([1, 256, 256, 3], dtype=tf.float32)
endpoints, body_prefix = model.endpoints(images, is_training=False)
with tf.name_scope('head'):
    endpoints = head.head(endpoints, 128, is_training=False)
tf.train.Saver().restore(sess, 'exps/checkpoint-20000')


def sim_vector(img):
    resize_img = cv2.resize(img, (256, 256))
    resize_img = np.expand_dims(resize_img, axis=0)
    emb = sess.run(endpoints['emb'], feed_dict={images: resize_img})
    return emb


if __name__ == '__main__':

    dir_images = './oxford/*.jpg'
    imgs_path = glob.glob(dir_images)

    h5f = h5py.File('oxford_feats.h5', 'w')

    features = []
    image_names = []
    for i, path in enumerate(imgs_path):
        img = cv2.imread(path)[:,:,::-1]
        feat = sim_vector(img)
        features.append(np.array(feat.flatten()))
        image_names.append(os.path.basename(path))
        print "%d (%d)" %((i+1), len(imgs_path))
    features = np.array(features)
    print features.shape
    h5f['feats'] = features
    h5f['names'] = image_names
    h5f.close()