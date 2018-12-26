#!/usr/bin/env python
# encoding: utf-8


import os
import cv2
import glob
import h5py
import tensorflow as tf
import numpy as np
import nets.resnet_v1_50 as model
import heads.fc1024 as head
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as sknormalize


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


def normalize(x, copy=False):
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1, -1), copy=copy))
        #return np.squeeze(x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis])
    else:
        return sknormalize(x, copy=copy)
        #return x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis]


def save_results_image(rank_dists, rank_names, dir_images, save_directory):
    """
    save result images
    """
    for j, tmp_dist in enumerate(rank_dists):
        if 0.9999 > tmp_dist > 0.9:
            if not os.path.exists(directory):
                os.makedirs(save_directory)
            copyfile(os.path.join(dir_images, rank_names[j]),
                     os.path.join(directory, str(j) + '_a_' + rank_names[j]))
        elif 0.9 >= tmp_dist > 0.8:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            copyfile(os.path.join(dir_images, rank_names[j]),
                     os.path.join(save_directory, str(j) + '_b_' + rank_names[j]))

        if os.path.exists(save_directory):
            copyfile(os.path.join(dir_images, rank_names[0]), os.path.join(directory, rank_names[0]))


def query_images(img_path, cropped = True):
    img = cv2.imread(img_path)[:,:,::-1]
    feat = sim_vector(img)
    # L2-normalize feature
    feat = normalize(feat, copy=False)
    return feat


def compute_cosin_distance(Q, feats, names):
    """
    feats and Q: L2-normalize, n*d
    """
    dists = np.dot(Q, feats.T).flatten()
    idxs = np.argsort(dists)[::-1].flatten()
    rank_dists = dists[idxs]
    rank_names = [names[k] for k in idxs]
    return (idxs, rank_dists, rank_names)

def compute_euclidean_distance(Q, feats, names, k = None):
    if k is None:
        k = len(feats)

    dists = ((Q - feats)**2).sum(axis=1)
    idx = np.argsort(dists) 
    dists = dists[idx]
    rank_names = [names[k] for k in idx]

    return (idx[:k], dists[:k], rank_names)
    

def simple_query_expansion(Q, data, inds, top_k=10):
    Q += data[inds[:top_k], :].sum(axis=0)
    return normalize(Q)


def load_files(files):
    """
    Function: load features from files
    files: list
    """
    h5fs = {}
    for i, f in enumerate(files):
        h5fs['h5f_' + str(i)] = h5py.File(f, 'r')
    feats = np.concatenate([value['feats'] for key, value in h5fs.items()])
    names = np.concatenate([value['names'] for key, value in h5fs.items()])
    return (feats, names)

if __name__ == '__main__':
    
    img_path = '36.jpg'
    feats_files = ['landmarks_feats.h5']
    save_directory = 'temp'
 
    # query expansion
    do_QE = False
    topK = 6
    do_crop = False
    do_pca = False
    redud_d = 128

    # load all features
    feats, names = load_files(feats_files)

    # L2-normalize features
    feats = normalize(feats, copy=False)


    # PCA reduce dimension
    if do_pca:
        whitening_params = {}
        if os.path.isfile('pca_model.pkl'):
            with open( 'pca_model.pkl' , 'rb') as f:
                whitening_params['pca'] = pickle.load(f)
        _, whitening_params = run_feature_processing_pipeline(feats, d = redud_d, copy = True)
        with open('pca_model.pkl', 'wb') as f:
            pickle.dump(whitening_params['pca'], f, pickle.HIGHEST_PROTOCOL)
        feats, _ = run_feature_processing_pipeline(feats, params=whitening_params)
        print "pca finished ......"

    query_feat = query_images(img_path, cropped = False)
    Q = query_feat

    if do_pca:
        Q, _ = run_feature_processing_pipeline([Q], params=whitening_params)
        Q = np.squeeze(Q.astype(np.float32))

    idxs, rank_dists, rank_names = compute_cosin_distance(Q, feats, names)
    #idxs, rank_dists, rank_names = compute_euclidean_distance(Q, feats, names)

    if do_QE:
        Q = simple_query_expansion(Q, feats, idxs, top_k=topK)
        idxs, rank_dists, rank_names = compute_cosin_distance(Q, feats, names)
        #idxs, rank_dists, rank_names = compute_euclidean_distance(Q, feats, names)


    mw = 135 # 图片大小+图片间隔
    ms = 7 
    msize = mw * ms
    toImage = Image.new('RGBA', (msize, msize))
    count = 0
    for y in range(1, 8):
        for x in range(1, 8):
            path = rank_names[count]
            fromImage = Image.open(path)
            print path
            fromImage = fromImage.resize((mw, mw), Image.ANTIALIAS)
            toImage.paste(fromImage, ((x-1) * mw, (y-1) * mw))
            count += 1
    toImage.save('out.png')