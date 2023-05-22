# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# BrainTumorDataset.py
# 2023/05/10 to-arai

import os
import sys

import numpy as np

from tqdm import tqdm
import glob
# pip install scikit-image
from skimage.transform import resize
#from skimage.morphology import label
from skimage.io import imread, imshow
from matplotlib import pyplot as plt
import traceback

class BreastCancerDataset:

  def __init__(self, resized_image):
    self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS = resized_image


  def create(self, image_datapath, mask_datapath):
    image_files = sorted(glob.glob(image_datapath  + "/*.jpg"))
    mask_files  = sorted(glob.glob(mask_datapath + "/*.jpg"))
    
    if len(image_files) != len(mask_files):
      raise Exception("The number of the original and segmented image files is not matched")
    num_images  = len(image_files)
    X = []
    Y = []

    X = np.zeros((num_images, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
    Y = np.zeros((num_images, self.IMG_HEIGHT, self.IMG_WIDTH, 1                ), dtype=np.bool)

    for n, image_file in tqdm(enumerate(image_files), total=len(image_files)):
      #print("=== image_file {}".format(image_file))
      basename  = os.path.basename(image_file)
      name      = basename.split(".")[0]
      mask_file = mask_datapath + "/" + name + "_mask.jpg"
      #print("--- mask_file  {}".format(mask_file))
      image = imread(image_file)
      image = resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), mode='constant', preserve_range=True)
      X[n]  = image

      mask  = imread(mask_file) 
      mask  = resize(mask, (self.IMG_HEIGHT, self.IMG_WIDTH,                  1), mode='constant', preserve_range=True)       
      Y[n]  = mask

    return X, Y

    
if __name__ == "__main__":
  try:
    resized_image = (256, 256, 3)
    dataset = BreastCancerDataset(resized_image)

    # train dataset
    image_datapath = "./Breast-Cancer/train/malignant/images/"
    mask_datapath  = "./Breast-Cancer/train/malignant/masks/"


    x_train, y_train = dataset.create(image_datapath, mask_datapath)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # test dataset
    image_datapath = "./Breast-Cancer/test/malignant/images/"
    mask_datapath  = "./Breast-Cancer/test/malignant/masks/"

    x_test, y_test = dataset.create(image_datapath, mask_datapath)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

  except:
    traceback.print_exc()

