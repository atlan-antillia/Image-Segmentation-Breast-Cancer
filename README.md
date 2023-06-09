# Image-Segmentation-Breast-Cancer (Updated: 2023/06/25)
<h2>
1 Image-Segmentation-Breast-Cancer
</h2>
This is an experimental project to detect <b>Breast Cancer</b> based on 
Breast Ultrasound Images Dataset (BUSI), by using our <a href="https://github.com/atlan-antillia/Tensorflow-Slightly-Flexible-UNet">Tensorflow-Slightly-Flexible-UNet Model.</a><br>
The original dataset used here has been taken from the following web site:<br>
<b>Breast Ultrasound Images Dataset</b><br>
Breast ultrasound images for classification, detection & segmentation<br>

<pre>
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
</pre>
Citation:<br>
<pre>
Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. 
Dataset of breast ultrasound images. Data in Brief. 
2020 Feb;28:104863. 
DOI: 10.1016/j.dib.2019.104863.
</pre>

<p>
See also:
</p>
<pre>
U-Net: Convolutional Networks for Biomedical Image Segmentation
 https://arxiv.org/pdf/1505.04597.pdf
</pre>
<ul>
<li>2023/05/24: Modified to write the merged (image+mask) inferred image files.</li>
<li>2023/06/16: Created ./projects/Breast-Cancer folder.</li>
<li>2023/06/16: Modified to use ImageMaskDataset instead of BreastCancerDataset.</li>
<li>2023/06/25 Updated TensorflowUNet.py to backup copies of previous eval_dir and model_dir.</li>
<li>2023/06/25 Modified TensorflowUNet.py to copy a configuration file to a model saving directory.</li>

</ul>
<br>
<h2>
2. Install Image-Segmentation-Breast-Cancer
</h2>
Please clone Image-Segmentation-Breast-Cancer.git in the working folder <b>c:\google</b>.<br>
<pre>
>git clone https://github.com/atlan-antillia/Image-Segmentation-Breast-Cancer.git<br>
</pre>
You can see the following folder structure in your working folder.<br>

</h2>
<pre>
Image-Segmentation-Breast-Cancer
├─asset
└─projects
    └─Breast-Cancer
        ├─Breast-Cancer
        │  ├─test
        │  │  └─malignant
        │  │      ├─images
        │  │      └─masks
        │  └─train
        │      └─malignant
        │          ├─images
        │          └─masks
        ├─eval
        ├─mini_test
        ├─mini_test_output
        ├─mini_test_output_merged
        ├─models
        └─preprocessors
            └─Dataset_BUSI_with_GT
</pre>
<br>
<h2>
3 Prepare dataset
</h2>

<h3>
3.1 Download master dataset
</h3>
 Please download the original dataset from the following link<br>
<pre>
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
</pre>
<b>Dataset_BUSI_with_GT</b> dataset has the following folder structure.<br>
<pre>
Dataset_BUSI_with_GT
├─benign
├─malignant
└─normal
</pre>
For example, <b>malignant</b> folder contains Breast Ultrasound image files and corressponding mask files
as show below.<br>
<img src="./asset/Dataset_BUSI_with_GT_malignant_samples.png" width="1024" height="auto"><br>

<h3>
3.2 Split master dataset
</h3>
For simplicity, we have resized, augmented and split the original <b>malignant</b> only in <b>Dataset_BUSI_with_GT</b>
 dataset to <b>train</b> and <b>test</b> dataset 
by using Python script <a href="./preprocessors/create_augmented_master_256x256.py">create_augmented_master_256x256.py</a>.
<pre>
BUSI_augmented_master_256x256
├─test
│  └─malignant
└─train
    └─malignant
</pre>
<h3>
3.3 Split master to images and masks
</h3>
Furthermore, we have split image files in each <b>malignant</b> folder to <b>images</b> and <b>masks</b> 
by using Python script <a href="./preprocessors/split_to_images_masks.py">split_to_images_masks.py</a>.
<pre>
Breast-Cancer
├─test
│  └─malignant
│      ├─images
│      └─masks
└─train
    └─malignant
        ├─images
        └─masks
</pre>
<b>Samples of augmented train/malignant/images</b><br>
<img src="./asset/Breast-Cancer_train_malignant_images.png" width="1024" height="auto"><br>
<br>
<b>Samples of augmented train/malignant/masks</b><br>
<img src="./asset/Breast-Cancer_train_malignant_masks.png"  width="1024" height="auto"><br>

<b>Breast-Cancer dataset inspection</b><br>
<img src="./asset/dataset_inspection.png" width="720" height="auto"><br>

<br>
<h2>
4 Train TensorflowUNet Model
</h2>
 We have trained Breast-Cancer TensorflowUNet Model by using the following
 <b>train_eval_infer.config</b> file. <br>
Please move <b>./projects/Breast-Cancer/</b>, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../TensorflowUNetTrainer.py train_eval_infer.config
</pre>

<pre>
; train_eval_infer.config
; 2023/6/25 antillia.com
; num_layers     = 7
; normalization  = True

; Modified to use loss and metric
; Specify loss as a function nams
; loss =  "bce_iou_loss"
; Specify metrics as a list of function name
; metrics = ["binary_accuracy"]
; Please see: https://www.tensorflow.org/api_docs/python/tf/keras/Model?version=stable#compile

[model]
image_width    = 256
image_height   = 256
image_channels = 3
num_classes    = 1
base_filters   = 16
base_kernels   = (3,3)
;2023/06/24
num_layers     = 7
normalization  = True

dilation       = (1,1)
dropout_rate   = 0.08
learning_rate  = 0.001
loss           = "bce_iou_loss"
metrics        = ["binary_accuracy"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 4
patience      = 10
metrics       = ["binary_accuracy", "val_binary_accuracy"]
model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "./Breast-Cancer/train/malignant/images/"
mask_datapath  = "./Breast-Cancer/train/malignant/masks/"
create_backup  = True

[eval]
image_datapath = "./Breast-Cancer/test/malignant/images/"
mask_datapath  = "./Breast-Cancer/test/malignant/masks/"

[infer] 
images_dir    = "./mini_test" 
output_dir    = "./mini_test_output"
merged_dir    = "./mini_test_output_merged"

[mask]
blur      = True
binarize  = True
threshold = 100
</pre>

Since <pre>loss = "binary_crossentropy"</pre> and <pre>metrics = ["binary_accuracy"] </pre> are specified 
in <b>train_eval_infer.config</b> file,
<b>binary_crossentropy</b> and <b>binary_accuracy</b> functions are used to compile our model as shown below.
<pre>
    # Read a loss function name from a config file, and eval it.
    # loss = "binary_crossentropy"
    self.loss  = eval(self.config.get(MODEL, "loss"))

    # Read a list of metrics function names from a config file, and eval each of the list,
    # metrics = ["binary_accuracy"]
    metrics  = self.config.get(MODEL, "metrics")
    self.metrics = []
    for metric in metrics:
      self.metrics.append(eval(metric))
        
    self.model.compile(optimizer = self.optimizer, loss= self.loss, metrics = self.metrics)
</pre>
You can also specify other loss and metrics functions in the config file.<br>
Example: basnet_hybrid_loss(https://arxiv.org/pdf/2101.04704.pdf)<br>
<pre>
loss         = "basnet_hybrid_loss"
metrics      = ["dice_coef", "sensitivity", "specificity"]
</pre>
On detail of these functions, please refer to <a href="./losses.py">losses.py</a><br>, and 
<a href="https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/tree/master">Semantic-Segmentation-Loss-Functions (SemSegLoss)</a>.

We have also used Python <a href="./BreastCancerDataset.py">BreastCancerDataset.py</a> script to create
train and test dataset from the original and segmented images specified by
<b>image_datapath</b> and <b>mask_datapath </b> parameters in the configratration file.<br>
The training process has just been stopped at epoch 33 by an early-stopping callback as shown below.<br><br>
<img src="./asset/train_console_at_epoch_33_0624.png" width="720" height="auto"><br>
<br>
The <b>val_binary_accuracy</b> is very high as shown below from the beginning of the training.<br>
<b>Train metrics line graph</b>:<br>
<img src="./asset/train_metrics_33.png" width="720" height="auto"><br>

<br>
The val_loss is also very low as shown below from the beginning of the training.<br>
<b>Train losses line graph</b>:<br>
<img src="./asset/train_losses_33.png" width="720" height="auto"><br>


<h2>
5 Evaluation
</h2>
 We have evaluated prediction accuracy of our Pretrained Breast-Cancer Model by using <b>test</b> dataset.
Please move <b>./projects/Breast-Cancer/</b>, and run the following bat file.<br>
<pre>
>2.evalute.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python .,/../TensorflowUNetEvaluator.py train_eval_infer.config
</pre>
The evaluation result of this time is the following.<br>
<img src="./asset/evaluate_console_at_epoch_33_0624.png" width="720" height="auto"><br>
<br>

<h2>
6 Inference 
</h2>
We have also tried to infer the segmented region for <b>mini_test</b> dataset, which is a very small dataset including only ten images extracted from <b>test</b> dataset,
 by using our Pretrained Breast-Cancer Model.<br>
<pre>
Please move <b>./projects/Breast-Cancer/</b>, and the following bat file.<br>
>3.infer.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../TensorflowUNetInferencer.py train_eval_infer.config
</pre>

<b>Input images (mini_test) </b><br>
<img src="./asset/mini_test.png" width="1024" height="auto"><br>
<br>
<b>Inferred images (mini_test_output)</b><br>
<img src="./asset/mini_test_output.png" width="1024" height="auto"><br><br>

<br>
<b>Merged inferred images</b><br>
<img src="./asset/mini_test_output_merged.png" width="1024" height="auto"><br><br>

<br>

<!--
-->

<h3>
References
</h3>
<b>1. Breast Ultrasound Images Dataset</b><br>
<pre>
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
</pre>

<b>2. Breast lesion detection using an anchor-free network from ultrasound images with segmentation-based enhancement</b><br>
Yu Wang & Yudong Yao<br>
<pre>
https://www.nature.com/articles/s41598-022-18747-y
</pre>

<b>3. Classification of Breast Cancer Ultrasound Images with Deep Learning-Based Models </b><br>
Fatih Uysa,and Mehmet Murat Köse<br>
<pre>
https://www.mdpi.com/2673-4591/31/1/8/html
</pre>

<b>4. A CNN Deep Learning Technique for Prediction of Breast Cancer using Ultrasound Image
</b><br>
Atisham Khan and Silky Pareyani<br>
<pre>
https://www.jetir.org/papers/JETIR2303813.pdf
</pre>

<b>5. Discrimination of Breast Cancer Based on Ultrasound Images and Convolutional Neural Network
</b><br>
Rui Du,Yanwei Chen,Tao Li, Liang Shi,Zhengdong Fei,and Yuefeng Li
<br>
<pre>
https://www.hindawi.com/journals/jo/2022/7733583/
</pre>

<b>6. EfficientDet-Augmented-Breast-Ultrasound-Images
</b><br>
Toshiyuki Arai @antillia.com
<br>
<pre>
https://github.com/sarah-antillia/EfficientDet-Augmented-Breast-Ultrasound-Images
</pre>

<b>7. Semantic-Segmentation-Loss-Functions (SemSegLoss)</b><br>
<pre>
https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
</pre>
<pre>
Citation
@inproceedings{jadon2020survey,
  title={A survey of loss functions for semantic segmentation},
  author={Jadon, Shruti},
  booktitle={2020 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB)},
  pages={1--7},
  year={2020},
  organization={IEEE}
}
@article{JADON2021100078,
title = {SemSegLoss: A python package of loss functions for semantic segmentation},
journal = {Software Impacts},
volume = {9},
pages = {100078},
year = {2021},
issn = {2665-9638},
doi = {https://doi.org/10.1016/j.simpa.2021.100078},
url = {https://www.sciencedirect.com/science/article/pii/S2665963821000269},
author = {Shruti Jadon},
keywords = {Deep Learning, Image segmentation, Medical imaging, Loss functions},
abstract = {Image Segmentation has been an active field of research as it has a wide range of applications, 
ranging from automated disease detection to self-driving cars. In recent years, various research papers 
proposed different loss functions used in case of biased data, sparse segmentation, and unbalanced dataset. 
In this paper, we introduce SemSegLoss, a python package consisting of some of the well-known loss functions 
widely used for image segmentation. It is developed with the intent to help researchers in the development 
of novel loss functions and perform an extensive set of experiments on model architectures for various 
applications. The ease-of-use and flexibility of the presented package have allowed reducing the development 
time and increased evaluation strategies of machine learning models for semantic segmentation. Furthermore, 
different applications that use image segmentation can use SemSegLoss because of the generality of its 
functions. This wide range of applications will lead to the development and growth of AI across all industries.
}
}
</pre>
<br>

