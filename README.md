# GC-Net-Tensorflow
**End-to-End Learning of Geometry and Context for Deep Stereo Regression**

It is a simple Tensorflow implementation of the paper [https://arxiv.org/pdf/1703.04309.pdf](https://arxiv.org/pdf/1703.04309.pdf).

Test on images from Middlebury Stereo Dataset

![cones](https://github.com/kelkelcheng/GC-Net-Tensorflow/blob/master/middlebury/cones/test_disparity.jpg)

# Train
To train this model from scratch, you will need to download the data from 
[FlyingThings3D (cleanpass images 37GB)](https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass.tar)
and [FlyingThings3D (disparity 87GB)](https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2)

Then run **FlyingThings_TFRecord.py** to generate TFRecord format dataloader.

The directory is assumed to be:

    FlyingThings_TFRecord.py

    flyingthings3d_frames_cleanpass

      TEST
    
      TRAIN

    flyingthings3d__disparity

      disparity
    
        TEST

        TRAIN

After you get **fly_train.tfrecords** and **fly_test.tfrecords**, you can run **train.py** to train.
The temporary model files will be saved in directory **saved_model**.

# Pre-trained model
A pre-trained model can be downloaded [here](https://drive.google.com/open?id=1N64rp2sJieJJH-EoK59SyUxGK39HTmxK)

To load pre-trained model (trained after 60k steps), create directory **saved_model** and put all the downloaded files inside:
    
    -60000.data-00000-of-00001
    -60000.index
    -60000.meta
    checkpoint

# Test
Run **test.py** to test for new images. The default test images are from [Middlebury Stereo Dataset](http://vision.middlebury.edu/stereo/). 
You can change the file name and directory to test for your own data.

Sample outputs are also provided in the middlebury folder

# Comments
The training converges pretty fast. The training error, testing error, and training time are close to the paper.

However, you might need TitanX or 1080 Ti, otherwise the memory might not be enough. 

The code was written about a year ago so I used Tensorflow 1.3.0 and Python 3.5.

# To do next...
I forgot to give names to the placeholders and output of the graph, so test.py is quite cumbersome.

I will write a function to load the graph from meta file directly later.

# Reference
Kendall, Alex, et al. "End-to-End Learning of Geometry and Context for Deep Stereo Regression." arXiv preprint arXiv:1703.04309 (2017).
