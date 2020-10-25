# Documentation

## Image Plagarism
Plagiarism of images is really no different than plagiarism of words, music, or any original work.  Copying an image from a book or the internet without citing the original source (or gaining permission of the creator when necessary) constitutes plagiarism.  Plagiarism is academic fraud with serious repercussions.So our attempt is to make a Image based plagarism checker for checking the plagarism between two given images.
## Library used
## opencv contrib 
<br/> OpenCV (open source computer vision) is a very powerful library for image processing and machine learning tasks which also supports Tensorflow, 
Torch/Pytorch and Caffe. The library is cross platform and you can pip install it (where you are using it with Python) with CPU support. <br />
To install opencv contirb use: <br/>
  ```pip install opencv-contrib-python```
<br /> for more details about opencv refer the [documentation]("https://pypi.org/project/opencv-contrib-python/").

We use <a href="ttps://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html"> ORB</a> and <a href="https://docs.opencv.org/3.4/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html">FREAK</a> algorithms for identifying the features of the given images.<br/> 
We used <a href="https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html">Brute Force Knn Matcher </a> for matching the features of images to compare similarity.
Then using the distance ratio we calcuate the good matches from the matched descriptors.
