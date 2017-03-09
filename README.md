# Vehicle Detection Project

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector
* Train a classifier Linear SVM classifier 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images
* Create a heat map of window detections
* Estimate a bounding box for vehicles detected
* Run the pipeline on a video stream and output a video with tracked vehicles

[//]: # (Image References)
[image0a]: ./readme_images/03_car.png
[image0b]: ./readme_images/04_notcar.png
[image1]: ./readme_images/01_hog.png
[image2]: ./output_images/02_Test_image.jpg
[image3]: ./output_images/03_Test_windows.jpg
[image4]: ./output_images/04_All_windows.jpg
[image5]: ./output_images/05_Car_windows_all.jpg
[image6]: ./output_images/06_Heatmap.jpg
[image7]: ./output_images/07_Car_windows.jpg
[image81]: ./readme_images/02_test_images1.png
[image82]: ./readme_images/02_test_images2.png
[image83]: ./readme_images/02_test_images3.png
[image84]: ./readme_images/02_test_images4.png
[image85]: ./readme_images/02_test_images5.png
[image86]: ./readme_images/02_test_images6.png
[image87]: ./readme_images/02_test_images7.png
[image88]: ./readme_images/02_test_images8.png

###Histogram of Oriented Gradients (HOG)

The code for this step is contained in the 3rd code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image0a]
![alt text][image0b]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed first car image and displayed it to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

I tried various combinations of parameters and I have decide to make a fair trade-off between car feature recognition and speed of processing it. The problem, which I was was facing was to find best color space for both black and white car. I've decided to use those parameters:

* first channel 'YUV' color space
* orientations equal 9
* pix_per_cell equal 8
* cell_per_block equal 2

####Classifier training

Firstly I had iterate over whole dataset in order to get HOG features from all images. After doing so I have prepared labels for both vehicles and not vehicles images. The dataset and labels were splitted into train and test sets with propotion 8:2 as usually.

LinearSVC classifier was trained with usage of Pipeline and StandardScaler objects. Overall time of training was less than 10s and accuracy was equal 98%.

Training code is contained in 5th cell of the notebook.

###Sliding Window Search

In order to implement Window Search, I have implemented function for preparing all windows awailable for the default image size. Windows were created only for the bottom half of the images (without sky), with 50% edge overlapping and with different sizes depending on the vertical position - the more down it goes, the bigger the windows were.

![alt text][image4]
![alt text][image5]

I was checking each window and classifying it in order to know is there a car or not. Because of overlapping character of the windows I was able to generate heatmaps and find spots with high classifications of the cars.

![alt text][image6]

Code for this part is contained in cells 7 and 8 of the notebook.

#### Pipeline optimization

Ultimately I've decided to use only first channel of the 'YUV' color space, plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

Below you can see how the pipeline performs with test images:

![alt text][image81]
![alt text][image82]
![alt text][image83]
![alt text][image84]
![alt text][image85]
![alt text][image86]
![alt text][image87]
![alt text][image88]

---

### Video Implementation

####1. Lint to final video

Here's a [link to my video result](./project_video_output.mp4)

####2. False positives filter

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions and remove false positives. In order boost correct classification of the cars I have implemented heatmap list for keeping last 20 in memory and use sum of them in order to determine final heatmap. This solution gave me better stability of the vehicle tracking.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

---

###Discussion

I have decided to implement the solution as it was suggested in the course - using Linear SVM for car classification. It turned out that training was really fast, but in the end classifications were not always true. I think that using Neural Networks could give better results. Extracting features and training Linear SVM was also problematic because of big memory consumption.