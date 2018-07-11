## Vehicle Detection  - Term 1  - P5
### In this project I attempted to identify and track the vehicles/cars that are visible from the dashboard camera, mark it using rectangle using OpenCV Image detection features.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/example_data_set.jpg
[image2]: ./examples/bin_spatial.jpg
[image3]: ./examples/ycrb_hog.png
[image4]: ./examples/color_histogram.jpg
[image5]: ./examples/HOG_image.jpg
[image6]: ./examples/slide_window.jpg
[image7]: ./examples/search_window.jpg
[image8]: ./examples/sub_sampling_window.jpg
[image9]: ./examples/template_matching.jpg
[image10]: ./examples/heatmap_false_positive.jpg
[image11]: ./examples/final_sample.jpg
[image12]: ./examples/video.jpg
[image13]: ./examples/GridSearchCV.png
[image14]: ./examples/heatmap_order.png
[video1]: ./project_video.mp4

### Final Video Posted here

[you tube video] (https://youtu.be/L5_a7KUzvbI)

[Youtube Link](http://img.youtube.com/vi/L5_a7KUzvbI/maxresdefault.jpg)


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook [Vehicle_Detection_V3.ipynb](./Vehicle_Detection_V3.ipynb) 

I used the function which is present here :
``` skimage.feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm=None, visualize=False, visualise=None, transform_sqrt=False, feature_vector=True, multichannel=None)```

The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection.

This is how I have done this.

1. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![DataSet Exploration][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![RGB HOG Image][image5]


Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

More references here
1.[scikit hog feature](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog)
2.[wikipedia page](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)

![YCRB HOG][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

|Parameter name        | Description                                  |
| -------------------- | ---------------------------------------------|
| `Color Space`        | Color space RGB, LUV, HLS, HSV, YcCrCb       |
| `Orient`             | Orientation usually '9' works best           |
| `pixel per cell`     | Pixels for Cell '8'                          |
| `cell per block`     | Cells per Block '2'                          |
| `hog_channel`        | Hog Channel, 0, 1, 2 or 'ALL'                |
| `spatial_size`       | Spatial size (16,16) or (32,32)              |
| `hist_bins`          | Hist bins either 16, or 32                   |
| `spatial feat`       | True or False                                |
| `hist feat`          | True or False                                |
| `hog feat`           | True or False                                |
| `block norm`         | L1-Hys is default                            |


We can experiment with various different Color spaces, Orientation, Pixel per Cell and Cells per Block, I settled for almost the default values as discussed in the class notes.

There is huge difference though if the values for Color Space and Hog Channel changes, this will directly impact the outcome of the SVC classifier so we need to make sure we use the optimal sizes/parameters, which can be done only via trial & error method.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained the Classifier in following way.

1. Extract the  Color Histogram
```python
def color_hist(img, nbins=32):
    hist1 = np.histogram(img[:,:,0], bins=nbins)
    hist2 = np.histogram(img[:,:,1], bins=nbins)
    hist3 = np.histogram(img[:,:,2], bins=nbins)
    return hist_features
 ```
 [color histogram][image4]
 

2. Spatial binning
```python
def bin_spatial(img, size=(32,32)):```

[spatial binning][image2]


3. Extract the Color Features for car and non car objects.
```python
def extract_features(test_cars, color_space = color_space ...

def extract_features(nontest_cars ...
```

4. Extract Scalar from the car and non features
```python
X = np.vstack((car_features, notcar_features)).astype(np.float64)
#Fit per X column
X_scaler = StandardScaler().fit(X)

#Apply the scaler to X
scaled_X = X_scaler.transform(X)
```

5. Label them as 1 for cars and 0 for non cars.
```python

y = np.hstack ((np.ones(len(car_features)), np.zeros(len(notcar_features))))
```

6. split the Data as Training test and Test set with random shuffling.

```python
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state = rand_state)
```

7. Classification using Support Vector Machines(SVC)

We can classify the data using any of the below methods.
1. [LinearSVC ](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
       
   Faster compared the other two discussed here, I am able to reach the accuracy of 98.9% and trained 25 seconds.
   One of the advantage of this method is ability to train faster but can show many false positives in the images.

2. [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

    Much slower compared to all the three takes about 45+ minutes to do exhaustive search over specified parameter values for an estimator.
    Able to reach accuracy close to 99.49%
    [GridSearchCV Output][image13]
    
3. [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
    Much slower but better than GridSearch CV, I ran into few issues choosing this, so did not proceed further.


Finally after analyzing the pros and cons I settled for LinearSVC which saves tons of time while processing the image as well.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Initially started with Sliding windows search where in I searched the entire image but this could consume lot of CPU Power and we would be unnecessarily searching /looking for cars where they are not supposed to be !!

![Sliding Windows][image7]


Later I reduced the search space to 400 to 656 in y-axis with different scales such as 0.7 to 1.5 to find out the suitable range to reduce the false positives.

![Sub Sampling Window][image14]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried many ways to arrive at the best detection methods 

1. Altering the Y axis with different values.

2. Modifying the scale to 0.7 upto 3.0

3. Trying different color spaces RGB, LUV, HSV, YcrCb ..

4. Trying different classifier such as SVC, RandomizedCV, GridSearchCV etc.

![Sub Sampling Window][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_result_final_v3.mp4)

[you tube video](https://youtu.be/L5_a7KUzvbI)
[Youtube Link](http://img.youtube.com/vi/L5_a7KUzvbI/maxresdefault.jpg)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:
![heatmap order][image14]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![Heatmap False Positive][image10]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Initially I started experimenting with Template Matching, as you can see here is the result, 
![Template Matching][image9]


I was under the impression I could possibly remove the false positives after I get the bounding box from the find_cars method, but I was not very successful in that method, this is something worth the future effort.


I experiemented with the GridSearchCV Classifier as explained before, this appears to be much better than Classic LinearSVC but here the amount of time consumed is astronomically high and each image with the GridSearchCV classifer would take almost 10-12 seconds to process which would nearly 12-24 hours for a video to generate. I abandoned this idea to sheer amount of time it took, otherwise it had very less false positives.


Tuning the parameters took most of the time, finding the right threshold, finding the false positives took more time than I anticipated.

Another aspect which I wanted to try is to use Real-time Object Detection with YOLO (You Look Only Once) YOLO v3.
I am going to try out few experiements around it in the coming days : https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088


**Where the pipeline likely to fail ? **

1. Poor light conditions. Rainy or extreme weather situations.

2. Too many cars or in a crowded objects could be confusing 

3. Trucks and other vehicles are not classified.

4. Different Road Surfaces.

5. Steep Hill or Down hill areas.
