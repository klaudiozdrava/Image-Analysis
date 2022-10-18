# Image-Analysis
A Python project that was developed as a university assignment for the subject of Image Processing. 
The program takes an input image and a reference dataset of photos. The goal is to colorize the greyscale image using a trained support vector machine.
To achieve that, we have implemented a variety of image processing techniques.
First, we change color spaces from RGB to LAB. Then, we apply the SLIC algorithm to find the group of superpixels for each image.
These segments along with SIFT and GABOR features are given as input for the SVM.
Using scikit-learn, we use machine learning techniques to predict the color of a superpixel using the dataset superpixels as reference. 
The output of the program returns the colorized version of the input image.
To run the algorithm user should provide at runtime the absolute path of the folder that contains training images and the path of testing image.

![BW](https://user-images.githubusercontent.com/47723760/196312855-68ae244a-2dbe-4551-83f2-22b7c556681d.jpg)
![Figure 1 10_18_2022 2_40_49 AM (2)](https://user-images.githubusercontent.com/47723760/196312877-72ac950e-b125-423d-8138-c04e93203fb6.png)
