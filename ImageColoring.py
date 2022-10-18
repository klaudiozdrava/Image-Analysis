from matplotlib import pyplot as plt
import cv2
import numpy as np
from skimage import color
from skimage.segmentation import slic
from sklearn.cluster import KMeans
import skimage
import pickle
from sklearn.svm import SVC
from scipy.spatial import Voronoi, voronoi_plot_2d
import os

ksize = 25 #kernel size for gabor
sigma = 5
gamma = 0.6
phi = 0
DESIRABLE_SIZE=800 #Before 600
CLUSTERS=20#k-means clusters
SUPERPIXELS=90 #Before 80


def RGBtoLAB(image,Color_Vectors):
    # A_and_B_Colors=[]
    #Extract all RGB colors of all pixels and then convert it in Lab color space
    img=skimage.io.imread(fname=image)
    img=cv2.resize(img,(DESIRABLE_SIZE,DESIRABLE_SIZE))
    lab = color.rgb2lab(img) #convert RGB to Lab
    L, a, b = cv2.split(lab) #get only a and b values
    a=a.ravel() #flatten the array
    b=b.ravel()
    a_b=np.column_stack((a,b))
    print(f"The total colors before removing duplicates is : {len(a_b)}")
    a_b=np.unique(a_b, axis=0).tolist()#get unique combinations of a and b
    print(f"The total colors after removing duplicates is : {len(a_b)}")
    Color_Vectors+=a_b

    return Color_Vectors


def Kmeans(Cluster_numbers,Color_Space):

    font1 = {'family': 'serif', 'color': 'blue', 'size': 15}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}

    # Color_Space=RGBtoLAB()#Get all combinations of a and b from all images
    plt.scatter(*zip(*Color_Space))
    kmeans = KMeans(n_clusters=Cluster_numbers,init='k-means++', random_state=0).fit(Color_Space)
    centroids=kmeans.cluster_centers_#centroids of k-means
    plt.scatter(*zip(*centroids), marker="x", s=150, linewidths=5, zorder=10, c='r')
    plt.title(f"Color space with {Cluster_numbers} clusters (CIE(L)AB)", fontdict=font1)
    plt.xlabel("a vector", fontdict=font2)
    plt.ylabel("b vector", fontdict=font2)
    vor = Voronoi(centroids)
    voronoi_plot_2d(vor, show_vertices=False, line_colors='red',
                    line_width=2, line_alpha=0.6, point_size=8)#display area of kmeans clusters
    plt.show()  # Display colors in a xy-coordinate
    # pickle.dump(kmeans, open("Kmeans"+ IMAGE + str(Cluster_numbers) + ".pkl", "wb"))
    return kmeans


def findClusterOfSuperpixel(abmean,kmeans):
    #Predict cluster that a and b mean color belong
    ClusterPred = kmeans.predict(abmean)
    return ClusterPred


def extractFeatures(image,Clusters):
    SIFT_DATA={}
    ab_colors=[]
    features=[]
    img = skimage.io.imread(fname=image)
    img = cv2.resize(img, (DESIRABLE_SIZE, DESIRABLE_SIZE))

    gray_scale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#convert image to grayscale for SIFT and gabor

    # Extract superpixels using slic algorithm
    segments = slic(img, n_segments=SUPERPIXELS, compactness=1.0, max_num_iter=70, sigma=9, spacing=None, convert2lab=True,
                    enforce_connectivity=True, min_size_factor=0.5,
                    max_size_factor=3, slic_zero=False, start_label=1, mask=None, channel_axis=-1)

    sift = cv2.SIFT_create(300)  # Create SIFT object
    keypoints_sift, descriptors = sift.detectAndCompute(gray_scale, None)
    for position,point in enumerate(keypoints_sift):#get in which superpixels sift features belong to
        x,y=point.pt
        x=int(round(x))
        y=int(round(y))
        array_as_list=descriptors[position].tolist()
        if segments[x][y] in SIFT_DATA:
            SIFT_DATA[segments[x][y]].append(array_as_list)
        else:
            SIFT_DATA[segments[x][y]]=[array_as_list]


    for key,value in SIFT_DATA.items():#calculate average value of features in all superpixels
        SIFT_DATA[key]=np.mean(value, axis = 0).tolist()

    Lab = color.rgb2lab(img)
    L, a, b = cv2.split(Lab)

    for i in np.unique(segments):#get gabor filters for every superpixel
        mask = np.ones(img.shape[:2])
        mask[segments == i] = 0
        superpixel_gabor=[]
        for lamda in np.arange(np.pi/5, np.pi+np.pi/5, np.pi / 5):
            for theta in np.arange(0, np.pi, np.pi / 8):
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                fimg = cv2.filter2D(gray_scale, cv2.CV_8UC3, kernel)
                temp_i=np.ma.masked_array(fimg, mask=mask)
                mean_gabor=np.average(temp_i)
                superpixel_gabor.append(mean_gabor)
        temp_a = np.ma.masked_array(a, mask=mask)
        temp_b = np.ma.masked_array(b, mask=mask)
        amean = np.average(temp_a)
        bmean = np.average(temp_b)
        temp=[[amean,bmean]]
        label=findClusterOfSuperpixel(temp,Clusters)
        if i not in SIFT_DATA:
            SIFT_DATA[i]=128*[0]
        ab_colors+=label.tolist()
        features.append(SIFT_DATA[i]+superpixel_gabor)

    return features,ab_colors


def SVM(Features,Labels):#svm classifier
    classifier=SVC(gamma=50,C=100).fit(Features,Labels)
    return classifier

#[OPTIONAL] Gabor filtered images in 5 scales and 8 orientations
def DisplayAllGaborImagesKernels():
    for i in range(1,RELEVANT):
        image=IMAGE+str(i)+EXTENSION
        img = skimage.io.imread(fname=image)
        img = cv2.resize(img, (DESIRABLE_SIZE, DESIRABLE_SIZE))
        gray_scale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        DisplayGaborKernels(gray_scale,False)


#[OPTIONAL] Get all gabor kernels in 5 scales and 8 orientations
def DisplayGaborKernels(IMG,kernel=True):
    counter=0
    fig = plt.figure(figsize=(10, 10))
    counter = 0
    rows = 5
    columns = 8
    for lamda in np.arange(np.pi / 5, np.pi + np.pi / 5, np.pi / 5):
        for theta in np.arange(0, np.pi, np.pi / 8):
            counter += 1
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
            # showing image
            fig.add_subplot(rows, columns, counter)
            # showing image
            if kernel:
                plt.imshow(kernel)
            else:
                fimg = cv2.filter2D(gray_scale, cv2.CV_8UC3, kernel)
                plt.imshow(fimg)
            plt.axis('off')
            format_lamda = "{:.2f}".format(lamda)
            format_theta = "{:.2f}".format(theta)
            plt.title(f"S: {format_lamda},O: {format_theta}")
    plt.show()

def TestingImage(gray_image,classifier,Clusters):
    Testing={}
    centroids = Clusters.cluster_centers_
    img = skimage.io.imread(fname=gray_image)
    img = cv2.resize(img, (DESIRABLE_SIZE, DESIRABLE_SIZE))
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    CIELab = color.rgb2lab(img)
    L, a, b = cv2.split(CIELab)

    segments = slic(img, n_segments=SUPERPIXELS, compactness=1.0, max_num_iter=70, sigma=9, spacing=None, convert2lab=True,
                    enforce_connectivity=True, min_size_factor=0.5,
                    max_size_factor=3, slic_zero=False, start_label=1, mask=None, channel_axis=-1)

    sift = cv2.SIFT_create(300)  # Create SIFT object
    keypoints_sift, descriptors = sift.detectAndCompute(gray, None)

    for position, point in enumerate(keypoints_sift):
        x, y = point.pt
        x = int(round(x))
        y = int(round(y))
        array_as_list = descriptors[position].tolist()
        if segments[x][y] in Testing:
            Testing[segments[x][y]].append(array_as_list)
        else:
            Testing[segments[x][y]] = [array_as_list]

    for key, value in Testing.items():
        Testing[key] = np.mean(value, axis=0).tolist()


    for i in np.unique(segments):
        mask = np.ones(img.shape[:2])
        mask[segments == i] = 0
        superpixel_gabor = []
        for lamda in np.arange(np.pi/5, np.pi+np.pi/5, np.pi / 5):
            for theta in np.arange(0, np.pi, np.pi / 8):
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                fimg = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                temp_i = np.ma.masked_array(fimg, mask=mask)
                mean_gabor = np.average(temp_i)
                superpixel_gabor.append(mean_gabor)

        if i not in Testing:
            Testing[i] = 128 * [0]

        Features=Testing[i]+superpixel_gabor
        Labels=classifier.predict([Features])
        estimated_color=centroids[Labels[0]]
        a[segments == i] = estimated_color[0]
        b[segments == i] = estimated_color[1]


    Lab = np.dstack([L, a, b])
    RGB = color.lab2rgb(Lab)
    RGB=cv2.resize(RGB, (1100, 1100))
    plt.imshow(RGB)
    plt.show()



def TrainingModel(DIRECTORY,TEST):
    X_train=[]
    corresponding_Labels=[]
    Color_Vector = []
    print('----------------------Time to calculate clusters using K-Means------------------------------------')
    # Extract a and b color vectors for all training images
    for filename in os.listdir(DIRECTORY):
        file = os.path.join(DIRECTORY, filename)
        print(file)
        # checking if it is a file
        if os.path.isfile(file):
            Color_Vector = RGBtoLAB(file, Color_Vector)

    #Calculate centroids using K-means
    Kmean_Clusters=Kmeans(CLUSTERS,Color_Vector)

    print('-------------------------Time to extract features for all relevant images-----------------------------------------')
    for filename in os.listdir(DIRECTORY):
        file = os.path.join(DIRECTORY, filename)
        # checking if it is a file
        if os.path.isfile(file):
            train_data,labels_data=extractFeatures(file,Kmean_Clusters)
            X_train+=train_data
            corresponding_Labels+=labels_data

    print('-------------------------------Time to train classsifier-----------------------------------------------------')
    SVM_classifier=SVM(X_train,corresponding_Labels)
    print('---------------------------------------Results------------------------------------------')
    TestingImage(TEST,SVM_classifier,Kmean_Clusters)


if __name__ == "__main__":

    DIRECTORY = input('Type directory path and folder name of training images : ')
    TESTING_IMAGE=input('Type path and name of testing image : ')
    print('The process will take approximately 15 minutes for CPU [Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz   2.30 GHz]')
    TrainingModel(DIRECTORY,TESTING_IMAGE)
