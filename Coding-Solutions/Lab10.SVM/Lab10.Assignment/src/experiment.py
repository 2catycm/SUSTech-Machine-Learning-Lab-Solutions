import cv2
from train import processFiles, trainSVM
from detector import Detector

# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "samples/vehicles"
neg_dir = "samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "videos/test_video.mp4" # 测试用的video，根据这个video做验证


def experiment1():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,spatial_features=True
                                , hist_bins=16, spatial_size=(20, 20))
                                
    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data)
    # detector = Detector(x_overlap=0.7, x_range=(0.2, 0.85),
    #                     y_step=0.004, y_range=(0.4, 0.8)
    #     )
    detector = Detector(x_overlap=0.7,y_step=0.004,  x_range=(0.5, 0.95), y_range=(0.6, 0.75)
        )
    detector = detector.loadClassifier(classifier_data=classifier_data)
  
    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)

    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap, show_video=False, write=True
                         )


def experiment2():
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,hog_features=True)
    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data)
    import pickle
    # pickle.s

if __name__ == "__main__":
    experiment1()
    # experiment2() 
    # experiment3 ...


