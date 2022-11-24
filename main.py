# Importing the instance_segmentation object type from pixellib.instance
import pixellib
import cv2
import tensorflow as tf
from pixellib.instance import instance_segmentation
from IPython.display import Video

# Setting the file path for the input video
input_video_file_path = "test_video.mp4"
Video(input_video_file_path, embed=True)
# Creating an instance_segmentation object
segment_video = instance_segmentation()

# Loading the Mask R-CNN model trained on the COCO dataset
segment_video.load_model("mask_rcnn_coco.h5")

# Processing the video
segment_video.process_video(
    input_video_file_path, 
    show_bboxes=True, 
    extract_segmented_objects=True, 
    save_extracted_objects=False,
    frames_per_second=30,
    output_video_name="instance_segmentation_output.mp4",
)