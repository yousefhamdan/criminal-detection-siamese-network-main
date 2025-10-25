import streamlit as st
import cv2
import os
import subprocess
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import os
from glob import glob
import shutil
import numpy as np
from utility.model_utils import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from ffmpy import FFmpeg

face=YOLO("yolov8n-face.pt")

def empty_files():
    
    shutil.rmtree("detected_faces", ignore_errors=True)
    shutil.rmtree("uploaded_videos", ignore_errors=True)
    shutil.rmtree("video_detection", ignore_errors=True)

    os.mkdir("detected_faces")
    os.mkdir("uploaded_videos")
    os.mkdir("video_detection")
    

def get_conf():
    confidences=[]
    for img in glob("detected_faces/*.jpg"):
        confidences.append(float(img.split("_")[-1].split(".jpg")[0]))
    return confidences

def best_images(img_folder,confidences):
    
    # Load all images in the folder and preprocess them for ResNet18
    features=[]
    orig_images=[]
    for filename in glob("detected_faces/*"):
        features.append(get_image_embedding(filename))
        orig_images.append(cv2.imread(filename))
    
    features=np.array(features)

    # Flatten the features into a 2D numpy array
    features = features.reshape(features.shape[0], -1)

    # Apply hierarchical clustering to the feature data
    hierarchical = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=0.15)
    hierarchical.fit(features)

    # Get the labels assigned to each image
    labels = hierarchical.labels_.astype(int)

    # Create a new folder for the highest score images
    high_score_folder = os.path.join(img_folder, 'highest_score_images')
    if not os.path.exists(high_score_folder):
        os.mkdir(high_score_folder)
    # Process images from each cluster
    for i in range(np.max(labels) + 1):
        print(f'Processing images in cluster {i}...')
        # Get the indices of the images in this cluster
        indices = np.where(labels == i)[0]
        # Calculate the score for each image in this cluster
        scores = []
        for idx in indices:
            img = orig_images[idx]
            brightness_score = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
            contrast_score = np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            sharpness_score = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            confidence = confidences[idx]  # get the confidence value for this image
            score = (0.5 * brightness_score + 0.3 * contrast_score + 0.2 * sharpness_score) * confidence
            scores.append(score)
        # Save the image with the highest score to the appropriate folder
        highest_score_idx = np.argmax(scores)
        highest_score_img = orig_images[indices[highest_score_idx]]
        filename = os.path.join(high_score_folder, f'score_{scores[highest_score_idx]:.4f}_{indices[highest_score_idx]}.jpg')
        cv2.imwrite(filename, highest_score_img)


def image_face_detector(image,n,conf_thresh=0.75,flag=True):
    img=image.copy()
    results=face(img)
    boxes=results[0].boxes.xyxy
    i=1
    for box in boxes:
        (x1,y1,x2,y2)=[int(x) for x in box.tolist()]
        cv2.rectangle(img,(x1,y1),(x2,y2),(255, 0, 0),2)
        (w, h), _ = cv2.getTextSize("face", cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)

        img = cv2.rectangle(img, (x1, y1 - 20), (x1+ w, y1), (255, 0, 0), -1)
        img = cv2.putText(img, "face", (x1, y1 - 5),cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
        if (float(results[0].boxes.conf[i-1])>=conf_thresh):
 
            try:
                if flag:
                    cv2.imwrite(f"detected_faces/img{n}_face{i}_{float(results[0].boxes.conf[i-1])}.jpg",image[y1:y2, x1:x2,::-1])
                else:
                    cv2.imwrite(f"detected_faces/img{n}_face{i}_{float(results[0].boxes.conf[i-1])}.jpg",image[y1:y2, x1:x2])
                i=i+1
            except:
                print("empty img error")
        else:
            i=i+1
            continue
    return img

def video_face_detector(uploaded_file):
    path=os.path.join("uploaded_videos",uploaded_file.name)
    
    with open(path,"wb") as f:
         f.write(uploaded_file.getbuffer())
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("video_detection/output.mp4",fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(3)),int(cap.get(4))))
    
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
            
    my_bar = st.progress(0, text="Processing the video. Please Wait.")
    counter=0
    n_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            out.write(image_face_detector(frame,counter+1,flag=False,conf_thresh=0.75))
            my_bar.progress((counter + 1)/n_frame, text="Processing the video. Please Wait.")
            counter=counter+1
        else:
            break
        
    cap.release()
    out.release()
    
    try:
        convertedVideo = "video_detection/testh264.mp4"
        ff = FFmpeg(executable='ffmpeg/ffmpeg.exe',
            inputs={'video_detection/output.mp4': "-y"},
            outputs={convertedVideo: '-c:v libx264'}
        )
        ff.run()
        st.video(convertedVideo)

    except:
        convertedVideo = "video_detection/testh264.mp4"
        ff = FFmpeg(
            inputs={'video_detection/output.mp4': "-y"},
            outputs={convertedVideo: '-c:v libx264'}
        )
        ff.run()
        st.video(convertedVideo)
    
def video_frame_callback(frame):
    if not hasattr(video_frame_callback, "frame_number"):
        video_frame_callback.frame_number = 1

    frame = frame.to_ndarray(format="bgr24")

    processed = image_face_detector(frame, video_frame_callback.frame_number, flag=False,conf_thresh=0.75)

    video_frame_callback.frame_number += 1

    return av.VideoFrame.from_ndarray(processed, format="bgr24")
