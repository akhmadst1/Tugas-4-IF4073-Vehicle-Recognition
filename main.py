import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import matplotlib.pyplot as plt
import cv2
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import tkinter as tk
import cv2
import numpy as np
import tkinter.messagebox 

def load_image(file):
    # load image
    image = tf.io.read_file(file)
    # detect format (JPEG, PNG, BMP, or GIF) and converts to Tensor:
    image = tf.io.decode_image(image)
    return image

def browse_image():
    # browse image in gui from computer
    global panelA, panelB, image, filename
    f_types = [('Jpg Files', '*.jpg'),('PNG Files','*.png')] 
    path = filedialog.askopenfilename(filetypes=f_types)
    filename = path.split("/")[-1]
    # display image in gui
    image = cv2.imread(path)
    image = cv2.resize(image, (576,416))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image)
    image1 = ImageTk.PhotoImage(image1)
    panelA = Label(image=image1, borderwidth=5, relief="sunken")
    panelA.image = image1
    panelA.grid(row= 2, column=0 , rowspan= 13, columnspan= 3, padx=20, pady=20)

def browse_video():
    # browse video in gui from computer
    global panelC, filenamevideo
    f_types = [('Mp4 Files', '*.mp4'),('AVI Files','*.avi')] 
    path = filedialog.askopenfilename(filetypes=f_types)
    filenamevideo = path.split("/")[-1]
    # display video name in gui
    panelC = Label(text=filenamevideo, relief="groove", font=('calibre',10,'normal'))
    panelC.grid(row= 17, column=0)
    
def resize_image(image):
    WIDTH, HEIGHT = (576,416)
    # resize the output_image:
    image = tf.image.resize(image, (HEIGHT, WIDTH))
    # add a batch dim:
    images = tf.expand_dims(image, axis=0)/255
    return images

def model_YOLOV4():
    WIDTH, HEIGHT = (576,416)
    # load trained yolov4 model
    model = YOLOv4(
        input_shape=(HEIGHT, WIDTH, 3),
        anchors=YOLOV4_ANCHORS,
        num_classes=80,
        training=False,
        yolo_max_boxes=50,
        yolo_iou_threshold=0.5,
        yolo_score_threshold=0.4,
    )
    model.load_weights('yolov4.h5')
    return model

def detect_image(boxes, scores, classes, detections,image):
    WIDTH, HEIGHT = (576,416)
    boxes = (boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(int)
    scores = scores[0]
    classes = classes[0].astype(int)
    detections = detections[0]

    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    image_cv = image.numpy()

    for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):
        if score > 0:
            # add bounding box 
            # convert from tf.Tensor to numpy
            cv2.rectangle(image_cv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (83,59,100), thickness= 2)
            # Add detection text to the prediction
            text = CLASSES[class_idx] + ': {0:.2f}'.format(score)
            cv2.putText(image_cv, text, (int(xmin), int(ymin) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (83,59,100), 2)

    return image_cv

def proccess_frame(image, model):
    # resize and do detection per frame
    images = resize_image(image)
    boxes, scores, classes, detections = model.predict(images)
    result_img = detect_image(boxes, scores, classes, detections,images[0])
    return result_img, detections

def vehicle_recognition_image(filename):
    # input image
    image = load_image(filename)
    # load trained yolov4 model
    model = model_YOLOV4()
    # do detection
    image, detections = proccess_frame(image, model)
    # display result to gui
    image1 = Image.fromarray((image*255).astype(np.uint8))
    image1 = ImageTk.PhotoImage(image1)
    panelB = Label(image=image1, borderwidth=5, relief="sunken")
    panelB.image = image1
    panelB.grid(row= 2, column=15 , padx=20, pady=20)
    return image, detections

def vehicle_recognition_video(input_video_name, output_video_name, frames_to_save = 50):
    WIDTH, HEIGHT = (576,416)
    # load trained yolov4 model
    model = model_YOLOV4()
    # load video
    my_video = cv2.VideoCapture(input_video_name)
    # write resulted frames to file
    out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (WIDTH ,HEIGHT))

    success = 1
    i = 0
    # While there are more frames in the video
    while success and i < frames_to_save:                                 
        # function extract frames
        success, image = my_video.read()                             
        if success:
            result_img, detections = proccess_frame(tf.convert_to_tensor(image), model) 
            # write resulted frame to the video file
            out.write((result_img*255).astype(np.uint8))                                             
            i = i + 1
            print(i)
    out.release()

def onClick_detect_video():
    # pop up message when detect video is done
    tkinter.messagebox.showinfo("Video Vehicle Recognition",  "Vehicle Detection from Video is Complete. Open your Output Filename Video to Check The Result")

#GUI
root = Tk()
root.title("VEHICLE RECOGNITION")
width= root.winfo_screenwidth() 
height= root.winfo_screenheight()
root.geometry("%dx%d" % (width, height))
output_var = tk.StringVar()
frame_var = tk.StringVar()

btn= Button(root, text="BROWSE IMAGE", fg="black", bg="lavender", command=browse_image)
btn.grid(row= 0, column= 0, padx=10, pady=10, sticky='nesw')

btn1= Button(root, text="DETECT IMAGE VEHICLES", fg="white", bg="red", command= lambda: vehicle_recognition_image(filename))
btn1.grid(row= 0, column= 1, padx=10, pady=10, sticky='nesw')

image_label = tk.Label(root, text='Input Image', font=('calibre',10, 'bold'))
image_label.grid(row=1, column=1)
image_entry = tk.Label(borderwidth=5, relief="sunken", height = 28, width = 82)
image_entry.grid(row= 2, column=0 , rowspan= 13, columnspan= 3, padx=20, pady=20)

detected_label = tk.Label(root, text='Detected Image', font=('calibre',10, 'bold'))
detected_label.grid(row=1, column=15)
detected_entry = tk.Label(borderwidth=5, relief="sunken", height = 28, width = 82)
detected_entry.grid(row= 2, column=15 , padx=20, pady=20)

btn= Button(root, text="BROWSE VIDEO", fg="black", bg="lavender", command=browse_video)
btn.grid(row= 15, column= 0, padx=10, pady=10, sticky='nesw')

btn1= Button(root, text="DETECT VIDEO VEHICLES", fg="white", bg="blue", command= lambda: [vehicle_recognition_video(filenamevideo, output_var.get(), int(frame_var.get())), onClick_detect_video()])
btn1.grid(row= 15, column= 1, padx=10, pady=10, sticky='nesw')

video_label = tk.Label(root, text = 'Input Filename Video', font=('calibre',10, 'bold'))
video_label.grid(row=16, column=0)

output_label = tk.Label(root, text = 'Output Filename Video', font=('calibre',10, 'bold'))
output_label.grid(row=16, column=1)
output_entry = tk.Entry(root,textvariable = output_var, font=('calibre',10,'normal'))
output_entry.grid(row=16, column=2)

frame_label = tk.Label(root, text = 'Frame to Save', font=('calibre',10, 'bold'))
frame_label.grid(row=17, column=1)
frame_entry = tk.Entry(root,textvariable = frame_var, font=('calibre',10,'normal'))
frame_entry.grid(row=17, column=2)

root.mainloop()