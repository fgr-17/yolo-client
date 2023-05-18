#!/opt/venv/bin/python

import base64
import io
import logging
import sys
import uuid
from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, UploadFile
from model import yolov5
from PIL import Image
from starlette.responses import Response

##########

import torch
import numpy as np
import cv2
from time import time

import socket, time

from PIL import Image

from io import BytesIO

from common import remote_data

import os

remote = remote_data("gst", os.environ['GST_YOLO_PORT'])

class ObjectDetectionGstreamer:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, url=None, port=None, out_file="Labeled_Video.avi"):
        """ """
        self._port = port
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """

        first_frame = True
        last_frame = False

        print(f'Connecting to {self._URL}:{self._port}')

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._URL, self._port))
            print(f"Connected to port:", self._port)

            while True:             
                
                #First Receive the frame number
                raw_frame = s.recv(4, socket.MSG_WAITALL)
                frame = int.from_bytes(raw_frame, byteorder="little")
                if frame == 2949: #TODO improve detection of last frame
                    last_frame = True
                
                #Second Receive the frame size
                raw_size = s.recv(4, socket.MSG_WAITALL)                
                size = int.from_bytes(raw_size, byteorder="little") 
                
                if (size == 0):
                    print(f"size is 0")
                
                #Third Receive the frame
                raw_img = s.recv(size, socket.MSG_WAITALL)                
                assert raw_img
                
                if (first_frame or last_frame):
                    print(f"Rcv Frame: {frame} - with lenght: {size}")

                #f = open (str(frame)+".jpg", "wb")
                #f.write(raw_img)
                #f.close()

                # init video output                
                file_jpgdata = np.asarray(bytearray(raw_img), dtype="uint8")
                dt = cv2.imdecode(file_jpgdata, cv2.IMREAD_COLOR)
                
                if first_frame:
                    x_shape = dt.shape[1]
                    y_shape = dt.shape[0]

                    #four_cc = cv2.VideoWriter_fourcc(*"MPEG")
                    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
                    first_frame = False


                results = self.score_frame(dt)
                frame = self.plot_boxes(results, dt)
                
                # end_time = time()
                # fps = 1/np.round(end_time - start_time, 3)
                # print(f"Frames Per Second : {fps}")
                                
                out.write(frame)

                if last_frame:
                    out.release()
                    break

# Create a new object and execute.
a = ObjectDetectionGstreamer(url=remote.get_ip(), port=remote.get_port())
a()