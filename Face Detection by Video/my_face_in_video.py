# -*- coding: utf-8 -*-
"""My_Face in Video.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zGDACBbNvyDEbsiG5hSG7RMimig7vtNF

Video Sources:  https://www.youtube.com/watch?v=SfpPB9fMVKA
"""

#Installing modules we need. And doing it only once.
import pkgutil; 
if not pkgutil.find_loader("missingno"):
  !pip install missingno -q

def get_file(url):
  fname = url.split('/')[-1]
  if not Path(fname).exists():
    print("Getting ", fname)
    !wget {url} -q

#importing modules we need 
from pathlib import Path
from matplotlib import pyplot as plt

import cv2
import os
import numpy as np

#downloading files. This will run only once.
get_file("https://www.dropbox.com/s/mq7julne4cudghx/haarcascade_frontalface_default.xml")

get_file("https://www.dropbox.com/s/2q3ouvb5r8ejedm/videoplayback%20%281%29.3gp?dl=0")
camera = cv2.VideoCapture("videoplayback (1).3gp?dl=0")

def imshow(image):
  plt.grid(False)
  if len(image.shape) == 3:
    conv = cv2.COLOR_BGR2RGB
  else:
    conv = cv2.COLOR_GRAY2RGB
  plt.imshow(cv2.cvtColor(image,conv ))

def detect_face(image, scaleFactor = 1.2, minNeighbors = 5, minSize = (30, 30)):
  faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
  gr_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  rects = faceCascade.detectMultiScale(gr_image,
               scaleFactor = scaleFactor,
               minNeighbors = minNeighbors, minSize = minSize,
               flags = cv2.CASCADE_SCALE_IMAGE)
  for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  return image

def make_video(outvid, images=None, fps=30, size=None,
               is_color=True, format="FMP4"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

rm *.jpg

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

ROOT_DIR = os.getcwd()

camera = cv2.VideoCapture("videoplayback (1).3gp?dl=0")
frame_count = 0 
frames = []
while True:
  (grabbed, frame) = camera.read()

  if not grabbed:
    break

  frame = resize(frame, width = 300)
  detect_face(frame, scaleFactor = 1.1, minNeighbors = 5,
                      minSize = (30, 30))
  if (frame_count%50 == 0):
    print (frame_count)
  frame_count += 1
  name = '{0}.jpg'.format(frame_count)
  name = os.path.join(ROOT_DIR, name)
  cv2.imwrite(name, frame)

rm *.mp4

import glob
import os

# Directory of images to run detection on
ROOT_DIR = os.getcwd()
images = list(glob.iglob(os.path.join(ROOT_DIR, '*.jpg')))
# Sort the images by integer index
images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

outvid = os.path.join(ROOT_DIR, "out.mp4")
make_video(outvid, images, fps=30)

ls -l *.mp4

from google.colab import files

files.download('out.mp4')

