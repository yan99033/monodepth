#!/usr/bin/env python

# created by Shing-Yan (lsyan@ualberta.ca)
# Subscribe:
#   image topic
# Publish:
#   Depth map
#   image + depth Map

# CNN model is adapted from Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
# Paper: Unsupervised Monocular Depth Estimation with Left-Right Consistency

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np

from utils import *
from Model import *

import tensorflow as tf


# ROS Node
rospy.init_node('Monodepth', anonymous=True)

#ROS topic
image_topic = rospy.get_param('~image_topic')
cropped_image_topic = rospy.get_param('~out_image_topic')
depthimage_topic = rospy.get_param('~depth_topic')
depthimage_view_topic = rospy.get_param('~depth_topic_view')

# define publisher
pub_co = rospy.Publisher(cropped_image_topic, Image, queue_size=1)
pub_do = rospy.Publisher(depthimage_topic, Image, queue_size=1)
pub_dov = rospy.Publisher(depthimage_view_topic, Image, queue_size=1)

# ROS refresh rate (10hz)
# rate = rospy.Rate(30)

# Create CvBridge instance
bg = CvBridge()

# Initialize Tensorflow
## Placeholder for image input
x = tf.placeholder(tf.float32, shape=[None, 256, 512, 3])

## Load model
model = MonodepthModel(x, 'vgg')
y = model.get_output()

## Create session
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

## Model path (Tensorflow checkpoint)
ckpt = rospy.get_param('~path_to_model')

## Initialize variables
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

## Restore model
saver = tf.train.Saver()
saver.restore(sess, ckpt)

## Scale used to convert to depth (from disparity)
SCALE = 0.3128 # (718.856 * 0.54 / 1241) where 0.54 is stereo baseline distance


def img_callback(data):
    try:
        img = bg.imgmsg_to_cv2(data, "bgr8") #"passthrough") #"bgr8"
    except CvBridgeError as e:
        print(e)

    # Crop, resize and flip frame, and stack original and flipped frames
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop(img)

    # Publish cropped original image
    pub_co.publish(bg.cv2_to_imgmsg(img, 'rgb8'))

    # Convert to grayscale
    img_ori_color = img
    img_ori = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_ori_float = img_ori.astype(np.float32)

    img = img.astype(np.float32)
    img_flipped = np.fliplr(img)
    img = np.stack([img, img_flipped])
    img = img.reshape(2, 256, 512, 3)
    img = img / 255

    # Get disparity map from CNN
    output = sess.run([y], feed_dict={x: img})

    # Disparity_pp
    disp = post_process_disparity(output[0].squeeze())
    disp = disp.astype(np.float32)

    # Convert disparity to depth (inverse disparity)
    out = SCALE / disp

    # Combine depth and original images (view)
    min_val = np.amin(disp)
    max_val = np.amax(disp)
    disp_n = 255 * ((disp - min_val) / (max_val - min_val))
    disp_n = disp_n.astype(np.uint8)
    disp_n = cv2.applyColorMap(disp_n, 2) # cv2.COLORMAP_JET
    # img_ori_color = cv2.cvtColor(img_ori_color, cv2.COLOR_RGB2BGR)
    depth_ori_view = np.vstack((disp_n, img_ori_color))
    depth_ori_view = depth_ori_view.astype(np.uint8)

    # Combine depth and original images (for visual odom)
    depth_ori = np.vstack((out, img_ori_float))

    # Publish depth map and original images (view)
    pub_dov.publish(bg.cv2_to_imgmsg(depth_ori_view, encoding='bgr8')) # depthnoriginal_view
    pub_do.publish(bg.cv2_to_imgmsg(depth_ori, encoding='32FC1')) # depthnoriginal



def listener():
    rospy.Subscriber(image_topic, Image, img_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
