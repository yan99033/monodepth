import cv2
import numpy as np

# Given any image, crop it according to the aspect ratio (2:1), assuming width is larger than height
def crop(img):
    height, width = img.shape[:2]

    # To keep the aspect ratio
    if height * 2 >= width:
        new_height = int(width / 2)
        y_offset = int((width - height) / 2)
        img = img[y_offset:y_offset+new_height, 0:width, :]
        # print '{} {}'.format('new image shape', img.shape)
        img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
    else:
        new_width = height * 2
        x_offset = int((width - new_width) / 2)
        img = img[0:height, x_offset:x_offset+new_width, :]
        # print '{} {}'.format('new image shape', img.shape)
        img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)

    return img

# Improve disparity map by merging left depth map with flipped right depth map
def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

