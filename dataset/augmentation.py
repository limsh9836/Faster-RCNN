from typing import Type
import torch
import numpy as np
import cv2
import random

def read_image(path, convert=True):
    """
    Read image in provided path

    Args:
        path (string): Image path
        convert (boolean): Convert image to RGB
    Return:
        ndarray[H, W, C]: Image in numpy array
    """
    image = cv2.imread(path)
    if convert:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    return image

def resize_box(boxes, in_size, out_size):
    """
    Resize bounding boxes according to image resize
    
    Args:
        boxes (ndarray[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        in_size (tuple): Image original size in format (H, W)
        out_size (tuple): Image resized size in format (H, W)
    Return:
        ndarray[N, 4]: Resized bouding boxes in format (x1, y1, x2, y2)
    """
    assert isinstance(in_size, tuple) and len(in_size) == 2
    assert isinstance(out_size, tuple) and len(out_size) == 2

    boxes = boxes.copy()
    scale = (out_size[0] / in_size[0], out_size[1] / in_size[1])

    boxes[:, ::2] = boxes[:, ::2] * scale[1]
    boxes[:, 1::2] = boxes[:, 1::2] * scale[0]

    return boxes


def translate_box(boxes, offset):
    """
    Translate bounding boxes by horizontal and vertical offset

    Args:
        boxes (ndarray[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        offset (Union(tuple, int)): Horizontal and vertical offset
    Return:
        ndarray[N, 4]: Translated bounding boxes
    """
    if isinstance(offset, tuple) and len(offset) == 2:
        offset_x, offset_y = offset
    elif isinstance(offset, int):
        offset_x = offset_y = offset
    else:
        raise TypeError("Expected int or tuple, get {} type instead".format(type(offset)))
    
    boxes = boxes.copy()
    boxes = boxes + np.array([offset_x, offset_y, offset_x, offset_y])
    
    return boxes

def flip_image(image, x_flip=False, y_flip=False):
    """
    Flip image along the axis

    Args:
        image (ndarray[H, W, C]): Input image
        x_flip (boolean): Flip along horizontal axis
        y_flip (boolean): Flip along vertical axis
    Return:
        ndarray[H, W, C]: Flipped image
    """
    image = image.copy()

    if x_flip:
        image = image[:, ::-1, :]
    if y_flip:
        image = image[::-1, :, :]
    
    return image

def flip_box(boxes, size, x_flip=False, y_flip=False):
    """
    Flip bounding boxes along the axis

    Args:
        boxes (ndarray[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        size (tuple): Height and width of the image
        x_flip (boolean): Flip along horizontal axis
        y_flip (boolean): Flip along vertical axis
    Return:
        ndarray[N, 4]: Flipped bounding boxes in format (x1, y1, x2, y2)
    """
    assert isinstance(size, tuple)

    boxes = boxes.copy()
    H, W = size
    
    if x_flip:
        x1 = W - boxes[:, 2]
        x2 = W - boxes[:, 0]
        boxes[:, 0] = x1
        boxes[:, 2] = x2
        # boxes[:, 0::2] = x1, x2

    if y_flip:
        y1 = H - boxes[:, 3]
        y2 = H - boxes[:, 1]
        boxes[:, 1] = y1 
        boxes[:, 3] = y2
        # boxes[:, 1::2] =s y1, y2

    return boxes


def rotate_image(image, angle, scale=1.):
    """
    Rotate image according to the given angle

    Args:
        image (ndarray[H, W, C]): Input image
        angle (int): Rotation angle
        scale (float): Scale
    Return:
        ndarray[nH, nW, C]: Rotated image
    """
    image = image.copy()

    H, W = image.shape[:2]
    cX, cY = W / 2, H / 2
    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
    cos_theta, sin_theta = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW = int(W * cos_theta + H * sin_theta)
    nH = int(H * cos_theta + W * sin_theta)
    M[0, 2] += nW / 2 - cX
    M[1, 2] += nH / 2 - cY
    image = cv2.warpAffine(image, M, (nW, nH))

    return image

def rotate_box(image, boxes, angle, scale=1.):
    """
    Rotate bounding boxes in image according to the given angle

    Args:
        image (ndarray[H, W, C]): Input image
        boxes (ndarray[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        angle (int): Rotation angle
        scale (float): Scaling factor
    Return:
        ndarray[nH, nW, C]: Rotated image
    """
    image = image.copy()
    boxes = boxes.copy()

    H, W = image.shape[:2]
    cX, cY = W / 2, H / 2
    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
    cos_theta, sin_theta = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW = int(W * cos_theta + H * sin_theta)
    nH = int(H * cos_theta + W * sin_theta)
    M[0, 2] += nW / 2 - cX
    M[1, 2] += nH / 2 - cY

    for i, box in enumerate(boxes):
        box = box.reshape(2,2)
        ones = np.ones(2,1)
        box = np.concatenate((box, ones), axis=1)
        box = M @ box.T
        box = box.reshape(-1)
        boxes[i] = box
    
    return boxes

def scale_image(image, scale):
    """
    Scale image according to the given scaling factor

    Args:
        image (ndarray[H, W, C]): Input image
        scale (Union(tuple, float, int)): Scaling factor
    Return:
        ndarray[H, W, C]: Scaled image
    """
    if isinstance(scale, tuple):
        scale_x, scale_y = scale
    elif isinstance(scale, float) or isinstance(scale, int):
        scale_x = scale
        scale_y = scale
    else:
        raise TypeError("Expected tuple, float or int, but get {} instead".format(type(scale)))
    
    image = image.copy()
    H, W, C = image.shape
    nH, nW = int(H * scale_y),  int(W * scale_x)
    
    image = cv2.resize(image, (nW, nH))
    
    rescaled_image = np.zeros((H, W, C), dtype=np.int32)
    x = int(min(scale_x, 1) * W) 
    y = int(min(scale_y, 1) * H)
    
    rescaled_image[:y, :x, :] = image[:y, :x, :]
    
    return rescaled_image

def scale_box(boxes, scale):
    """
    Scale bounding boxes according to the given scaling factor

    Args:
        boxes (ndarray[N, 4]): Bounding boxes in the format (x1, y1, x2, y2)
        scale (Union(tuple, float, int)): Scaling factor
    Return:
        ndarray[N, 4]: Scaled bounding boxes
    """
    if isinstance(scale, tuple):
        scale_x, scale_y = scale
    elif isinstance(scale, float) or isinstance(scale, int):
        scale_x = scale
        scale_y = scale
    else:
        raise TypeError("Expected tuple, float or int, but get {} instead".format(type(scale)))
    
    boxes = boxes.copy()
    boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y])
    
    return boxes

def translate_image(image, translate):
    """
    Translate image according to the translation factor

    Args:
        image (ndarray[H, W, C]): Input image
        translate (Union(tuple, float, int)): Translation factor
    Return:
        ndarray[H, W, C]: Translated image
    """
    if isinstance(translate, tuple):
        translate_factor_x, translate_factor_y = translate
    elif isinstance(translate, float) or isinstance(translate, int):
        translate_factor_x = translate
        translate_factor_y = translate
    else:
        raise TypeError("Expected tuple, float or int, but get {} instead".format(type(translate)))
    image = image.copy()
    H, W, C = image.shape
    translated_image = np.zeros((H, W, C), dtype=np.uint8)
    
    corner_x = W * translate_factor_x
    corner_y = H * translate_factor_y
    
    coord = [max(0, corner_y), max(0, corner_x), min(H, corner_y+H), min(W, corner_x + W)]
    mask = image[
        max(-corner_y, 0):min(H, H-corner_y), max(-corner_x, 0):min(W, W-corner_x), :
    ]
    translated_image[coord[0]:coord[2], coord[1]:coord[3], :] = mask
    
    return translated_image

def translate_box(boxes, size, translate):
    """
    Translate bounding boxes according to the given translation factor

    Args:
        boxes (ndarray[N, 4]): Bounding boxes in the format (x1, y1, x2, y2)
        size (tuple): Height and width of the image
        translate (Union(tuple, float, int)): Translation factor
    """
    if isinstance(translate, tuple):
        translate_factor_x, translate_factor_y = translate
    elif isinstance(translate, float) or isinstance(translate, int):
        translate_factor_x = translate
        translate_factor_y = translate
    else:
        raise TypeError("Expected tuple, float or int, but get {} instead".format(type(translate)))
    boxes = boxes.copy()
    H, W = size
    
    corner_x = W * translate_factor_x
    corner_y = H * translate_factor_y
    
    boxes = boxes + np.array([corner_x, corner_y, corner_x, corner_y])
    
    return boxes

def clip_box(boxes, border, alpha=.25):
    """
    Clip the bounding boxes to the border of the image
    
    Args:
        boxes (ndarray[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        border (ndarray[4]): Border of image in format (x1, y1, x2, y2)
        alpha (float): Fraction of bounding box left in the image after being clipped less than alpha is dropped
    Return:
        ndarray[M, 4]: Clipped boxes in format (x1, y1, x2, y2)
    """
    boxes = boxes.copy()

    xmin = np.maximum(boxes[:, 0], border[0])
    ymin = np.maximum(boxes[:, 1], border[1])
    xmax = np.minimum(boxes[:, 2], border[2])
    ymax = np.minimum(boxes[:, 3], border[3])
    clipped = np.stack((xmin, ymin, xmax, ymax), axis=-1)

    box_area = area(boxes)
    clipped_area = area(clipped)
    delta = (box_area - clipped_area) / box_area
    
    keep = delta < (1 - alpha)
    boxes = boxes[keep]

    return boxes
    
def area(boxes):
    """
    Compute bounding boxes area
    
    Args:
        boxes (ndarray[N, 4]): Bounding boxes in the format (x1, y1, x2, y2)
    Return:
        ndarray[N]: Bounding boxes area
    """
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return area 
    
    
def random_horizontal_flip(image, boxes):
    """
    Randomly flip image and bounding boxes along horizontal axis

    Args:
        image (ndarray[H, W, c]): Input image
        boxes (ndarray[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
    Return:
        image (ndarray[H, W, c]): Randomly flipped image along horizontal axis
        boxes (ndarray[N, 4]): Randomly flipped bounding boxes along horizontal axis
    """
    image = image.copy()
    boxes = boxes.copy()
    flip = random.choice([True, False])
    size = image.shape[:2]

    if flip:
        # print("Flip")
        image = image[:, ::-1, :]
        boxes = flip_box(boxes, size, x_flip=True)

    return image, boxes

def random_flip(image, boxes):
    """
    Randomly flip image and bounding boxes along vertical and horizontal axis

    Args:
        image (ndarray[H, W, c]): Input image
        boxes (ndarray[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
    Return:
        image (ndarray[H, W, c]): Randomly flipped image along vertical and horizontal axis
        boxes (ndarray[N, 4]): Randomly flipped bounding boxes along vertical and horizontal axis
    """
    image = image.copy()
    boxes = boxes.copy()
    size = image.shape[:2]

    x_flip = random.choice([True, False])
    y_flip = random.choice([True, False])

    image = flip_image(image, x_flip, y_flip)
    boxes = flip_box(boxes, size, x_flip=x_flip, y_flip=y_flip)

    return image, boxes

def random_rotate(image, boxes, angles=[0, 90, 180, 270]):
    """
    Randomly rotate image and bounding boxes according to the provided angle

    Args:
        image (ndarray[H, W, c]): Input image
        boxes (ndarray[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        angles (list): Candidate angles to perform rotation
    Return:
        image (ndarray[H, W, c]): Randomly rotated image according to the randomly selected angle
        boxes (ndarray[N, 4]): Randomly rotated bounding boxes according to the randomly selected angle
    """  
    image = image.copy()
    boxes = boxes.copy()

    angle = random.choice(angles)

    rotated_image = rotate_image(image, angle)
    rotated_boxes = rotate_box(image, boxes, angle)

    return rotated_image, rotated_boxes

def random_scale(image, boxes, scale=.2):
    """
    Randomly scale image and bounding boxes according to the scaling factor

    Args:
        image (ndarray[H, W, c]): Input image
        boxes (ndarray[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        scale (float): Range to sample the scaling factor
    Return:
        image (ndarray[H, W, c]): Randomly scaled image according to the sampled scaling factor
        boxes (ndarray[N, 4]): Randomly scaled bounding boxes according to the sampled scaling factor
    """  
    scale_x = random.uniform(-scale, scale)
    scale_y = random.uniform(-scale, scale)
    H, W = image.shape[:2]

    image = scale_image(image, (scale_x, scale_y))
    boxes = scale_box(boxes, (scale_x, scale_y))
    boxes = clip_box(boxes, [0, 0, W, H])

    return image, boxes

def random_translate(image, boxes, translate=0.2):
    """
    Randomly translate image and bounding boxes according to the translation factor

    Args:
        image (ndarray[H, W, c]): Input image
        boxes (ndarray[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        translate (float): Range to sample the translation factor
    Return:
        image (ndarray[H, W, c]): Randomly translated image according to the sampled translation factor
        boxes (ndarray[N, 4]): Randomly translated bounding boxes in the sampled translation factor
    """  
    translate_x = random.uniform(-translate, translate)
    translate_y = random.uniform(-translate, translate)
    H, W = image.shape[:2]

    image = translate_image(image, (translate_x, translate_y))
    boxes = translate_box(boxes, (H, W), (translate_x, translate_y))
    boxes = clip_box(boxes, [0, 0, W, H])

    return image, boxes