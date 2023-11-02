import cv2
import imutils
from .Stitch import Stitcher

"""
self.image_path   : str, path/to/image
self.image        : cv image, numpy array.
"""

class Img:
  def __init__(self, image):
    self.image_path = image
    self.image = cv2.imread(image)

  def getH(self, image, img_sz):
    image = imutils.resize(image, height=img_sz[0], width=img_sz[1])
    base_image = imutils.resize(self.image, height=img_sz[0], width=img_sz[1])
    stitcher = Stitcher()
    _, H = stitcher.stitch([base_image, image])
    return H

  def _save_json(self):
    return {'image': self.image_path}

  @classmethod
  def _load_from_json(self, j):
    image_path = j['image']
    return Img(image_path)