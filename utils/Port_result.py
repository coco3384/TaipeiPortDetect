import cv2
import json
import imutils
import numpy as np
from .utils import box_center
from .utils import NumpyArrayEncoder
from .Img import Img
from .Predict_result import Predict_result


"""
Class 概述
  # self.Img            : Img Class
  新一期對應的照片。
  # self.predict_result : Predict_result Class
  對應照片所偵測的結果。
  # self.base_info      : dict
  自基期.json紀錄的資訊。
    # 基期照片位置。
    # 基期照片大小。
    # 基期人工標註的物件框。
  # self.H              : numpy array
  基期照片對應新期照片的Homography Transformation。
  # self.isExist        : numpy array
  基期物件是否有找到對應物件的boolean list。
  # self.base_boxes     : 2D-numpy array，基期物件框。
  # self.predict_number : dict
  新一期照片找到的各物件數量。
  # self.missing_number : 基期物件未找到的數量。

  Function 概述
  # combine_result(Port_Result)       : 合併另一Port_Result Class的結果。
  # get_missing_number()              : 回傳dict，各物件種類缺失的數量。
  # get_predict_number()              : 回傳dict，各物件偵測到的數量。
  # save_json(file_path)              : 存成json file。
  # load_from_json(file_path)         : 回傳Port_Result Class，讀依照save_json格式儲存的json file。

  # 結果展示
    # show_find     : 回傳cv2 image(numpy array)，Img原圖與基期物件中有找到對應物件的物件（綠色框）
    # show_missing  : 回傳cv2 image(numpy array)，Img原圖與基期物件中未找到對應物件的物件（紅色框）
    # show_result   : 回傳cv2 image(numpy array)，Img原圖與這次的偵測結果，以點標示。
    # show_all      : 回傳cv2 image(numpy array)，Img原圖與上述三種資訊的結果。

"""
class Port_Result:
  def __init__(self, Img, predict_result, base_info, **kwargs):
      self.Img = Img
      self.predict_result = predict_result
      self.base_info = base_info
      if len(kwargs) == 0:
        self.H = self.Img.getH(cv2.imread(self.base_info['img']), self.predict_result.img_sz)
        self.base_boxes, self.isExist = self.predict_result.compare_result_with_base(self.base_info, self.H)
        self.missing_number = len(np.where(self.isExist == False)[0])
        self.predict_number = self.set_predict_number()
      else:
        self.H = kwargs['H']
        self.base_boxes = kwargs['base_boxes']
        self.isExist = kwargs['isExist']
        self.missing_number = kwargs['missing_number']
        self.predict_number = kwargs['predict_number']

  def combine_result(self, Port_Result):
    H = self.Img.getH(Port_Result.Img.image, self.predict_result.img_sz)
    if H is not None:
      reproject_Predict_result = Port_Result.predict_result.reproject_boxes(H)
      self.predict_result._combine_result(reproject_Predict_result)
      self.H = self.Img.getH(cv2.imread(self.base_info['img']), self.predict_result.img_sz)
      self.base_boxes, self.isExist = self.predict_result.compare_result_with_base(self.base_info, self.H)

  def show_find(self):
    img = imutils.resize(np.copy(self.Img.image), height=self.predict_result.img_sz[0], width=self.predict_result.img_sz[1])
    for box, exist in zip(self.base_boxes, self.isExist):
      if exist:
        cv2.rectangle(img, box[:2]-5, box[2:]+5, color=(0, 222, 0), thickness=1)
    return img

  def show_missing(self):
    img = imutils.resize(np.copy(self.Img.image), height=self.predict_result.img_sz[0], width=self.predict_result.img_sz[1])
    for box, exist in zip(self.base_boxes, self.isExist):
      if not exist:
        cv2.rectangle(img, box[:2]-5, box[2:]+5, color=(0, 0, 222), thickness=1)
    return img

  def show_result(self):
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    img = imutils.resize(np.copy(self.Img.image), height=self.predict_result.img_sz[0], width=self.predict_result.img_sz[1])
    predict_classes = self.predict_result.get_cls()
    predict_boxes = self.predict_result.get_boxes_n()
    for cls, box in zip(predict_classes, predict_boxes):
      center = box_center(box, (self.predict_result.img_sz[1], self.predict_result.img_sz[0]))
      cv2.circle(img, center, 3, color[int(cls)], -1)
    return img

  def show_all(self):
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    img = imutils.resize(np.copy(self.Img.image), height=self.predict_result.img_sz[0], width=self.predict_result.img_sz[1])
    predict_classes = self.predict_result.get_cls()
    predict_boxes = self.predict_result.get_boxes_n()
    # draw result
    for cls, box in zip(predict_classes, predict_boxes):
      center = box_center(box, (self.predict_result.img_sz[1], self.predict_result.img_sz[0]))
      cv2.circle(img, center, 3, color[int(cls)], -1)

    for box, exist in zip(self.base_boxes, self.isExist):
      if exist:
        cv2.rectangle(img, box[:2]-5, box[2:]+5, color=(0, 222, 0), thickness=1)
      else:
        cv2.rectangle(img, box[:2]-5, box[2:]+5, color=(0, 0, 222), thickness=1)

    return img

  def get_missing_number(self):
    return self.missing_number

  def set_predict_number(self):
    all_cls = self.predict_result.get_cls()
    return {'bollard': len(list(np.where(all_cls == 0))[0]),  'bumper': len(list(np.where(all_cls == 1)[0])), 'fender': len(list(np.where(all_cls == 2))[0])}

  def get_predict_number(self):
    return self.predict_number
  
  def get_base_number(self):
    return self.base_boxes.shape[0]

  def save_json(self, save_path):
    output = {'Port_result':{
              'base_info': self.base_info,
              'H': self.H,
              'base_boxes': self.base_boxes,
              'isExist': self.isExist,
              'missing_number': self.missing_number,
              'predict_number': self.predict_number,
            },
            'Img': self.Img._save_json(),
            'Predict_result': self.predict_result._save_json()
            }
    with open(save_path, 'w') as f:
      json.dump(output, f, cls=NumpyArrayEncoder)
    print('save result in json!')


  @classmethod
  def load_from_json(self, j):
    with open(j, 'r') as f:
      j = json.load(f)
    base_info = np.array(j['Port_result']['base_info'])
    H = np.array(j['Port_result']['H'])
    base_boxes = np.array(j['Port_result']['base_boxes'])
    isExist = np.array(j['Port_result']['isExist'])
    missing_number = np.array(j['Port_result']['missing_number'])
    predict_number = np.array(j['Port_result']['predict_number'])
    img = Img._load_from_json(j['Img'])
    predict_result = Predict_result._load_from_json(j['Predict_result'])
    port_result = Port_Result(img, predict_result, base_info, H=H, base_boxes=base_boxes, isExist=isExist, missing_number=missing_number, predict_number=predict_number)
    return port_result

