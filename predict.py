import os
import cv2
import glob
import math
import pandas
import numpy as np
import torch
from utils.Img import Img
from utils.Port_result import Port_Result
from utils.Predict_result import Predict_result
from utils.utils import *
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
from pyproj import Transformer
from ultralytics import YOLO
from SETTINGS import *


def load_poi_points(poi_path):
    # 全部的POI一起讀會有一些問題在，但若要分開的話可能得是使用者端要自行分好巡檢照片所對應的POI路線
    # 因為並非每次的拍攝都是依照1 2 3的順序去拍的，有時候他們是3 2 1拍回來，所以也沒辦法直接用資料夾名稱順序來判斷
    poi1_points = read_poi_csv(os.path.join(poi_path, 'POI1.csv'))
    poi2_points = read_poi_csv(os.path.join(poi_path, 'POI2.csv'))
    poi3_points = read_poi_csv(os.path.join(poi_path, 'POI3.csv'))
    return {'poi1': poi1_points, 'poi2': poi2_points, 'poi3':poi3_points}

def load_single_strip_of_poi_points(poi_path):
    # 如果要讀取單一一條poi的路線的話可以用這個讀
    poi_points = read_poi_csv(poi_path)
    poi_number = os.path.basename(poi_path).split('.')[0].lower()
    return {poi_number: poi_points}


def predict(poi_name, poi_point, position_dict, n, model, base_dir):
    # 預測POI點對應的新期影像
    # 讀取基期資料 > 找尋最近的五(n=5)張影像 > YOLOv8預測五張照片 > 結果轉換成自定義的Port_Result Class > 合併五張結果
    # Port_Result Class：
    #   詳細可看uitls/Port_Reulst.py檔中的說明。
    #   輸出方面的funciton有：
    #       基本資訊：
    #           get_missing_number()              : 回傳dict，各物件種類缺失的數量。
    #           get_predict_number()              : 回傳dict，各物件偵測到的數量。
    #       結果展示：
    #           Port_Result.show_find()     : 回傳cv2 image(numpy array)，Img原圖與基期物件中有找到對應物件的物件（綠色框）
    #           Port_Result.show_missing()  : 回傳cv2 image(numpy array)，Img原圖與基期物件中未找到對應物件的物件（紅色框）
    #           Port_Result.show_result()   : 回傳cv2 image(numpy array)，Img原圖與這次的偵測結果，以點標示。
    #           Port_Result.show_all()      : 回傳cv2 image(numpy array)，Img原圖與上述三種資訊的結果。
    #       存取與讀取：
    #           save_json(file_path)              : 存成json file。
    #           load_from_json(file_path)         : 回傳Port_Result Class，讀依照save_json格式儲存的json file。

    # 使用CPU或是GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 讀取基期資料
    base_json = glob.glob(os.path.join(base_dir, f'{poi_name}', '*.json'))[0]
    with open(base_json, 'r') as f:
        base_info = json.load(f)
    # 找最近的n張照片
    nearest_images = find_nearest(n, poi_point, position_dict)
    # 依照等名排序，中間會是最近的那張，理論上的POI影像
    nearest_images.sort()
    # YOLOv8預測五張照片
    results = model.predict(nearest_images, conf=0.5, device=device, verbose=False)

    # Yolov8 Results -> Prediction_result Classes
    predictions = []
    for i in range(len(nearest_images)):
        # img_sz, use 1280 to detect，Yolov8預設使用1280進行預測，同時也是偵測效果最好的影像大小。
        img_sz = refine_shape_value(results[i].orig_shape, 1280)
        # box
        boxes = results[i].boxes.xyxyn.cpu().numpy()
        # volume
        wh = results[i].boxes.xywhn[:, 2:].cpu().numpy()
        volumes = box_volume(wh)
        # cls
        cls = results[i].boxes.cls.cpu().numpy()
        # conf
        conf = results[i].boxes.conf.cpu().numpy()
        # names
        names = results[i].names

        # Img Class
        img = Img(nearest_images[i])
        # Predict_result Class
        predict_result = Predict_result(boxes=boxes, img_sz=img_sz, volumes=volumes, cls=cls, conf=conf, names=names)
        # 匯成 Port_Result Class
        Result = Port_Result(img, predict_result, base_info)
        predictions.append(Result)

    # 找到POI照片（最近的）
    base = int(len(nearest_images)/2)
    image_name = os.path.basename(nearest_images[base]).split('.')[0]
    # 合併其他張的結果到POI影像
    for i in range(len(nearest_images)):
        if i != base:
            predictions[base].combine_result(predictions[i])

    return predictions[base], image_name
    

if __name__ == '__main__':
    # sample image list
    image_list = []
    for dir in os.listdir(NEW_IMAGE_DIR_PATH):
        image_list = image_list + glob.glob(os.path.join(NEW_IMAGE_DIR_PATH, dir, '*.JPG'))

    # 獲取影像的ＧＰＳ位置
    position_dict = get_position_dict(image_list)
    # 獲取POI的ＧＰＳ位置
    poi_points = load_poi_points(POI_PATH)
    # 每張照片的結果是 n 張照片的合併結果
    n = N
    # Load Model
    model = YOLO(MODEL)
    # 基期影像位置
    base_dir = BASE_PERIOD_DIR
    # 創建輸出預測結果的資料夾
    os.makedirs(NEW_PERIOD_DIR, exist_ok=True)

    # 依照ＰＯＩ點依序讀取位置、讀取對應的基期影像、進行預測。
    for i, (key, val) in enumerate(zip(poi_points.keys(), poi_points.values())):
        for poi_idx, poi_point in enumerate(val):
            poi_name = f'{key}_{poi_idx}'
            result, image_name = predict(poi_name, poi_point, position_dict, n, model, base_dir)
            # save
            save_dir = os.path.join(NEW_PERIOD_DIR, poi_name)
            os.makedirs(save_dir, exist_ok=True)
            result.save_json(os.path.join(save_dir, image_name+'.json'))