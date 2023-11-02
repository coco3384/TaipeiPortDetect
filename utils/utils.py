from PIL import Image
from PIL.ExifTags import TAGS
from json import JSONEncoder
from pyproj import Transformer
import math
import os
import glob
import pandas as pd
import json
import numpy as np


def _dis(value1, value2):
    return math.sqrt((value1[0] - value2[0])**2 + (value1[1] - value2[1])**2)

def _convert_to_degress(lat, lon):
    """Helper function to convert the GPS coordinates stored in the EXIF to degress in float format"""
    return (lat[0] + (lat[1] / 60.0) + (lat[2] / 3600.0), lon[0] + (lon[1] / 60.0) + (lon[2] / 3600.0))

def _convert_to_TWD97(value):
    wgs84 = _convert_to_degress(value[2], value[4])
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3826")
    twd97 = transformer.transform(wgs84[0], wgs84[1])
    return twd97

def read_poi_csv(poi_csv):
    poi_point = pd.read_csv(poi_csv)
    lon = poi_point['POINT_X'].to_numpy()
    lon = np.expand_dims(lon, axis=1)
    lat = poi_point['POINT_Y'].to_numpy()
    lat = np.expand_dims(lat, axis=1)
    poi_points = np.concatenate([lon, lat], axis=1)
    return poi_points


def find_nearest(n, poi, position_dict):
    image2poi_distances = []
    for position in position_dict.values():
        image2poi_distances.append(_dis(position, poi))
    output = []
    nearst = np.argsort(image2poi_distances, kind='stable')[:n]
    return [name for i, name in enumerate(position_dict.keys()) if i in nearst]


def box_center(box, img_sz):
  box = box * (img_sz[0], img_sz[1], img_sz[0], img_sz[1])
  return np.array((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))

def homo_trans(point, H):
  position = np.expand_dims(point, axis=1)
  pos_homo = np.vstack((position, [1]))
  trans_homo = np.matmul(H, pos_homo).squeeze()
  trans = (trans_homo / trans_homo[-1])[:2]
  return trans.astype(int)

def box_volume(wh):
  volumes = []
  for info in wh:
    volumes.append(info[0] * info[1])
  volumes = np.expand_dims(volumes, axis=1)
  return volumes

def refine_shape_value(orig_shape, pred_shape):
  sf = min(pred_shape/orig_shape[0], pred_shape/orig_shape[1])
  return np.array((sf * orig_shape[0], sf * orig_shape[1])).astype(int)


def get_position_dict(img_list):
    position_dict = {}
    for image in img_list[:]:
        img = Image.open(image)
        exif = img._getexif()
        if exif is not None:
            gps_Info = {}
            for (tag,value) in exif.items():
                key = TAGS.get(tag, tag)
                if key == 'GPSInfo':
                    gps_info = value
        lon, lat = _convert_to_TWD97(gps_info)
        position_dict[image] = np.array([lon, lat])
    return position_dict


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)