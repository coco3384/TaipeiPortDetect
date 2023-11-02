from utils.Port_result import Port_Result
import os
import glob
import cv2

if __name__ == '__main__':
    result_json = glob.glob(os.path.join('20230908', 'poi3_23', '*.json'))[0]
    result = Port_Result.load_from_json(result_json)
    print(result.get_base_number())
    cv2.imshow('all', result.show_all())
    cv2.imshow('missing', result.show_missing())
    cv2.imshow('find', result.show_find())
    cv2.imshow('predict result', result.show_result())
    cv2.waitKey(0)