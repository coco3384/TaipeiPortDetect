import numpy as np
from .utils import box_center, homo_trans


"""
self.boxes      : numpy array
self.cls        : numpy array
self.conf       : numpy array
self.names      : dict
self.volumes    : numpy array
self.img_sz     : numpy array 
"""
class Predict_result:
    def __init__(self, boxes, volumes, cls, conf, names, img_sz):
        self.boxes = boxes # numpy array
        self.cls = cls # numpy array
        self.conf = conf # numpy array
        self.names = names # dictionary
        self.volumes = volumes # numpy array
        self.img_sz = img_sz # numpy array

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        return {'cls': self.cls[idx], 'conf': self.conf[idx], 'box': self.boxes[idx], 'volumes': self.volumes[idx] * self.img_sz[0] * self.img_sz[1]}

    def _save_json(self):
      return {'boxes': self.boxes, 'cls': self.cls, 'conf': self.conf, 'names': self.names, 'volumes': self.volumes, 'img_sz': self.img_sz}

    @classmethod
    def _load_from_json(self, j):
      boxes = np.array(j['boxes'])
      cls = np.array(j['cls'])
      conf = np.array(j['conf'])
      names = j['names']
      volumes = np.array(j['volumes'])
      img_sz = np.array(j['img_sz'])
      return Predict_result(boxes=boxes, volumes=volumes, cls=cls, conf=conf, names=names, img_sz=img_sz)

    def _resize_box_to_compare(self, base_boxes, base_imgsz):
      sf = self.img_sz / base_imgsz
      boxes = np.array(list((box * np.array((sf[1], sf[0], sf[1], sf[0])) for box in base_boxes))).astype(int)
      return boxes

    def _reproject_base_boxes(self, base_boxes, H):
      boxes = []
      for box in base_boxes:
        x1y1 = box[:2]
        x2y2 = box[2:]
        x1y1_trans = homo_trans(x1y1, H)
        x2y2_trans = homo_trans(x2y2, H)
        box = np.concatenate((x1y1_trans, x2y2_trans), axis=0)
        boxes.append(box)
      return boxes


    def _in_box(self, box_center, box, buffer=5):
      x = box_center[0]
      y = box_center[1]
      if (x > (box[0] - buffer) and x < (box[2] + buffer)):
        if (y > (box[1] - buffer) and y < (box[3] + buffer)):
          return True
      else:
        return False

    def _is_in_base_period(self, box_center, base_boxes):
      for i, base_box in enumerate(base_boxes):
        if self._in_box(box_center, base_box):
          return i, True
      return -1, False

    def _is_damaged_object(self, volume):
      pass

    def compare_result_with_base(self, base_info, H):
      # imgsz
      base_imgsz = base_info['ori_wh']
      base_imgsz = [base_imgsz[1], base_imgsz[0]]
      # boxes
      base_boxes = base_info['label']['boxes']
      base_boxes = self._resize_box_to_compare(base_info['label']['boxes'], base_imgsz)
      isExist = np.zeros(len(base_boxes))
      if H is not None:
        base_boxes_trans = self._reproject_base_boxes(base_boxes, H)
        # clses
        base_clses = base_info['label']['clses']

        for i, (cls, box_center) in enumerate(zip(self.get_cls(), self.get_boxes_center())):
          # filter base_object by same cls
          idx, is_base_period_object = self._is_in_base_period(box_center, base_boxes_trans)
          if is_base_period_object:
            isExist[idx] = True
            if self._is_damaged_object(self.volumes[i]):
              # Not Yet implement
              pass
        return base_boxes_trans, isExist
      
      else:
        return base_boxes, isExist


    def get_boxes_n(self):
      return self.boxes

    def get_boxes_rs(self):
      boxes = []
      for box in self.get_boxes_n():
        boxes.append(box * (self.img_sz[1], self.img_sz[0], self.img_sz[1], self.img_sz[0]))
      return np.array(boxes)

    def get_boxes_center(self):
      center = []
      for box in self.get_boxes_n():
        center.append(box_center(box, (self.img_sz[1], self.img_sz[0])))
      return center

    def reproject_boxes(self, H):
      boxes = []
      for box in self.get_boxes_rs():
        x1y1 = box[:2]
        x2y2 = box[2:]
        x1y1_trans = homo_trans(x1y1, H)
        x2y2_trans = homo_trans(x2y2, H)
        box = np.concatenate((x1y1_trans, x2y2_trans), axis=0)
        box = box / (self.img_sz[1], self.img_sz[0], self.img_sz[1], self.img_sz[0])
        boxes.append(box)
      return Predict_result(boxes=boxes, volumes=self.volumes, cls=self.cls, conf=self.conf, names=self.names, img_sz=self.img_sz)

    def get_cls(self):
      return self.cls

    def get_conf(self):
      return self.conf

    def get_missing(self):
      return self.missing

    def get_volumes(self):
      return self.volumes

    def get_names(self):
      return self.names

    def get_img_sz(self):
      return self.img_sz

    def get_blocked(self):
      return self.blocked

    def get_damaged(self):
      return self.damaged

    def _delete(self, idx):
      self.boxes = np.delete(self.boxes, idx, 0)
      self.cls = np.delete(self.cls, idx, 0)
      self.volumes = np.delete(self.volumes, idx, 0)
      self.conf = np.delete(self.conf, idx, 0)

    def _combine_result(self, Predict_result):
      if len(Predict_result) != 0:
        self.boxes = np.concatenate((self.get_boxes_n(), Predict_result.get_boxes_n()), axis=0)
        self.cls = np.concatenate((self.get_cls(), Predict_result.get_cls()), axis=0)
        self.volumes = np.concatenate((self.get_volumes(), Predict_result.get_volumes()), axis=0)
        self.conf = np.concatenate((self.get_conf(), Predict_result.get_conf()), axis=0)
        self._remove_duplicate()
      else:
        pass

    def _remove_duplicate(self):
      duplicate = np.zeros(len(self))
      boxes_center = self.get_boxes_center()
      duplicate_indices = self._find_duplicate_indices(boxes_center)
      for idx in np.where(duplicate_indices):
        delete_count = 0
        self._delete(idx - delete_count)
        delete_count += 1

    def _find_duplicate_indices(self, boxes_center, buffer=20):
      duplicate = np.zeros(len(self))
      for i in range(len(self)):
        cls = self.cls[i]
        distance = boxes_center[i:] - boxes_center[i]
        distance = np.sqrt(np.sum(distance ** 2, axis=1))
        indices = np.where(distance < buffer)
        if len(indices[0]) != 1:
          duplicate[i + indices[0][1]] = True
      return duplicate