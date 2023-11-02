##### 目前與基期的比對高機率出錯，因為兩個之間的投影轉換做起來很怪...

> 在poi的邊邊更容易出錯，因為會有另一角度的影像混進『最近照片』。

##### 但最後出去的資料格式是這樣已經確定了，可以先串接平台的顯示。

### 比較成功的例子有```20230908/POI3_23```的這個點

### 如果要測試效果的話建議先使用這個。

____

## 說明

* 執行檔案 > ```predict.py```
* 要跑新期的影像 > 修改```SETTING.py```
* 因為此次加入了多張影像的偵測結果合併，所以預測的時候必須是整批一起跑，才能把巡檢影像中相近的影像結果合併再一起。

____

* **load.py**
* **create_base_poi_json.py**：把基期```.txt```的結果轉成```.json```
* **predict.py**：預測執行檔
* **SETTINGS.py**：```predict.py```所需的參數

<img src="/Users/user/Library/Application Support/typora-user-images/截圖 2023-11-02 上午7.46.18.png" alt="截圖 2023-11-02 上午7.46.18" style="zoom:50%;" />

#### POI 資料夾格式

```
POI
│   POI1.csv
│   POI2.csv
│   POI3.csv
```

#### NEW_IMAGE_DIR_PATH 資料夾格式

```
20230908岸邊設施(新期影像)
	└── xxxFTASK
	│   │   xxx.JPG
	│   │   ...
	└── xxxFTASK
	│   │   xxx.JPG
	│   │   ...
	└── xxxFTASK
	│   │   xxx.JPG
	│   │   ...
```

要直接跑```predict.py```的話新期影像資料夾須是上面的格式。

若後台的資料格式不是那樣的話，稍微修改下面```predict.py```第三行的```image_list```就好，需要一個有整期影像路徑的```list```。

<img src="/Users/user/Desktop/截圖 2023-11-02 上午8.16.58.png" alt="截圖 2023-11-02 上午8.16.58" style="zoom:50%;" />

#### BASE_PERIOD_DIR 資料夾格式

```
base_period
	└── poi1_0
	│   │   xxx(base_image).JPG
	│   │   xxx(label).txt
	│   │   xxx(label).json
	└── poi1_1
	│   │   ...
	└── poi1_2
	└── ...
	└── poi2_0
	└── ...
	└── poi3_0
	└── ...
```

> poi1_0 對應 poi1.csv 中第一個poi位置

如果有更新基期的需求的話，更換```xxx.JPG```和```xxx.txt```，```xxx.txt```是yolo的label格式。再跑```create_base_poi_json.py```把```.txt```轉換成```.json```。

##### ```base.json```格式

```json
{
  "img": "base_period/poi1_1/MAX_0306.JPG", # image_path
  "ori_wh": [7680, 4320],                   # 原始影像大小
  "label": {
    "boxes": [                              # 2-D list
      [2758, 1015, 2809, 1072],             # b-box xyxy
      ...
    ],
    "clses": [
      0,                                    # cls
      ...                                   # 0: 繫船柱
    ]                                       # 1: 車擋
  }                                         # 2: 碰墊
}
```

#### NEW_PERIOD_DIR

```
20230908(save_name)
└── poi1_0
│   │   xxx(predict_result).json
└── poi1_1
└── ...
```

以與```base_period```相同的資料夾格式儲存結果。

##### ```predict_result.json```格式

```json
{
  "Port_result": {
    "base_info": {
      "img": "base_period/poi1_20/MAX_0331.JPG", #基期影像位置
      "ori_wh": [7680, 4320], # 基期影像大小
      "label": {
        "boxes": [            # 基期影像物件的原始b-box位置
          [15, 1396, 197, 1545],
          ...
         ],
      "clses": [              # 基期影像的物件類別
      		0,                  # 0: 繫船柱 1: 車擋 2: 碰墊
      		...                                  
    		]                                    
			"H": # 基期影像轉到新期影像的homography transformation 
				[
          [ -0.355, -0.430, 125.951],
      		[ -0.575, -1.980, 683.052],
      		[ -0.001, -0.002, 1.000  ],
    		],
    	"base_boxes": [            # 基期影像的b-box轉到新期後的位置
      		[ -67, 548, 33142, 14803], # 會有超出圖片的位置
      		...
     		],
    	"isExist": [                 # 基期物件是否在新期有被找到
      		1,                       # 1代表有, 0代表沒有
      		...
      	],
			"missing_number": 51,        # 缺失物件數量
      "predict_number": {          # 新期影像找到的物件數量
            "bollard": 0,
            "bumper": 15,
            "fender": 6
       	}
			},
	"Img": {
    	"image":"20230908設施/11FTASK/MAX_442.JPG"# 新期影像位置
  	},
	"Predict_result": {
    	"boxes": [                    # 新期影像物件b-box位置
      	[0.279, 0.567, 0.305, 0.597],  # xyxy的normalize
     		...
     	],
    	"cls": [
      	2,                          # 新期影像的物件類別
        ...                         # 0: 繫船柱 1: 車擋 2: 碰墊
       ],
    	"conf": [
      	0.9081075191497803,         # 新期影像物件的信心度
        ...
      ],
      "names": {                    # cls對應的物件名稱
        "0": "bollard",
        "1": "bumper",
        "2": "fender"
    	},
      "volumes": [                  # 物件的體積(normalize)
        [0.000783876224886626]
        ...
      ],
      "img_sz": [960, 1280]         # 新期影像預測時的大小
  	}
      
```

___

## 程式說明

#### ```SETTINGS.py```

<img src="/Users/user/Library/Application Support/typora-user-images/截圖 2023-11-02 上午7.46.18.png" alt="截圖 2023-11-02 上午7.46.18" style="zoom:50%;" />

#### ```predict.py```

###### 整批影像的```list``` > 找影像的GPS位置 > 所有POI的GPS位置 > ```loop```每個POI點去找POI照片和預測POI照片。

<img src="/Users/user/Desktop/截圖 2023-11-02 上午8.16.58.png" alt="截圖 2023-11-02 上午8.16.58" style="zoom:50%;" />

所有需要的參數：

1. ```python
   1. position_dict = {str(image_path): np.array([lon, lat])}
   2. poi_points = {'poi1': 2-D np.array, 'poi2', 2-D np.array, 'poi3', 2-D np.array}
   # poi_points也可以是單獨一條poi line，但得是dict：{'poi1', 2-D np.array}。
   3. n = int(N)
   4. model = YOLO(path/to/model)
   5. base_dir = str(path/to/base_period)
   ```

運行預測的部分是```function predict()```

```python
def predict(poi_name, poi_point, position_dict, n, model, base_dir):
```

* ```poi_name```：對應的 POI 的點名稱，為了從```base_period```資料夾中找到對應的基期資訊```.json```。
* ```poi_point```：單一 POI 點的GPS位置```(lon, lat)```。
* ```position_dict```：所有照片的GPS位置```dict{"image_path": str, "gps_info": np.array[lon, lat]}```
* ```n```：會合併幾張照片的結果
* ```model```：Yolov8的模型
* ```base_dir```：基期的資料夾位置

![截圖 2023-11-02 上午9.52.08](/Users/user/Library/Application Support/typora-user-images/截圖 2023-11-02 上午9.52.08.png)

![截圖 2023-11-02 上午9.53.56](/Users/user/Desktop/截圖 2023-11-02 上午9.53.56.png)

![截圖 2023-11-02 上午10.01.49](/Users/user/Desktop/截圖 2023-11-02 上午10.01.49.png)

###### 讀取基期資訊 > 找最近的照片 > 依檔名排序 > 模型預測 > 預測結果構成自定義Class > 合併預測結果

##### 三個自定義的Class

```python
class Port_Result
class Predict_result
class Img
```

每張照片偵測完後會存成```Port_Result```，再調用```Class```的```合併function```進行合併。

* ```Port_Result```

  * ```.to_json(save_path)```轉換成```.json```，最後儲存的格式。
  * ```.load_from_json(json)```讀取轉換出來的```json```檔案。

  因為預測跑一張大約要5-10秒左右(cpu)，所以應該會是使用者上傳整批影像後先全部偵測過儲存成```json```格式，之後的顯示再讀取```json```來獲取資訊。

  * ```Function```

    * ```python
      get_missing_number(): 回傳dict, 各物件缺失的數量
      ```

    * ```python
      get_predict_number(): 回傳dict，各物件偵測到的數量。
      ```

    * ```python
      Port_Result.show_find():
      # 回傳cv2 image(numpy array)
      # 新期影像與基期物件中有找到對應物件的位置（綠色框）
      ```

    * ```python
      Port_Result.show_missing(): 
      # 回傳cv2 image(numpy array)
      # 新期影像與基期物件中未找到對應物件的物件（紅色框）
      ```

    * ```python
      Port_Result.show_result():
      # 回傳cv2 image(numpy array)
      # 新期影像與這次的偵測結果，以點標示。
      # 藍色點：碰墊, 綠色點：車擋, 紅色點：繫船柱
      ```

    * ```python
      Port_Result.show_all():
      # 回傳cv2 image(numpy array)
      # 新期影像與上述三種資訊的結果。
      ```

* ```Predict_result```

  * 詳細可看```Predict_result.py```的註解

* ```Img```

  * 詳細可看```Img.py```的註解