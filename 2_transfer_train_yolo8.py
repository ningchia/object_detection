from ultralytics import YOLO
import torch

# 如果dataset被安排成:
# my_dataset/
# ├── train/    
# │   ├── images/
# │   └── labels/*.txt (YOLO格式標註檔, 可以用label-studio或Roboflow產生)
# └── val/
#     ├── images/
#     └── labels/
#
# 假設 1_collect_dataset.py的目標path是 my_dataset/train/images, 最終全部的影像會出現在 my_dataset/train/images 裡.
# 挑選約10~20%的影像改放到 my_dataset/val/images 裡作為驗證集.
# 之後開始用 label-studio 或 Roboflow 做標註，標註結果放到對應的 labels/ 資料夾裡.
# 假設針對 my_dataset/train/images 做完標註後, export出來的結果像這樣,
#   classes.txt  (裡面有三個類別名稱, 分別是 cup, hand, phone. data.yaml 裡的 names 要跟這邊對應)
#   notes.json   (用不到，可以忽略)
#   images/ (空目錄)
#   labels/*.txt  (裡面是標註結果, YOLO格式)
# 
# 直接把 labels/ 放到train/ 下即可(變成 train/labels/)。 val/ 也一樣處理.
# 然後要準備一個 .yaml 檔案，用來在model.train()時告訴模型圖片在哪裡、有幾個類別。
# 假設是data.yaml, 可以寫成:
#   --------------------------------------------------
#   path: ./my_dataset              # 資料集根目錄
#   train: train/images
#   val: val/images
#   nc: 3                            # 類別數量
#   names: ['cup', 'hand', 'phone']  # 類別名稱. 須嚴格遵守label-studio產出的class.txt裡的順序
#   --------------------------------------------------

def train_yolo():
    # 1. 檢查是否有 GPU，有的話使用 GPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"正在使用設備: {device}")

    # 2. 載入預訓練模型
    # YOLOv8n (Nano): 速度最快，適合手機/嵌入式
    # YOLOv8s (Small): 速度與精度的平衡，建議從這裡開始
    model = YOLO('yolov8s.pt') 

    # 3. 開始訓練
    # data: 指向你的 yaml 檔案
    # epochs: 訓練輪次，初次測試可以設 50-100
    # imgsz: 圖片輸入大小，通常為 640
    # batch: 批次大小，視顯存大小調整 (如 16, 32)
    results = model.train(
        data='data.yaml',           # 告訴模型你的圖片在哪裡、有幾個類別
        epochs=100,      
        imgsz=640,       
        batch=16,        
        device=device,              # 使用 GPU 或 CPU
        name='cup_hand_phone_model' # 儲存結果的資料夾名稱
    )

    # 4. 訓練結束後驗證模型
    metrics = model.val()
    print(f"訓練完成，Map50-95: {metrics.box.map}")

    # 5. 導出模型 (例如轉成 ONNX 格式方便部署)
    # model.export(format='onnx')

if __name__ == '__main__':
    train_yolo()