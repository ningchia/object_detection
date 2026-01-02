from ultralytics import YOLO
import torch

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
        data='data.yaml', 
        epochs=100,      
        imgsz=640,       
        batch=16,        
        device=device,
        name='cup_hand_phone_model' # 儲存結果的資料夾名稱
    )

    # 4. 訓練結束後驗證模型
    metrics = model.val()
    print(f"訓練完成，Map50-95: {metrics.box.map}")

    # 5. 導出模型 (例如轉成 ONNX 格式方便部署)
    # model.export(format='onnx')

if __name__ == '__main__':
    train_yolo()