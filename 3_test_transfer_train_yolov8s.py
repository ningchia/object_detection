import cv2
from ultralytics import YOLO
import platform

RUN_IN_WSL = False
# 也可以用下面判斷
# RUN_IN_WSL = (platform.system() == 'Linux' and 'microsoft' in platform.release().lower())

def main():
    # --- 設定區 ---
    model_path = 'runs/detect/cup_hand_phone_model/weights/best.pt'  # 請確認你的權重路徑
    is_wsl = True  # 如果你在 WSL 內執行，請設為 True
    
    # 1. 載入模型
    model = YOLO(model_path)
    
    # 2. 開啟攝影機
    cap = cv2.VideoCapture(0) # 通常 0 是筆電鏡頭或第一個 USB 鏡頭

    if not cap.isOpened():
        print("錯誤: 無法開啟攝影機。")
        return

    if RUN_IN_WSL:
        # WSL 環境下使用 MJPG 格式能顯著提升效能
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("開始偵測... 按下 'q' 鍵停止。")

    # 定義類別顏色 (B, G, R)
    color_map = {
        'cup': (0, 255, 0),    # 綠色
        'hand': (255, 0, 0),   # 藍色
        'phone': (0, 0, 255)   # 紅色
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. 執行推論 (Inference)
        # stream=True 會使用生成器，對即時影像效能較好
        # conf=0.5 代表信心門檻設為 50%
        results = model(frame, stream=True, conf=0.5)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 取得座標 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # 取得類別名稱與信心值
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])

                # 選擇顏色
                color = color_map.get(cls_name, (255, 255, 255))

                # 4. 畫出邊框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # 5. 標註文字 (放在框框上方)
                label = f"{cls_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                # 確保背景色塊不超出畫面
                cv2.rectangle(frame, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 顯示畫面
        cv2.imshow('YOLOv8 Live Detection', frame)

        # 按下 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()