import cv2
import os
import time

RUN_IN_WSL = False  # 如果在 WSL 環境下運行，請設置為 True
# 也可以用下面判斷
# RUN_IN_WSL = (platform.system() == 'Linux' and 'microsoft' in platform.release().lower())

def collect_data(output_folder='my_dataset/train/images'):
    # 1. 建立儲存資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"建立資料夾: {output_folder}")

    # 2. 打開相機 (0 通常是預設鏡頭)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("錯誤: 無法打開相機")
        return

    # 關鍵設定 for WSL：將格式設為 MJPG (降低頻寬需求，增加相容性)
    if RUN_IN_WSL:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
    print("--- 拍照模式開啟 ---")
    print("按 'S' 鍵：拍照並存檔 (Save)")
    print("按 'Q' 鍵：退出程式 (Quit)")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取影像")
            break

        # 為了不把文字也拍進去，我們在顯示用的 frame 上畫圖，存檔則用原始 frame
        display_frame = frame.copy()
        
        # 在預覽畫面上顯示已收集數量 (左上角綠色文字)
        status_text = f"Collected: {count}"
        cv2.putText(display_frame, status_text, (20, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
        
        # 顯示操作提示 (左下角)
        cv2.putText(display_frame, "S: Save | Q: Quit", (20, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('Data Collector', display_frame)

        # 偵測按鍵
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # 檔名包含時間戳記與序號，確保唯一性
            timestamp = int(time.time())
            filename = f"img_{timestamp}_{count}.jpg"
            filepath = os.path.join(output_folder, filename)
            
            # 儲存原始影像 (不含文字)
            cv2.imwrite(filepath, frame)
            count += 1
            print(f"[{count}] 已儲存: {filepath}")

        elif key == ord('q'):
            print("停止收集。")
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
    print(f"--- 任務完成 ---")
    print(f"總共收集了 {count} 張照片，存放在: {output_folder}")

if __name__ == "__main__":
    # 你可以自定義資料夾名稱
    collect_data('my_dataset/train/images')