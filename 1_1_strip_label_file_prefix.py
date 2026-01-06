import os
import argparse

def remove_label_prefix(label_dir):
    # 檢查路徑是否存在
    if not os.path.isdir(label_dir):
        print(f"錯誤: 找不到目錄 '{label_dir}'")
        return

    print(f"正在處理目錄: {label_dir}")
    count = 0
    
    for filename in os.listdir(label_dir):
        # 檢查是否為 .txt 檔且包含 Label Studio 產生的 "-" 前綴
        if filename.endswith(".txt") and "-" in filename:
            # 找到第一個 "-" 的位置，並保留之後的部分
            # 例如 "00ba02e0-imageXXX.txt" -> "imageXXX.txt"
            new_name = filename.split("-", 1)[1]
            
            old_path = os.path.join(label_dir, filename)
            new_path = os.path.join(label_dir, new_name)
            
            # 執行重新命名
            try:
                os.rename(old_path, new_path)
                print(f"成功: {filename} -> {new_name}")
                count += 1
            except Exception as e:
                print(f"跳過: 無法重新命名 {filename}，原因: {e}")

    print(f"--- 處理完成，共修改了 {count} 個檔案 ---")

if __name__ == "__main__":
    # 設定命令列參數解析器
    parser = argparse.ArgumentParser(description="移除 Label Studio 導出檔案的 UUID 前綴")
    
    # 加入 path 參數
    parser.add_argument("path", type=str, help="標註檔 (.txt) 所在的資料夾路徑")
    
    # 解析參數
    args = parser.parse_args()
    
    # 執行函數
    remove_label_prefix(args.path)