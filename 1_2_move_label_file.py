import os
import shutil
import argparse
from pathlib import Path

def sync_labels(path_img, path_txt_src, path_txt_dst):
    # 檢查路徑是否存在
    for p in [path_img, path_txt_src]:
        if not os.path.isdir(p):
            print(f"錯誤: 找不到目錄 '{p}'")
            return

    # 建立目標資料夾（如果不存在）
    os.makedirs(path_txt_dst, exist_ok=True)

    # 取得圖片目錄下所有檔案的主檔名 (不含副檔名)
    # 支援常見圖片格式
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    img_names = {Path(f).stem for f in os.listdir(path_img) if f.lower().endswith(img_extensions)}

    print(f"在圖片目錄中找到 {len(img_names)} 個有效影像檔。")
    
    count = 0
    not_found = 0

    for name in img_names:
        target_txt = f"{name}.txt"
        src_file = os.path.join(path_txt_src, target_txt)
        dst_file = os.path.join(path_txt_dst, target_txt)

        # 檢查來源標註檔是否存在
        if os.path.isfile(src_file):
            shutil.move(src_file, dst_file)
            print(f"已移動: {target_txt} -> {path_txt_dst}")
            count += 1
        else:
            # 如果這張圖沒有對應的標註檔，通常代表它是負樣本
            not_found += 1

    print(f"\n--- 處理完成 ---")
    print(f"成功移動: {count} 個標註檔")
    print(f"未找到對應標註 (可能是背景圖): {not_found} 個")
    print(f"標註檔現在位於: {path_txt_dst}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根據圖片清單，從來源目錄選取對應的標註檔並移動到目標目錄")
    
    parser.add_argument("path_img", type=str, help="參考圖片所在的資料夾路徑")
    parser.add_argument("path_txt_src", type=str, help="所有標註檔 (.txt) 的來源路徑")
    parser.add_argument("path_txt_dst", type=str, help="配對成功的標註檔要存放的目標路徑")
    
    args = parser.parse_args()
    
    sync_labels(args.path_img, args.path_txt_src, args.path_txt_dst)