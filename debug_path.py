import os

print("=== 診斷報告 ===")
print(f"1. 目前終端機的工作位置: {os.getcwd()}")

target_folder = 'weather_data'
if os.path.exists(target_folder):
    print(f"2. ✅ 找到 '{target_folder}' 資料夾了！")
    
    # 檢查子資料夾
    subdirs = [f.path for f in os.scandir(target_folder) if f.is_dir()]
    if subdirs:
        print(f"3. ✅ 在裡面發現 {len(subdirs)} 個測站資料夾：")
        for s in subdirs:
            print(f"   - 📁 {s}")
            # 檢查裡面的檔案
            files = os.listdir(s)
            csv_count = len([f for f in files if f.endswith('.csv')])
            print(f"     └─ 內含 {csv_count} 個 CSV 檔案")
            
            if csv_count < 6:
                print("     ❌ 警告：CSV 檔案不足 6 個，或是檔名不符合規則！")
    else:
        print("3. ❌ 錯誤：'weather_data' 裡面是空的！")
        print("   👉 請在裡面建立一個資料夾 (例如 'G2F820_霧峰')，然後把 CSV 放進去。")
        print("   ⚠️ 注意：不要把 CSV 直接放在 weather_data 根目錄下！")
else:
    print(f"2. ❌ 找不到 '{target_folder}' 資料夾！")
    print("   👉 請確認您是否有建立這個資料夾，或者您是否在正確的層級執行程式？")

print("==================")
