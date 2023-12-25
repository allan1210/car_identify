import cv2
import numpy as np
import easyocr
from pymongo import MongoClient
import time

# 設定 EasyOCR 的語言和模型
reader = easyocr.Reader(['en'], gpu=True)  # 如果有 GPU，啟用 GPU

# 連接到 MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['車位資料庫']
collection = db['車位']

# 初始化總空位數
total_spaces = 3  # 假設總共有三個停車位

def process_parking_spaces(frame, parking_spaces):
    global total_spaces  # 使用全局變數

    for space_name, space_info in parking_spaces.items():
        # 創建掩膜，只保留停車位區域
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [space_info["points"]], (255, 255, 255))
        result = cv2.bitwise_and(frame, mask)

        # 將停車位區域轉換為灰階
        roi_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # 使用高斯模糊進行降噪
        roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

        # 使用 Canny 邊緣檢測
        roi_edged = cv2.Canny(roi_blurred, 30, 150)

        # 尋找輪廓
        roi_contours, _ = cv2.findContours(roi_edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 100
        max_area = 1000
        min_aspect_ratio = 1.5
        max_aspect_ratio = 3.0

        # 初始化停車位狀態
        status = "Vacant"  # 停車位預設為空閑

        # 初始化文字結果
        text = ""

        # 進行車牌辨識
        for roi_contour in roi_contours:
            # 計算輪廓面積
            area = cv2.contourArea(roi_contour)

            # 計算包圍矩形
            x, y, w, h = cv2.boundingRect(roi_contour)

            # 計算寬高比
            aspect_ratio = float(w) / h

            # 篩選輪廓
            if min_area < area < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                # 進行多邊形逼近
                epsilon = 0.02 * cv2.arcLength(roi_contour, True)
                approx = cv2.approxPolyDP(roi_contour, epsilon, True)

                # 使用形狀擬合確保是矩形
                if len(approx) == 4:
                    # 繪製車牌區域的矩形
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                    # 提取車牌區域
                    plate_roi = frame[y:y + h, x:x + w]

                    # 將車牌區域轉換為灰階
                    plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

                    # 使用 EasyOCR 進行文字辨識
                    results = reader.readtext(plate_gray)

                    if results:
                        text = results[0][1]
                        status = "Occupied"  # 如果有偵測到車牌，表示停車位被占用
                        total_spaces -= 1  # 每有一個車位被占用就減一
                        break  # 停止迴圈，只取一個車牌

        # 將停車位狀態和文字結果顯示在視窗左上角
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 檢查車位狀態，如果為 Occupied，將文字顏色設定為紅色，否則為綠色
        text_color = (0, 0, 255) if status == "Occupied" else (0, 255, 0)

        cv2.putText(frame, f"{space_name} Status: {status} Plate: {text}", (10, 30 * int(space_name[-1])), font, 0.5, text_color, 2)

        # 在控制台輸出結果
        print(f"{space_name} Status: {status} Plate: {text}")

        # 將資料寫入 MongoDB
        data = {
            'space_name': space_name,
            'status': status,
            'plate': text
        }
        collection.insert_one(data)

    # 在視窗右上角顯示剩餘空位數
    cv2.putText(frame, f"Vacant Spaces: {total_spaces}", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    # 顯示即時影像
    cv2.imshow('Real-time Parking - Image', frame)

    # 在控制台輸出剩餘空位數
    print(f"Vacant Spaces: {total_spaces}")

# 使用攝影機捕獲即時影像
cap = cv2.VideoCapture(0)  # 0 表示默認攝影機，可以更改為其他數字，例如1，依據實際情況

A1_left_top = (68, 287)
A1_left_bottom = (2, 385)
A1_right_top = (319, 311)
A1_right_bottom = (238, 389)

A2_left_top = (320, 301)
A2_left_bottom = (241, 387)
A2_right_top = (495, 304)
A2_right_bottom = (569, 389)

A3_left_top = (510, 311)
A3_left_bottom = (571, 383)
A3_right_top = (729, 284)
A3_right_bottom = (798, 386)

# 停車位資訊
parking_spaces = {
    "A1": {"points": np.array([A1_left_top, A1_left_bottom, A1_right_bottom, A1_right_top], dtype=np.int32)},
    "A2": {"points": np.array([A2_left_top, A2_left_bottom, A2_right_bottom, A2_right_top], dtype=np.int32)},
    "A3": {"points": np.array([A3_left_top, A3_left_bottom, A3_right_bottom, A3_right_top], dtype=np.int32)},
}

# 主迴圈處理即時影像
while True:
    # 讀取當前幀
    ret, frame = cap.read()

    # 如果讀取失敗，則退出迴圈
    if not ret:
        break

    # 在這裡添加您的車位檢測和顯示邏輯
    process_parking_spaces(frame, parking_spaces)

    # 等待 30 秒
    for _ in range(10):
        key = cv2.waitKey(1000) & 0xFF
        if key == ord('q'):
            break

    # 如果按下 'q' 鍵，則退出迴圈
    if key == ord('q'):
        break


# 釋放資源
cap.release()
cv2.destroyAllWindows()

# 關閉 MongoDB 連接
client.close()
