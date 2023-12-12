import cv2
import pytesseract
import numpy as np
import re
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
import time


# 設定 Tesseract OCR 的執行檔路徑
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 連接到 MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['車位資料庫']
collection = db['車位']

# 使用攝像機
cap = cv2.VideoCapture(0)  # 0 表示使用預設攝像機

# 使用實際坐標表示停車位的位置
A1_left_top = (235, 616)
A1_left_bottom = (14, 723)
A1_right_top = (403, 608)
A1_right_bottom = (335, 722)

A2_left_top = (403, 608)
A2_left_bottom = (335, 722)
A2_right_top = (570, 613)
A2_right_bottom = (643, 722)

A3_left_top = (570, 613)
A3_left_bottom = (643, 722)
A3_right_top = (753, 620)
A3_right_bottom = (968, 724)

# 更新停車位字典
parking_spots = {
    "A1": {"coordinates": [A1_left_top, A1_left_bottom, A1_right_top, A1_right_bottom], "status": "Vacant", "plate": ""},
    "A2": {"coordinates": [A2_left_top, A2_left_bottom, A2_right_top, A2_right_bottom], "status": "Vacant", "plate": ""},
    "A3": {"coordinates": [A3_left_top, A3_left_bottom, A3_right_top, A3_right_bottom], "status": "Vacant", "plate": ""},
}

while True:
    # 讀取一幀
    ret, frame = cap.read()
    if not ret:
        print("無法讀取攝像機視訊流。")
        break

    # 將照片轉換為灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊進行降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny邊緣檢測
    edged = cv2.Canny(blurred, 50, 150)

    # 尋找輪廓
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 50
    max_area = 1000
    contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    # 根據輪廓範圍獲取車牌區域
    if contours:
        plate_contour = max(contours, key=cv2.contourArea)

        # 進行多邊形逼近
        epsilon = 0.02 * cv2.arcLength(plate_contour, True)
        plate_contour = cv2.approxPolyDP(plate_contour, epsilon, True)

        # 繪製車牌區域的矩形
        x, y, w, h = cv2.boundingRect(plate_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # 提取車牌區域
        plate_roi = frame[y:y + h, x:x + w]

        # 進行腐蝕和擴張
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(plate_roi, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # 將車牌區域轉換為灰階
        plate_gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        # 使用自適應閾值進行二值化處理
        thresh = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)

        # 進行車牌辨識
        text = pytesseract.image_to_string(thresh, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', lang='eng')
        text = re.sub(r'[^A-Z0-9-]', '', text)

        # 更新 MongoDB 中的每一個車位狀態
        for spot_id, parking_spot_info in parking_spots.items():
            if (
                parking_spot_info["coordinates"][0][0] <= x <= parking_spot_info["coordinates"][2][0] and
                parking_spot_info["coordinates"][0][1] <= y <= parking_spot_info["coordinates"][1][1]
            ):
                parking_spots[spot_id]["plate"] = text
                parking_spots[spot_id]["status"] = "Occupied"

                # 顯示所有車格的狀態
                for other_spot_id, other_parking_spot_info in parking_spots.items():
                    print(f"車位：{other_spot_id}, 狀態：{other_parking_spot_info['status']}, 車牌號碼：{other_parking_spot_info['plate']}")

                # 添加以下兩行以更新圖像和顯示其他車格的狀態
                parking_lot_status = "Occupied"
                cv2.putText(frame, f"{spot_id}: {parking_lot_status}", (parking_spot_info["coordinates"][0][0], parking_spot_info["coordinates"][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 顯示其他車格的狀態
                for other_spot_id, other_parking_spot_info in parking_spots.items():
                    if other_spot_id != spot_id:
                        cv2.putText(frame, f"{other_spot_id}: {other_parking_spot_info['status']}", (other_parking_spot_info["coordinates"][0][0], other_parking_spot_info["coordinates"][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                break

    # 更新剩余停车位数量
    remaining_spots = sum(1 for spot_info in parking_spots.values() if spot_info["status"] == "Vacant")

    # 在左上角顯示車牌號碼和剩餘停車位數量
    cv2.putText(frame, f"Remaining Parking Spots : {remaining_spots}", (10, 30 * (len(parking_spots) + 1) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    print(f"Remaining Parking Spots: {remaining_spots}")

    # 顯示即時視訊
    cv2.imshow('Parking - Real-time', frame)

    # 按下 'q' 鍵退出迴圈
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break
    


# 釋放資源
cap.release()
cv2.destroyAllWindows()

