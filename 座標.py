import cv2

# 全域變數，用於存儲車位座標
parking_spots = []

# 鼠標點擊事件處理函數
def mouse_callback(event, x, y, flags, param):
    global parking_spots

    # 左鍵點擊事件
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(parking_spots) < 3:
            parking_spots.append((x, y))
            print(f"車位 {len(parking_spots)} 左上角座標：({x}, {y})")
        elif len(parking_spots) == 3:
            parking_spots[-1] += (x, y)
            print(f"車位 {len(parking_spots)} 右上角座標：({x}, {y})")
        elif len(parking_spots) == 6:
            parking_spots[-1] += (x, y)
            print(f"車位 {len(parking_spots)} 左下角座標：({x}, {y})")
        elif len(parking_spots) == 9:
            parking_spots[-1] += (x, y)
            print(f"車位 {len(parking_spots)} 右下角座標：({x}, {y})")
            print(f"車位 {len(parking_spots)//3} 的座標：{parking_spots[-3:]}")
        else:
            print("已標記完所有車位座標")

# 使用攝像頭
cap = cv2.VideoCapture(0)

# 創建一個窗口
cv2.namedWindow("Mark Parking Spots")

# 設置鼠標點擊事件的回調函數
cv2.setMouseCallback("Mark Parking Spots", mouse_callback)

while True:
    # 讀取一幀
    ret, frame = cap.read()
    if not ret:
        print("無法讀取攝像機視訊流。")
        break

    # 顯示車位座標
    for i, spot in enumerate(parking_spots):
        cv2.putText(frame, f"P{i+1}: {spot}", (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 顯示攝像頭畫面
    cv2.imshow("Mark Parking Spots", frame)

    # 等待用戶操作，按 'q' 鍵退出
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()

# 輸出標記的車位座標
print("標記的車位座標：", parking_spots)
