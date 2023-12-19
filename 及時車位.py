import cv2
import numpy as np
import easyocr
from pymongo import MongoClient

# Set EasyOCR language and model
reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU if available

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['車位資料庫']
collection = db['車位']

# Initialize total parking spaces
total_spaces = 3  # Assuming there are three parking spaces

def process_parking_spaces(frame, parking_spaces):
    global total_spaces  # Use global variable

    for space_name, space_info in parking_spaces.items():
        # Create a mask to keep only the parking space region
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [space_info["points"]], (255, 255, 255))
        result = cv2.bitwise_and(frame, mask)

        # Convert the parking space region to grayscale
        roi_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization for low-light conditions
        roi_gray = cv2.equalizeHist(roi_gray)

        # Gaussian blur for noise reduction
        roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

        # Canny edge detection
        roi_edged = cv2.Canny(roi_blurred, 30, 150)

        # Adaptive thresholding for better binary image
        roi_thresh = cv2.adaptiveThreshold(roi_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)

        # Find contours
        roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 100
        max_area = 1000
        min_aspect_ratio = 1.5
        max_aspect_ratio = 3.0

        # Initialize parking space status
        status = "Vacant"  # Default status is vacant

        # Initialize text result
        text = ""

        # License plate recognition
        for roi_contour in roi_contours:
            # Calculate contour area
            area = cv2.contourArea(roi_contour)

            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(roi_contour)

            # Calculate aspect ratio
            aspect_ratio = float(w) / h

            # Filter contours
            if min_area < area < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                # Polygon approximation
                epsilon = 0.02 * cv2.arcLength(roi_contour, True)
                approx = cv2.approxPolyDP(roi_contour, epsilon, True)

                # Use shape fitting to ensure it is a rectangle
                if len(approx) == 4:
                    # Draw a rectangle around the license plate area
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                    # Extract the license plate area
                    plate_roi = frame[y:y + h, x:x + w]

                    # Convert the license plate area to grayscale
                    plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

                    # Use EasyOCR for text recognition
                    results = reader.readtext(plate_gray)

                    if results:
                        text = results[0][1]
                        status = "Occupied"  # If a license plate is detected, the parking space is occupied
                        total_spaces -= 1  # Decrease by one for each occupied space
                        break  # Stop the loop after detecting one license plate

        # Display parking space status and text result in the top left corner of the window
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Check parking space status, set text color to red for "Occupied" and green for "Vacant"
        text_color = (0, 0, 255) if status == "Occupied" else (0, 255, 0)

        cv2.putText(frame, f"{space_name} Status: {status} Plate: {text}", (10, 30 * int(space_name[-1])), font, 0.5, text_color, 2)

        # Output results to the console
        print(f"{space_name} Status: {status} Plate: {text}")

        # Write data to MongoDB
        data = {
            'space_name': space_name,
            'status': status,
            'plate': text
        }
        collection.insert_one(data)

    # Display the remaining vacant spaces in the top right corner
    cv2.putText(frame, f"Vacant Spaces: {total_spaces}", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    # Display real-time image
    cv2.imshow('Real-time Parking - Image', frame)

    # Output remaining vacant spaces to the console
    print(f"Vacant Spaces: {total_spaces}")

# Use a higher resolution camera (e.g., change '0' to the appropriate camera index)
cap = cv2.VideoCapture(0)  # 0 means using the default camera

# 停車位資訊
A1_left_top = (43, 305)
A1_left_bottom = (0, 386)
A1_right_top = (321, 307)
A1_right_bottom = (241, 386)

A2_left_top = (321, 307)
A2_left_bottom = (241, 386)
A2_right_top = (494, 297)
A2_right_bottom = (571, 391)

A3_left_top = (494, 297)
A3_left_bottom = (571, 391)
A3_right_top = (751, 291)
A3_right_bottom = (798, 382)

# 停車位資訊
parking_spaces = {
    "A1": {"points": np.array([A1_left_top, A1_left_bottom, A1_right_bottom, A1_right_top], dtype=np.int32)},
    "A2": {"points": np.array([A2_left_top, A2_left_bottom, A2_right_bottom, A2_right_top], dtype=np.int32)},
    "A3": {"points": np.array([A3_left_top, A3_left_bottom, A3_right_bottom, A3_right_top], dtype=np.int32)},
}

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("Unable to read camera video stream.")
        break

    # Process parking spaces
    process_parking_spaces(frame, parking_spaces)

    # Wait for a key event to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Close MongoDB connection
client.close()
