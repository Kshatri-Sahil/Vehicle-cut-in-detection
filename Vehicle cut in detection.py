#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2

# Define object specific variables
focal_length = 15  # Use the calculated focal length
car_width = 200    # Actual width of a car in centimeters
warning_distance = 10  # Distance threshold for displaying warning message (in centimeters)

# Load pre-trained MobileNet-SSD model and prototxt
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Class labels MobileNet-SSD was trained on
class_labels = {7: 'car'}

# Function to calculate distance from the camera
def calculate_distance(rect_width):
    distance = (car_width * focal_length) / rect_width
    return distance

# Function to draw a transparent overlay
def draw_transparent_overlay(img, overlay_text, position, color, alpha=0.6):
    overlay = img.copy()
    output = img.copy()

    # Draw a rectangle with transparency
    cv2.rectangle(overlay, position[0], position[1], color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Put the text on the transparent rectangle
    cv2.putText(output, overlay_text, (position[0][0] + 10, position[0][1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    return output

# Extract frames from webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow('Vehicle cut in detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Vehicle cut in detection', 700, 600)

while True:
    ret, img = cap.read()
    
    if not ret:
        break

    # Prepare the frame for object detection
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Iterate over all detected objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if confidence > 0.5 and idx in class_labels:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            rect_width = endX - startX

            # Calculate and display distance from camera
            distance = calculate_distance(rect_width)
            label = f"{class_labels[idx]}: {confidence * 100:.2f}% Distance: {distance:.2f} cm"
            
            # Choose color based on distance
            if distance < warning_distance:
                box_color = (0, 0, 255)  # Red for warning
                img = draw_transparent_overlay(img, "Warning! STOP", ((startX, startY - 60), (endX, startY - 10)), (0, 0, 255))
            elif distance < 2 * warning_distance:
                box_color = (0, 255, 255)  # Yellow for caution
            else:
                box_color = (0, 255, 0)  # Green for safe

            cv2.rectangle(img, (startX, startY), (endX, endY), box_color, 2, cv2.LINE_AA)
            img = draw_transparent_overlay(img, label, ((startX, startY - 40), (endX, startY)), box_color, alpha=0.4)

    # Display the image
    cv2.imshow('Vehicle cut in detection', img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

