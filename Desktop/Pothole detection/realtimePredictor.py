import cv2
import time
import imutils
import numpy as np
from keras.models import load_model

global loadedModel
size = 300

def predict_pothole(currentFrame):
    currentFrame = cv2.resize(currentFrame, (size, size))
    currentFrame = currentFrame.reshape(1, size, size, 1).astype('float')
    currentFrame = currentFrame / 255
    prob = loadedModel.predict(currentFrame)
    max_prob = np.max(prob)
    if max_prob > 0.5:
        predicted_class = np.argmax(prob)
        return predicted_class, max_prob
    return "none", 0

def detect_squares(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            squares.append(approx)
    
    return squares

if __name__ == '__main__':
    loadedModel = load_model("C:/Users/Himanshu Chitoria/latest_full_model.h5") 

    # camera = cv2.VideoCapture(0)

    # show_pred = False

    # while True:
    #     (grabbed, frame) = camera.read()
    #     frame = imutils.resize(frame, width=700)
    #     frame = cv2.flip(frame, 1)
    #     clone = frame.copy()

    #     grayClone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

    #     pothole, prob = predict_pothole(grayClone)

    #     keypress_toshow = cv2.waitKey(1)

    #     if keypress_toshow == ord("e"):
    #         show_pred = not show_pred

    #     if show_pred:
    #         cv2.putText(clone, str(pothole) + ' ' + str(prob * 100) + '%', (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

    #     cv2.imshow("GrayClone", grayClone)
    #     cv2.imshow("Video Feed", clone)

    camera = cv2.VideoCapture(0)
    
    while True:
        ret, frame = camera.read()
        
        if not ret:
            break

        squares = detect_squares(frame)

        for square in squares:
            cv2.drawContours(frame, [square], 0, (0, 255, 0), 2)

        cv2.imshow("Squares Detection", frame)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break



    camera.release()
    cv2.destroyAllWindows()