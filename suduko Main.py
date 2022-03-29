print('Setting Up...')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from utils import *
import sudukoSolver
import time



###################################################################
pathImage = "sudukoTable.jpg"
heightImg = 387
widthImg = 387
model = initializePredictionModel()
###################################################################



#### 1. Prepairing the image
img = cv2.imread(pathImage)

img = cv2.resize(img, (widthImg, heightImg))
imgBlank = np.zeros((widthImg, heightImg, 3), np.uint8)
imgThreshold = preProcess(img)

#### 2. Find all countors
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierachy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

#### 3. Find the biggest contours
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 12)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2. warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
#### 4.
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    numbers = getPrediction(boxes, model)
    # print(numbers)
    imgDetectedigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 255, 0))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    print(posArray)

    board = np.array_split(numbers, 9)

    try:
        sudukoSolver.solve(board)
    except:
        pass

    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList*posArray
    imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

    #### 6. Overlay solution
    pts2 = np.float32(biggest)
    pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedigits = drawGrid(imgDetectedigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

else:
    print("no suduko found")


imageArray = ([img, imgContours, imgThreshold, imgBigContour],
              [imgDetectedigits, inv_perspective, imgInvWarpColored, inv_perspective])
#
#
# cv2.imshow("Original Image",img)
# cv2.imshow("Solved Image" ,inv_perspective)
# cv2.waitKey(0)
#
stackedImage = stackImages(imageArray, 1)
cv2.imshow('stacked images', stackedImage)
cv2.waitKey(0)


