# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:22:45 2024

@author: Lenovo
"""
import cv2
import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def checkParkSpace(imgg, img):  #Park alanlarını kontrol eder, dolu veya boş olup olmadıklarını belirler.
    spaceCounter = 0
    
    for pos in posList:
        x, y = pos
        
        img_crop = imgg[y: y + height, x:x + width]
        count = cv2.countNonZero(img_crop)
        
        if count < 150:
            color = (0, 255, 0)
            spaceCounter += 1
        else:
            color = (0, 0, 255)
    
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, 2) 
        cv2.putText(img, str(count), (x, y + height - 2), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    
    cv2.putText(img, f"Free: {spaceCounter}/{len(posList)}", (15, 24), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 4)
    return spaceCounter

def addTimestamp(img):  #Görüntüye tarih ve saat damgası ekler.
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, current_time, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

def saveDailyReport(spaceCounter, totalSpaces):  #Günlük raporu kaydeder.
    now = datetime.now()
    report_filename = now.strftime("%Y-%m-%d") + ".txt"
    
    with open(report_filename, "a") as report_file:
        current_time = now.strftime("%H:%M:%S")
        report_file.write(f"{current_time} - Free: {spaceCounter}/{totalSpaces}\n")

def plotOccupancy(occupancy_data):   #Zaman içindeki doluluk oranını grafik olarak gösterir.
    times, occupancy_rates = zip(*occupancy_data)
    plt.figure(figsize=(10, 5))
    plt.plot(times, occupancy_rates, marker='o')
    plt.xlabel('ZAMAN')
    plt.ylabel('DOLULUK ORANI(%)')
    plt.title('ZAMAN İÇERİSİNDE OTOPARK DOLULUK ORANI')
    plt.grid(True)
    
    # X eksenindeki etiketlerin sayısını azaltma
    plt.xticks(times[::len(times)//10])  # Yalnızca belirli aralıklarla etiket göster
    
    plt.show()

width = 27
height = 15

cap = cv2.VideoCapture(r"C:\Users\Lenovo\.spyder-py3\otopark.mp4")

with open("CarParkPos", "rb") as f:
    posList = pickle.load(f)

occupancy_data = []

while True:
    success, img = cap.read()
    if not success:
        break
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    imgDilate = cv2.dilate(imgMedian, np.ones((3, 3)), iterations=1)
    
    spaceCounter = checkParkSpace(imgDilate, img)
    addTimestamp(img)
    saveDailyReport(spaceCounter, len(posList))
    
    # Doluluk oranınının kaydedilmesi
    occupancy_rate = (len(posList) - spaceCounter) / len(posList) * 100
    now = datetime.now().strftime("%H:%M:%S")
    occupancy_data.append((now, occupancy_rate))
    
    cv2.imshow("img", img)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Grafik çizimi
plotOccupancy(occupancy_data)


