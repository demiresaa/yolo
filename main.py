from ultralytics import YOLO
# bu kodda webvam açıp nesne tanıma yaptık yolonun içindeki sınıflar ile
model=YOLO("yolov8n.pt")
model.predict(source="0",show=True) 


# VİDEO ÜZRİNDEN NESNE TANIMA
model=YOLO("yolov8n.pt")
model.predict(source="video.mp4",show=True)

# Video uzerinden nesne tanıma ve kaydetme
model=YOLO("yolov8n.pt")
model.predict(source="video.mp4",show=True,save=True)# run detetc ve predict içine kayıt ediyor

# bu kodlar kolay nesne tanımaydı


import cv2
import cvzone # type: ignore
import math
import time

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0


# opne cv ve yolo ile nesne tanıma  WEB CAM İÇİN
cap=cv2.VideoCapture(0) 
cap.set(3,1280)
cap.set(4,720)
model=YOLO("yolov8n.pt")
# video kayıt için fourcc ve VideoWriter tanımlama
cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
success, img = cap.read()
print(img.shape)
cv2.imwrite("ornek_resim.jpg", img)
size = list(img.shape)
del size[2]
size.reverse()
video = cv2.VideoWriter("kaydedilen_video.mp4", cv2_fourcc, 24, size) #output video name, fourcc, fps, size


while True:
    new_frame_time = time.time()
    succes,img=cap.read()# görseldeki her bir pikseli okumaya yarıyor
    results=model(img,stream=True)
    cv2.imshow("İmage",img)
    cv2.waitKey(0) # BURAYA kadar olan kodda sadece video alıyor nesne tanıma yapmıyor
    for r in results:
        boxes = r.boxes
        for box in boxes: # etrafındaki dikdortgen yapma
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0] # kordinatlar bunlar
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h)) # köşe görünüm yapıyor nesnede
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])     # bu kodda biz nesne tanıma yaptık dikdörtgen oluşturduk yani
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    # video kayıt
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print("fps: ", fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1) # 
    video.release()
    # kodlarda videodaki webcamdeki nesneleri kendşmşz tanıdık






