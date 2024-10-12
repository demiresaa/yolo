
import cv2
from ultralytics import YOLO

# Modeli yükleyin (alternatif olarak yolov8s.pt gibi farklı bir model de kullanabilirsiniz)
model = YOLO('yolov8n.pt')

# Görüntüyü yükleyin
image_path = 'C:/Users/Dell/Desktop/yolo/IMG_5750.JPG'  # Dosya yolunu doğru bir şekilde belirtin
image = cv2.imread(image_path)

# Görüntüde nesne tespiti yapın
results = model(image_path)

# Sonuçları kontrol edin ve belirli bir nesne olup olmadığını kontrol edin
object_name = 'dog'  # Buraya kontrol etmek istediğiniz nesne adını yazın

# Sonuçları kontrol edin
detected_objects = [model.names[int(cls)] for cls in results[0].boxes.cls]

if object_name in detected_objects:
    print(f'{object_name} bulundu.')
else:
    print(f'{object_name} bulunamadı.')

# Sonuçları görselleştirin
results[0].plot()
