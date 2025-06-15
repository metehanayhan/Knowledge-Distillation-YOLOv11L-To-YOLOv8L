import cv2
import time
from ultralytics import YOLO

COCO_CLASSES = [ 
    "insan", "bisiklet", "araba", "motorsiklet", "ucak", "otobus", "tren", "kamyon", "tekne",
    "trafik lambasi", "yangin hidrant", "dur tabela", "parkometre", "bank", "kus", "kedi",
    "kopek", "at", "koyun", "inek", "fil", "ayi", "zebra", "ziraf", "sirt cantasi",
    "semsiye", "el cantasi", "kravat", "valiz", "frisbi", "kayak", "kar tahtasi", "spor topu",
    "ucurtma", "beyzbol sopasi", "beyzbol eldiveni", "kaykay", "sorf tahtasi", "tenis raketi",
    "sise", "sarap bardagi", "kupa", "catalk", "bicak", "kasik", "kase", "muz", "elma",
    "sandvic", "portakal", "brokoli", "havuc", "sosisli", "pizza", "donut", "pasta", "sandalye",
    "kanepe", "saksili bitki", "yatak", "yemek masasi", "klozet", "televizyon", "laptop", "fare", "kumanda",
    "klavye", "cep telefonu", "mikrodalga", "firin", "tost makinesi", "lavabo", "buzdolabi", "kitap",
    "saat", "vazo", "makas", "oyuncak ayi", "sac kurutucu", "dis fircasi"
]


model = YOLO('final_ready_for_yolo.pt')  

cap = cv2.VideoCapture(0)  # 0: Bilgisayar kamerası
assert cap.isOpened(), "Kamera açılamadı!"

# FPS 
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    # Tahmin 
    results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)

    # Bounding Box
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        scores = result.boxes.conf.cpu().numpy()  # confidence skoru
        classes = result.boxes.cls.cpu().numpy()  # sınıf id'leri

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)

            # Kutu
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Sınıf adı ve skor
            class_name = COCO_CLASSES[int(cls)] if int(cls) < len(COCO_CLASSES) else str(int(cls))
            label = f"{class_name}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS Hesaplama
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # FPS yaz ekrana
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Görüntüyü göster
    cv2.imshow('YOLO Realtime', frame)

    # ESC'ye basınca çık
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()