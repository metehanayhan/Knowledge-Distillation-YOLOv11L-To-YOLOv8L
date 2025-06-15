import os
import cv2
import time
from ultralytics import YOLO
import subprocess


MODEL_PATH = "yolov8l.pt"                      
IMAGE_DIR = "images/val"                      
DATA_YAML = "data.yaml"               
NUM_IMAGES = 100                           

# FPS
def measure_fps(model_path, image_dir, num_images):
    model = YOLO(model_path)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))][:num_images]

    total_time = 0.0
    for file in image_files:
        img_path = os.path.join(image_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        start = time.time()
        _ = model(img)
        end = time.time()
        total_time += (end - start)

    avg_fps = len(image_files) / total_time
    print(f"\n Ortalama FPS: {avg_fps:.2f} (Toplam {len(image_files)} görsel)")

# mAP
def measure_map(model_path, data_yaml):
    print("\n mAP ölçümü başlatılıyor...")
    cmd = f"yolo task=detect mode=val model={model_path} data={data_yaml} imgsz=640"
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    measure_fps(MODEL_PATH, IMAGE_DIR, NUM_IMAGES)
    measure_map(MODEL_PATH, DATA_YAML)
