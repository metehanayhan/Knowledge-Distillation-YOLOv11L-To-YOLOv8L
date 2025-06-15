import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


CHECKPOINT_PATH = 'final_distilled_model_with_detect.pth'
YAML_PATH = 'yolov8.yaml'
OUTPUT_PATH = 'final_merged_model.pt'


print(" Checkpoint yükleniyor...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

# Öğrenci modelini DetectionModel ile yükleyelim
student = DetectionModel(cfg=YAML_PATH)

# Student ağırlıkları
student.load_state_dict(checkpoint['student_state_dict'], strict=False)

# Adaptör ağırlıklarını alalim
adapters_state = checkpoint['adapters_state_dict']

# Eğer adaptörler varsa yeniden kur
adapters = None
if adapters_state:
    print(" Adaptasyon katmanları oluşturuluyor...")
    # Burada adaptör isimlerini normalize edeceğiz
    clean_keys = set(k.split('.')[0] for k in adapters_state.keys())  # sadece 'A4', 'A6' gibi
    
    adapters = nn.ModuleDict()
    for k in clean_keys:
        adapters[k] = nn.Sequential()  # Boş sequential placeholder
    
    # State yüklemesi sırasında otomatik eşleştirecek
    adapters.load_state_dict({k.replace('.', '_'): v for k, v in adapters_state.items()}, strict=False)

print(" Öğrenci ve adaptörler yüklendi.")

# Adaptörleri Modele Gömme 
if adapters:
    print(" Adaptörler öğrenci modeline gömülüyor...")

    STUDENT_LAYERS = [4, 6, 8, 9]  # Feature layerlar
    
    for idx in STUDENT_LAYERS:
        adapter_key = f"A{idx}"
        if adapter_key in adapters:
            original_layer = student.model[idx]
            adapter_layer = adapters[adapter_key]
            # Sequential bağla: önce original, sonra adaptör
            student.model[idx] = nn.Sequential(original_layer, adapter_layer)

    print(" Tüm adaptörler modele entegre edildi.")
else:
    print(" Adaptör bulunamadı, öğrenci model yalnız kaydedilecek.")

# Modeli Kaydedelm..
torch.save(student.state_dict(), OUTPUT_PATH)
print(f" Final model kaydedildi: {OUTPUT_PATH}")
