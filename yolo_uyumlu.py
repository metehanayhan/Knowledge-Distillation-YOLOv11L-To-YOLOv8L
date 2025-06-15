import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO

MERGED_MODEL_PATH = 'final_merged_model.pt'
YAML_CFG_PATH = 'yolov8.yaml'
FINAL_SAVE_PATH = 'final_ready_for_yolo2.pt'

#  STATE_DICT DÜZELTME
def clean_state_dict(state_dict):
    """Layer isimlerini DetectionModel ile uyumlu hale getirir."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'model.' in k:
            # Örneğin: model.4.0.cv1.conv.weight → model.4.cv1.conv.weight
            parts = k.split('.')
            if parts[2].isdigit():
                new_key = f"{parts[0]}.{parts[1]}.{'.'.join(parts[3:])}"
            else:
                new_key = k
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

print(" DetectionModel yükleniyor...")

# DetectionModel oluştur
model = DetectionModel(cfg=YAML_CFG_PATH)

# Checkpoint oku
ckpt = torch.load(MERGED_MODEL_PATH, map_location='cpu')

# Checkpoint içeriğini düzelt
if 'model_state_dict' in ckpt:
    corrected_ckpt = clean_state_dict(ckpt['model_state_dict'])
elif isinstance(ckpt, dict):
    corrected_ckpt = clean_state_dict(ckpt)
else:
    raise ValueError(" Hatalı checkpoint formatı!")

# Ağırlıkları yükleyelim
missing, unexpected = model.load_state_dict(corrected_ckpt, strict=False)
print(" Model başarıyla yüklendi!")

# Eksik veya fazla kalanlar
if missing:
    print(f" Eksik parametreler: {missing}")
if unexpected:
    print(f" Beklenmeyen parametreler: {unexpected}")

# YOLO FORMATINDA KAYDETME 
print(" YOLO API uyumlu pt dosyası kaydediliyor...")

# Ultralytics tarzında kaydet
torch.save({'model': model}, FINAL_SAVE_PATH)

# Ultralytics YOLO ile test
yolo_model = YOLO(FINAL_SAVE_PATH)

# YOLO modeli torch olarak
yolo_model.export(format='torch', imgsz=640, optimize=True)

print(f" Başarıyla kaydedildi: {FINAL_SAVE_PATH}")
