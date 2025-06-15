import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ultralytics import YOLO  
import torchvision.transforms as T
from tqdm import tqdm
import cv2
import os
import glob
from dataset import ImageDataset  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


IMG_SIZE = (640, 640)           # Görsellerin yeniden boyutlandırılacağı boyut
BATCH_SIZE = 8                  # Her iterasyonda kullanılacak görsel sayısı
EPOCHS = 3                      # Eğitim döngüsü sayısı
LR = 1e-3                       # Öğrenme oranı
TEACHER_PATH = 'yolo11l.pt'     # Öğretmen modelin dosya yolu
STUDENT_PATH = 'yolov8l.pt'     # Öğrenci modelin dosya yolu

# Öğretmen ve öğrenci modellerde eşleştirilecek ara katmanların indeksleri
LAYER_PAIRS = [
    {'teacher': 4, 'student': 4},
    {'teacher': 6, 'student': 6},
    {'teacher': 8, 'student': 8},
    {'teacher': 10, 'student': 9}
]

# ====================  Özellik Kaybı (Feature Loss) Fonksiyonu ====================
def feature_loss(t_feats, s_feats, adapters):
    loss = 0
    for p in LAYER_PAIRS:
        t = t_feats[f"T{p['teacher']}"].detach()  # Öğretmen katman çıktısı (grad takibi kapalı)
        s = s_feats[f"S{p['student']}"]           # Öğrenci katman çıktısı
        if f"A{p['student']}" in adapters:
            s = adapters[f"A{p['student']}"](s)   # Gerekirse adaptörden geçir
        loss += F.mse_loss(s, t)                  # MSE loss ile karşılaştır
    return loss / len(LAYER_PAIRS)                # Ortalama loss

# ====================  Ana Eğitim Fonksiyonu ====================
def main():
    #  Görseller üzerinde dönüşüm: PIL -> yeniden boyutlandır -> tensöre çevir
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(IMG_SIZE),
        T.ToTensor()
    ])

    #  Eğitim veri yükleyici
    train_loader = DataLoader(
        ImageDataset('images', transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True
    )

    #  YOLO modelleri (eval modunda sadece inference yapılır)
    teacher = YOLO(TEACHER_PATH).model.to(device).eval()
    student = YOLO(STUDENT_PATH).model.to(device).eval()

    #  Ara katman çıktılarının tutulacağı sözlükler
    t_feats, s_feats = {}, {}

    #  Hook fonksiyonu: belirli katmanların çıktısını alıp sözlüğe kaydeder
    def hook(layer_dict, key):
        return lambda _, __, out: layer_dict.__setitem__(key, out[0] if isinstance(out, tuple) else out)

    #  Öğretmen ve öğrenci modellerin katmanlarına hook bağlanması
    for p in LAYER_PAIRS:
        teacher.model[p['teacher']].register_forward_hook(hook(t_feats, f"T{p['teacher']}"))
        student.model[p['student']].register_forward_hook(hook(s_feats, f"S{p['student']}"))

    #  Adaptörler sözlüğü (katman uyumsuzluklarını düzeltmek için kullanılacak katmanlar)
    adapters = nn.ModuleDict()

    #  Öğretmen ve öğrenci çıktılarının boyutlarını öğrenmek için dummy veri gönder
    with torch.no_grad():
        dummy = torch.randn(1, 3, *IMG_SIZE).to(device)
        teacher(dummy)
        student(dummy)

    #  Her katman çifti için gerekiyorsa adaptör oluştur
    for p in LAYER_PAIRS:
        t_shape = t_feats[f"T{p['teacher']}"].shape
        s_shape = s_feats[f"S{p['student']}"].shape
        layers = []

        #  H x W çözünürlüğü farklıysa ortalama havuzlama ile eşitle
        if t_shape[2:] != s_shape[2:]:
            layers.append(nn.AdaptiveAvgPool2d(t_shape[2:]))

        #  Kanal sayısı farklıysa 1x1 konvolüsyon ile eşitle
        if t_shape[1] != s_shape[1]:
            layers.append(nn.Conv2d(s_shape[1], t_shape[1], 1, bias=False))

        #  Adaptör katmanını oluştur ve kaydet
        if layers:
            adapters[f"A{p['student']}"] = nn.Sequential(*layers).to(device)

    #  Öğrenci modelin ağırlıkları donduruluyor (sadece adaptörler eğitilecek)
    for p in student.parameters():
        p.requires_grad = False

    #  Sadece adaptörleri optimize edecek optimizasyon algoritması
    optimizer = optim.AdamW(adapters.parameters(), lr=LR)

    #  Eğitim Döngüsü 
    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, desc=f"Adaptör Epoch {epoch+1}")
        for imgs in loop:
            imgs = imgs.to(device)          # Görselleri cihaza gönder
            t_feats.clear()                 # Önceki katman çıktıları temizleniyor
            s_feats.clear()
            with torch.no_grad():
                _ = teacher(imgs)           # Öğretmen çıktısı alınır (gradient takibi olmadan)
            _ = student(imgs)               # Öğrenci çıktısı alınır (o da sadece forward)
            loss = feature_loss(t_feats, s_feats, adapters)  # Özellik kaybı hesaplanır
            optimizer.zero_grad()           # Gradients sıfırlanır
            loss.backward()                 # Geri yayılım yapılır
            optimizer.step()                # Ağırlıklar güncellenir (sadece adaptörler)
            loop.set_postfix(loss=loss.item())  # Ekranda loss değeri gösterilir

    #  Eğitilen adaptörler kaydett
    torch.save(adapters.state_dict(), "adapters_pretrained.pth")
    print(" Adaptörler başarıyla kaydedildi.")


if __name__ == "__main__":
    main()
