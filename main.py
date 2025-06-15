import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ultralytics import YOLO  
import torchvision.transforms as T
from tqdm import tqdm
from dataset import ImageDataset 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


IMG_SIZE = (640, 640)         
BATCH_SIZE = 8                
EPOCHS = 20                    
LR = 1e-4                   
FEATURE_W = 0.6              
SOFT_W = 0.2                  
DETECT_W = 0.4             
TEMP = 2.0                  

TEACHER_PATH = 'yolo11l.pt'  
STUDENT_PATH = 'yolov8l.pt'    

# Öğretmen ve öğrenci modellerde eşleştirilecek ara katmanların indeksleri
LAYER_PAIRS = [
    {'teacher': 4, 'student': 4},
    {'teacher': 6, 'student': 6},
    {'teacher': 8, 'student': 8},
    {'teacher': 10, 'student': 9}
]

# Özellik Karşılaştırma Fonksiyonu 
def feature_loss(t_feats, s_feats, adapters):
    loss = 0
    for p in LAYER_PAIRS:
        t = t_feats[f"T{p['teacher']}"].detach()  # Öğretmen katman çıktısı (gradient takibi yok)
        s = s_feats[f"S{p['student']}"]           # Öğrenci katman çıktısı
        if f"A{p['student']}" in adapters:
            s = adapters[f"A{p['student']}"](s)   # Gerekirse adaptörden geçir
        loss += F.mse_loss(s, t)                  # MSE ile karşılaştır
    return loss / len(LAYER_PAIRS)                # Ortalama loss

#  Soft Target (KL Divergence) Kayıp Fonksiyonu 
def soft_target_loss(t_logits, s_logits, temp):
    if isinstance(t_logits, (list, tuple)):
        t_logits = t_logits[0]
    if isinstance(s_logits, (list, tuple)):
        s_logits = s_logits[0]

    t_logits = t_logits.view(t_logits.size(0), -1)  # [B, -1]
    s_logits = s_logits.view(s_logits.size(0), -1)

    min_dim = min(t_logits.shape[1], s_logits.shape[1])
    t_logits = t_logits[:, :min_dim]
    s_logits = s_logits[:, :min_dim]

    t_soft = F.softmax(t_logits / temp, dim=1)            # Yumuşatılmış öğretmen çıktısı
    s_log_soft = F.log_softmax(s_logits / temp, dim=1)    # Öğrenci çıktısı (log ile)

    return F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (temp ** 2)

# Çıktı Benzerlik Kayıp Fonksiyonu 
def detect_output_loss(t_logits, s_logits):
    if isinstance(t_logits, (list, tuple)):
        t_logits = t_logits[0]
    if isinstance(s_logits, (list, tuple)):
        s_logits = s_logits[0]

    t_logits = t_logits.view(t_logits.size(0), -1)
    s_logits = s_logits.view(s_logits.size(0), -1)

    min_dim = min(t_logits.shape[1], s_logits.shape[1])
    return F.mse_loss(s_logits[:, :min_dim], t_logits[:, :min_dim])

#  Ana Eğitim  
def main():
    #  Görselleri dönüştürme işlemleri (Resize, Tensor çevirisi)
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

    #  Doğrulama veri yükleyici
    val_loader = DataLoader(
        ImageDataset('val_images', transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    #  YOLO modellerini yükle (eval: dropout, bn vs. pasif)
    teacher = YOLO(TEACHER_PATH).model.to(device).eval()
    student = YOLO(STUDENT_PATH).model.to(device).train()

    #  Ara katman çıktılarının tutulacağı sözlükler
    t_feats, s_feats = {}, {}

    #  Hook fonksiyonu: belirli katmanların çıktısını alıp sözlüğe kaydeder
    def hook(layer_dict, key):
        return lambda _, __, out: layer_dict.__setitem__(key, out[0] if isinstance(out, tuple) else out)

    #  Öğretmen ve öğrenci modellerin katmanlarına hook bağlanması
    for p in LAYER_PAIRS:
        teacher.model[p['teacher']].register_forward_hook(hook(t_feats, f"T{p['teacher']}"))
        student.model[p['student']].register_forward_hook(hook(s_feats, f"S{p['student']}"))

    #  Adaptörler sözlüğü (katman boyut farkı varsa düzeltmek için)
    adapters = nn.ModuleDict()
    with torch.no_grad():
        dummy = torch.randn(1, 3, *IMG_SIZE).to(device)
        teacher(dummy)
        student(dummy)

    #  Gerekirse adaptör katmanları ekleniyor (Boyut uyuşmazlıkları için)
    for p in LAYER_PAIRS:
        t_shape = t_feats[f"T{p['teacher']}"].shape
        s_shape = s_feats[f"S{p['student']}"].shape
        layers = []
        if t_shape[2:] != s_shape[2:]:
            layers.append(nn.AdaptiveAvgPool2d(t_shape[2:]))
        if t_shape[1] != s_shape[1]:
            layers.append(nn.Conv2d(s_shape[1], t_shape[1], kernel_size=1, bias=False))
        if layers:
            adapters[f"A{p['student']}"] = nn.Sequential(*layers).to(device)

    #  Önceden eğitilmiş adaptörler yükleniyor
    adapters.load_state_dict(torch.load("adapters_pretrained.pth"))
    adapters.train()  # adaptörler eğitilecek

    #  Öğrenci + adaptör parametreleri optimize edilecek
    optimizer = optim.AdamW(
        list(student.parameters()) + list(adapters.parameters()), lr=LR
    )

    best_val_loss = float("inf")  # En iyi doğrulama loss'unu tutmak için

    #   Eğitim Döngüsü
    for epoch in range(1, EPOCHS + 1):
        student.train()
        adapters.train()
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs in loop:
            imgs = imgs.to(device)
            t_feats.clear()
            s_feats.clear()
            with torch.no_grad():
                t_out = teacher(imgs)       # Öğretmen çıktısı
            s_out = student(imgs)           # Öğrenci çıktısı

            #  Kayıplar hesaplanıyor
            f_loss = feature_loss(t_feats, s_feats, adapters)
            s_loss = soft_target_loss(t_out, s_out, TEMP)
            d_loss = detect_output_loss(t_out, s_out)

            #  Toplam kayıp ağırlıklı olarak hesaplanıyor
            total_loss = FEATURE_W * f_loss + SOFT_W * s_loss + DETECT_W * d_loss

            #  Geri yayılım ve optimizasyon
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_total += total_loss.item()
            loop.set_postfix(loss=total_loss.item())  # Ekranda loss göster

        #  VALIDATION 
        student.eval()
        adapters.eval()
        val_total = 0
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(device)
                t_feats.clear()
                s_feats.clear()
                t_out = teacher(imgs)
                s_out = student(imgs)

                #  Validation için kayıplar
                f_loss = feature_loss(t_feats, s_feats, adapters)
                s_loss = soft_target_loss(t_out, s_out, TEMP)
                d_loss = detect_output_loss(t_out, s_out)
                total_loss = FEATURE_W * f_loss + SOFT_W * s_loss + DETECT_W * d_loss

                val_total += total_loss.item()

        #  Ortalama doğrulama kaybı
        val_avg = val_total / len(val_loader)
        print(f"\n Val Loss: {val_avg:.4f}")

        #  En iyi model kaydı
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save({
                'student_state_dict': student.state_dict(),
                'adapters_state_dict': adapters.state_dict()
            }, "final_distilled_model_with_detect.pth")
            print(f"  En iyi model kaydedildi (Loss: {best_val_loss:.4f})")

    print("\n Ana distilasyon tamamlandı.")


if __name__ == "__main__":
    main()
