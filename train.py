!pip install --quiet torch torchvision scikit-learn


import random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ─────────── config ───────────
SEED, BATCH = 42, 32
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


CLASSES = [
   'Ultra cloudy', 'very cloudy', 'cloudy',
   'lightly cloudy', 'lightly clear', 'clear'
]
lbl2idx = {c:i for i,c in enumerate(CLASSES)}
NCLASS  = len(CLASSES)
EXPECTED = 212


# ─────────── gray-world WB ───────────
def gray_world(img: Image.Image)->Image.Image:
   arr = np.asarray(img).astype(np.float32)
   mean = arr.reshape(-1,3).mean(0) + 1e-6
   arr *= mean.mean()/mean
   return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
wb_tf = transforms.Lambda(gray_world)


# ─────────── transforms ───────────
train_tf = transforms.Compose([
   wb_tf,
   transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
   transforms.RandAugment(num_ops=2, magnitude=9),          # ← added
   transforms.RandomApply([transforms.ColorJitter(0.8,0.8,0.8,0.1)], p=0.9),
   transforms.RandomAutocontrast(p=0.5),
   transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
   transforms.RandomHorizontalFlip(),
   transforms.RandomRotation(15),
   transforms.ToTensor(),
   transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
   wb_tf,
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ─────────── dataset class ─────────
class ImgDS(Dataset):
   def __init__(self, paths, labels, tf):
       self.p, self.y, self.tf = paths, labels, tf
   def __len__(self): return len(self.p)
   def __getitem__(self, i):
       img = Image.open(self.p[i]).convert('RGB')
       return self.tf(img), self.y[i]


# ─────────── gather paths ──────────
ROOT = Path('/content/data')
IMG_DIR = ROOT/'images' if (ROOT/'images').exists() else ROOT


bucket = {c: [] for c in CLASSES}
for p in IMG_DIR.iterdir():
   if not p.is_file(): continue
   n = p.name.lower()
   if   'lightly clear'  in n: bucket['lightly clear'].append(p)
   elif 'clear'          in n: bucket['clear'].append(p)
   elif 'ultra'          in n: bucket['Ultra cloudy'].append(p)
   elif 'very'           in n: bucket['very cloudy'].append(p)
   elif 'lightly cloudy' in n: bucket['lightly cloudy'].append(p)
   elif 'cloudy'         in n: bucket['cloudy'].append(p)


for c, files in bucket.items():
   if len(files) != EXPECTED:
       raise RuntimeError(f"{c}: expected {EXPECTED}, found {len(files)}")


paths  = np.array(sum(bucket.values(), []))
labels = np.array([lbl2idx[c] for c in bucket for _ in bucket[c]])
print(f"✔️  Loaded {len(paths)} images ({EXPECTED} per class).")


# ─────────── 80/20 split ───────────
tr_idx, va_idx = train_test_split(
   np.arange(len(paths)), test_size=0.2,
   stratify=labels, random_state=SEED)


tr_ds = ImgDS(paths[tr_idx], labels[tr_idx], train_tf)
va_ds = ImgDS(paths[va_idx], labels[va_idx], val_tf)
tr_ld = DataLoader(tr_ds, BATCH, shuffle=True , num_workers=4, pin_memory=True)
va_ld = DataLoader(va_ds, BATCH, shuffle=False, num_workers=4, pin_memory=True)


# ─────────── MixUp helper ──────────
def mixup(x,y,a=0.8):   # ← increased alpha
   lam = np.random.beta(a,a) if a>0 else 1.
   idx = torch.randperm(x.size(0), device=x.device)
   return lam*x + (1-lam)*x[idx], y, y[idx], lam


# ─────────── model (ResNet-34) ─────
def make_model():
   m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
   f = m.fc.in_features
   m.fc = nn.Sequential(
       nn.Dropout(0.5),
       nn.Linear(f, 512), nn.ReLU(True),
       nn.Dropout(0.5),
       nn.Linear(512, NCLASS)
   )
   return m.to(device)


model = make_model()


# ─────────── loss w/ label smoothing ─
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# ─────────── epoch runner ──────────
def run_epoch(model, loader, optimizer=None, mix=True):
   train = optimizer is not None
   model.train() if train else model.eval()
   total_loss = correct = count = 0


   for xb, yb in loader:
       xb, yb = xb.to(device), yb.to(device)


       if train and mix:
           xb, y1, y2, lam = mixup(xb, yb)
           out = model(xb)
           loss = lam*criterion(out, y1) + (1-lam)*criterion(out, y2)
           preds = out.argmax(1)
           correct += lam*(preds==y1).sum().item() + (1-lam)*(preds==y2).sum().item()
       else:
           out = model(xb)
           loss = criterion(out, yb)
           correct += (out.argmax(1)==yb).sum().item()


       if train:
           optimizer.zero_grad()
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ← clip
           optimizer.step()


       total_loss += loss.item() * xb.size(0)
       count += xb.size(0)


   return total_loss/count, correct/count


# ─────────── training setup ──────────
max_epochs = 20
steps_per_epoch = len(tr_ld)


# Replace Stage-1/Stage-2 freeze logic with a single OneCycle schedule
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.OneCycleLR(
   optimizer, max_lr=1e-3,
   total_steps=steps_per_epoch * max_epochs,
   pct_start=0.3, div_factor=25.0, final_div_factor=1e4
)


best_val = 1e9
ckpt = '/content/best.pth'


for epoch in range(1, max_epochs+1):
   trL, trA = run_epoch(model, tr_ld, optimizer, mix=True)
   vaL, vaA = run_epoch(model, va_ld, None, mix=False)
   scheduler.step()


   print(f"E{epoch:02d}  TrL {trL:.3f} A {trA:.3f} |  VaL {vaL:.3f} A {vaA:.3f}")


   if vaL < best_val:
       best_val = vaL
       torch.save(model.state_dict(), ckpt)


# ─────────── final evaluation ──────────
model.load_state_dict(torch.load(ckpt))
model.eval()


preds, gts = [], []
with torch.no_grad():
   for xb, yb in va_ld:
       xb = xb.to(device)
       preds.extend(model(xb).argmax(1).cpu().tolist())
       gts.extend(yb.tolist())


acc = (np.array(preds) == np.array(gts)).mean()
print(f"\nFinal Validation Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(gts, preds))
