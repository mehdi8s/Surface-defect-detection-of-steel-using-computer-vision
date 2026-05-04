# Çelik yüzey kusuru sınıflandırması (NEU-DET)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-transfer%20learning-ee4c2c.svg)](https://pytorch.org/)

**English:** Six-class **surface defect classification** on steel strip images using the **NEU-DET** dataset, **ResNet-18** transfer learning, training/evaluation scripts, and an optional **Gradio** demo.

**Türkçe:** NEU-DET veri seti üzerinde sıcak haddelenmiş çelik şerit görüntülerinde **altı kusur sınıfı** için derin öğrenme tabanlı **sınıflandırma** hattı. Operatör yükünü azaltmak ve hat üzerinde tekrarlanabilir kalite kararı desteği hedeflenmiştir.

Detaylı akademik açıklamalar için kök dizindeki **`Rapor.pdf`** ve `steel-defect-project/RAPOR.html` dosyalarına bakınız.

---

## İçindekiler

- [Özellikler](#özellikler)
- [Yazarlar](#yazarlar)
- [Veri seti](#veri-seti-neu--det)
- [Yöntem ve hiperparametreler](#yöntem-ve-hiperparametreler)
- [Deneysel sonuçlar](#deneysel-sonuçlar)
- [Depo yapısı](#depo-yapısı)
- [Kurulum ve kullanım](#kurulum-ve-kullanım)
- [Çıktılar ve yeniden üretilebilirlik](#çıktılar-ve-yeniden-üretilebilirlik)
- [Sınırlamalar ve gelecek çalışmalar](#sınırlamalar-ve-gelecek-çalışmalar)
- [Kaynakça](#kaynakça)

---

## Özellikler

- **6 sınıf:** `crazing` (Cr), `inclusion` (In), `patches` (Pa), `pitted_surface` (PS), `rolled-in_scale` (RS), `scratches` (Sc)
- **Eğitim / doğrulama / tahmin:** `train.py`, `evaluate.py`, `predict.py`
- **Web arayüzü:** `gradio_app.py` ile tarayıcıdan görüntü yükleme
- **Raporlama:** karışıklık matrisi, sınıflandırma raporu, eğri grafikleri (`outputs/<run>/`)

---

## Veri seti (NEU-DET)

**NEU-DET**, gri tonlamalı çelik yüzey görüntülerinden oluşur; sınıf etiketleri klasör yapısı ile taşınır. Bu çalışmada sınıf başına **300** görüntü kullanılmış; standart bölünüşe uygun olarak **240 eğitim** ve **60 doğrulama** görüntüsü **sınıf başına** ayrılmıştır.

| Sınıf (klasör)     | Kısaltma | Eğitim / Doğrulama |
|--------------------|----------|--------------------|
| crazing            | Cr       | 240 / 60           |
| inclusion          | In       | 240 / 60           |
| patches            | Pa       | 240 / 60           |
| pitted_surface     | PS       | 240 / 60           |
| rolled-in_scale    | RS       | 240 / 60           |
| scratches          | Sc       | 240 / 60           |
| **Toplam**         |          | **1440 + 360 = 1800** |

Görüntüler tipik olarak yaklaşık **200×200** pikseldir; modele **224×224** yeniden ölçekleme ile verilir.

Ham veri bu repoda **`NEU-DET/`** alt dizinindedir (eğitim ve doğrulama alt kümeleri ile).

---

## Yöntem ve hiperparametreler

| Bileşen | Seçim |
|--------|--------|
| **Çatı (backbone)** | ResNet-18, ImageNet ön-eğitimli ağırlıklar; son tam bağlı katman 6 sınıfa uyarlanmış transfer learning + ince ayar |
| **Kayıp** | Çok sınıflı çapraz entropi |
| **Optimizasyon** | AdamW, weight decay ile düzenleme |
| **Erken durdurma** | Doğrulama doğruluğu iyileşmezse en fazla **6 epoch** bekleme |

**Ön işleme ve artırma**

- Gri görüntü **3 kanala** kopyalanır.
- Normalize: ortalama **0,5** / standart sapma **0,5** (tensör üzerinde).
- Eğitim: **yatay çevirme**, **küçük açılı rastgele döndürme**; doğrulamada yalnızca ölçekleme + normalizasyon.

| Parametre | Değer |
|-----------|--------|
| Girdi boyutu | 224 × 224 |
| Batch size | 32 |
| Öğrenme oranı | 5×10⁻⁴ |
| Weight decay | 1×10⁻⁴ |
| Azami epoch | 25 |
| Erken durdurma sabrı | 6 epoch |

*Yazılım yığını:* Python 3, PyTorch, torchvision, scikit-learn (metrikler), Matplotlib/Seaborn (grafikler).

---

## Deneysel sonuçlar

Doğrulama kümesinde (**360** örnek) raporda raporlanan en iyi koşu:

| Metrik | Değer |
|--------|--------|
| Doğrulama doğruluğu | **1,000** |
| Makro F1 | **1,000** |
| Ağırlıklı F1 | **1,000** |
| Sınıf bazlı precision / recall / F1 | **1,00** (tüm sınıflar) |
| Eğitilen epoch (erken durdurma) | **11** |
| Örnek çıktı klasörü | `baseline_resnet18_20260501_164325` |

**Örnek tek görüntü olasılıkları** (doğrulama kümesinden `*_241.jpg`): raporda belirtildiği gibi çoğu sınıfta **0,99+** güven; `inclusion_241` için yaklaşık **0,956** örnek güven değeri verilmiştir.

> **Önemli not:** Bu doğruluk düzeyi, veri setinin aynı kaynak ve çekim koşullarındaki bölünüşü için geçerlidir. Farklı hat, aydınlatma veya kamera ile elde edilen görüntülerde performans **ayrıca ölçülmelidir** (dış doğrulama, saha testi veya çapraz doğrulama). Yüksek skor, endüstriyel genelleme garantisi değildir.

---

## Depo yapısı

```text
.
├── README.md                 ← Bu dosya (GitHub ana sayfası)
├── Rapor.pdf                 ← Ders raporu (PDF)
├── NEU-DET/                  ← Veri seti (train / validation)
├── steel-defect-project/
│   ├── README.md             ← Kısa Türkçe kullanım notları
│   ├── RAPOR.html            ← Yazdırılabilir HTML rapor
│   ├── RESULTS_SUMMARY.md    ← Sayısal sonuç özeti
│   ├── configs/default.yaml
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── scripts/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── gradio_app.py
│   └── src/steel_defect/     ← Paket kaynak kodu
└── .gitignore
```

---

## Kurulum ve kullanım

Tüm komutları **`steel-defect-project`** dizininde çalıştırın; veri yolu varsayılan olarak bir üst dizindeki **`../NEU-DET`** şeklindedir (`configs/default.yaml`).

### Ortam

```powershell
cd "steel-defect-project"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Eğitim

```powershell
python scripts/train.py --config configs/default.yaml
```

### Değerlendirme

```powershell
python scripts/evaluate.py `
  --dataset-root ../NEU-DET `
  --model outputs/<run_name>/best_model.pt `
  --class-map outputs/<run_name>/class_to_idx.json `
  --output-dir outputs/<run_name>/eval_again `
  --model-name resnet18
```

### Tek görüntü tahmini

```powershell
python scripts/predict.py `
  --image ../NEU-DET/validation/images/scratches/scratches_241.jpg `
  --model outputs/<run_name>/best_model.pt `
  --class-map outputs/<run_name>/class_to_idx.json `
  --model-name resnet18
```

### Gradio arayüzü

```powershell
pip install gradio
python scripts/gradio_app.py
```

Tarayıcı: `http://127.0.0.1:7860` — farklı ağırlık için `--model` ve `--class-map` argümanlarına bakınız (`steel-defect-project/README.md`).

---

## Çıktılar ve yeniden üretilebilirlik

Eğitim sonrası tipik çıktılar (`outputs/<run_name>/`):

- `best_model.pt` — en iyi doğrulama ağırlıkları  
- `history.json`, `training_curves.png`  
- `class_to_idx.json`, `classification_report.json`, `confusion_matrix.png`  

**GitHub notu:** `.gitignore` nedeniyle `outputs/`, sanal ortamlar ve önbellekler repoya dahil edilmemiştir. Ağırlık dosyalarını paylaşmak için [GitHub Releases](https://docs.github.com/repositories/releasing-projects-on-github/about-releases) veya harici depolama kullanabilirsiniz. Aynı **`seed`** ve ortamla (`configs/default.yaml` içinde `seed: 42`) eğitimi yeniden çalıştırarak sonuçları yakından üretebilirsiniz.

---

## Sınırlamalar ve gelecek çalışmalar

- **Sınıflandırma** odaklıdır; kutular (bounding box) ile **tespit (detection)** veya **segmentasyon** bir sonraki adım olabilir (YOLO, Faster R-CNN, vb.).
- Daha hızlı çıkarım için `mobilenet_v3_small` gibi hafif mimarilerle hız/doğruluk karşılaştırması.
- **K-fold** çapraz doğrulama ile bölünüşe bağlı şans başarısını azaltma.
- Endüstriyel senaryolar için **alan verisi** ile ince ayar ve kalibrasyon.

---

## Kaynakça

- **NEU-DET** — Northeastern University çelik yüzey kusuru görüntü veri seti (literatürde yaygın kullanılan NEU yüzey kusuru setleri ailesi). Atıf için veri setinin resmi yayın sayfasındaki önerilen bibliyografyayı kullanın.
- Bu proje: [github.com/mehdi8s/Surface-defect-detection-of-steel-using-computer-vision](https://github.com/mehdi8s/Surface-defect-detection-of-steel-using-computer-vision)

---

## Lisans

Veri seti kullanım koşulları NEU-DET’in orijinal lisansına tabidir. Kod için ayrı bir lisans belirtilmemişse, akademik ve eğitim amaçlı kullanım varsayılır; ticari kullanım için veri sağlayıcı koşullarını kontrol edin.
