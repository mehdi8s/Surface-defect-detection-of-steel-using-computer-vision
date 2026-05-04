# Yuzey Kusur Tespiti Projesi (NEU-DET)

Bu proje, NEU-DET veri seti uzerinde 6 sinifli celik yuzey kusuru siniflandirmasi yapar:

- `crazing` (Cr)
- `inclusion` (In)
- `patches` (Pa)
- `pitted_surface` (PS)
- `rolled-in_scale` (RS)
- `scratches` (Sc)

Proje; egitim, degerlendirme, tek goruntu tahmini ve raporlama ciktilari (confusion matrix + classification report) uretir.

## 1) Proje Yapisi

```text
steel-defect-project/
  configs/default.yaml
  scripts/train.py
  scripts/evaluate.py
  scripts/predict.py
  scripts/gradio_app.py
  src/steel_defect/
    data.py
    model.py
    trainer.py
    utils.py
  requirements.txt
  pyproject.toml
```

## 2) Kurulum

Windows PowerShell:

```bash
cd "m:\UNIVRSITE DOC\Donem 8\Bilgisayar gormesi\Odev\steel-defect-project"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 3) Veri Seti Konumu

Varsayilan konfigurasyonda veri seti yolu:

`../NEU-DET`

Bu yol `configs/default.yaml` icinden degistirilebilir.

## 4) Egitim

```bash
python scripts/train.py --config configs/default.yaml
```

Egitim tamamlandiginda `outputs/` altinda bir kosu klasoru olusur:

- `best_model.pt`
- `history.json`
- `class_to_idx.json`
- `training_curves.png`
- `classification_report.json`
- `confusion_matrix.png`

## 5) Degerlendirme (Ayrica)

```bash
python scripts/evaluate.py ^
  --dataset-root ../NEU-DET ^
  --model outputs/<run_name>/best_model.pt ^
  --class-map outputs/<run_name>/class_to_idx.json ^
  --output-dir outputs/<run_name>/eval_again ^
  --model-name resnet18
```

## 6) Tek Goruntu Tahmini

```bash
python scripts/predict.py ^
  --image ../NEU-DET/validation/images/scratches/scratches_241.jpg ^
  --model outputs/<run_name>/best_model.pt ^
  --class-map outputs/<run_name>/class_to_idx.json ^
  --model-name resnet18
```

## 7) Web arayuzu (Gradio)

Tarayicida goruntu yukleyip tahmin almak icin (varsayilan model: `outputs/baseline_resnet18_20260501_164325/`):

```bash
pip install gradio
python scripts/gradio_app.py
```

Tarayicida `http://127.0.0.1:7860` adresini acin. Baska model klasoru icin:

```bash
python scripts/gradio_app.py --model outputs/<run>/best_model.pt --class-map outputs/<run>/class_to_idx.json
```

## 8) Proje Sunumu Icin Onerilen Basliklar

Rapor/slayt hazirlarken su basliklari kullanabilirsin:

1. Problem tanimi ve endustriyel onemi
2. Veri seti ve sinif dagilimi
3. On isleme ve veri artirma
4. Model mimarisi (ResNet18)
5. Egitim stratejisi (early stopping, optimizer, lr)
6. Sonuclar (accuracy, confusion matrix, sinif bazli F1)
7. Hata analizi ve iyilestirme onerileri
8. Gelecek calisma (tespit/detection + segmentasyon)

## 9) Gelistirme Onerileri

- `model_name` icin `mobilenet_v3_small` deneyip hiz/dogruluk karsilastirmasi yap.
- `k-fold` capraz dogrulama ekle.
- Veri setindeki bounding box etiketlerini kullanarak ikinci asama object detection calismasi (Faster R-CNN veya YOLO) ekle.

