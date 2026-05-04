# Yuzey Kusur Tespiti - Sonuc Ozeti

## Deney Bilgisi

- Veri seti: NEU-DET (6 sinif, toplam 1800 goruntu)
- Egitim/Test: 240/60 (sinif basi)
- Model: ResNet18 (transfer learning)
- Giris boyutu: 224x224
- Erken durdurma: aktif (patience=6)

## Final Performans

- En iyi validation accuracy: **1.0000**
- Macro F1: **1.0000**
- Weighted F1: **1.0000**
- Tum siniflarda precision/recall/F1: **1.00**

## Ornek Tek-Goruntu Tahminleri

- crazing_241.jpg -> `crazing` (0.9999+)
- inclusion_241.jpg -> `inclusion` (0.9564)
- patches_241.jpg -> `patches` (1.0000)
- pitted_surface_241.jpg -> `pitted_surface` (0.9999)
- rolled-in_scale_241.jpg -> `rolled-in_scale` (0.9984)
- scratches_241.jpg -> `scratches` (1.0000)

## Cikti Dosyalari

- `outputs/baseline_resnet18_20260501_164325/best_model.pt`
- `outputs/baseline_resnet18_20260501_164325/classification_report.json`
- `outputs/baseline_resnet18_20260501_164325/confusion_matrix.png`
- `outputs/baseline_resnet18_20260501_164325/training_curves.png`

## Sunumda Soylenebilecek Kritik Not

Bu veri setinde model cok yuksek performans vermistir. Endustriyel genelleme gucunu gostermek icin farkli kamera/saha kosullari ile ek dis veri testi veya capraz-dogrulama yapilmasi onerilir.
