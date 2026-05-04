"""
Web arayuzu: goruntu yukle -> kusur sinifi tahmini (NEU-DET / ResNet18).
Calistirma (proje kokunden): python scripts/gradio_app.py
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from steel_defect.model import build_model

CUSTOM_CSS = """
/* Genel sayfa */
.gradio-container {
  max-width: 1100px !important;
  margin: 0 auto !important;
}
footer.svelte-1ax1toa { display: none !important; }

/* Hero */
.sd-hero-wrap { margin-bottom: 1.25rem; }
.sd-hero {
  background: linear-gradient(125deg, #0c1222 0%, #1a365d 42%, #0f766e 100%);
  color: #f8fafc;
  padding: 1.6rem 1.75rem 1.5rem;
  border-radius: 14px;
  box-shadow: 0 12px 40px rgba(15, 23, 42, 0.22);
  border: 1px solid rgba(255,255,255,0.08);
}
.sd-hero h1 {
  margin: 0 0 0.35rem 0;
  font-size: 1.55rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.2;
}
.sd-hero p {
  margin: 0 0 1rem 0;
  opacity: 0.92;
  font-size: 0.95rem;
  line-height: 1.45;
  max-width: 52rem;
}
.sd-badges { display: flex; flex-wrap: wrap; gap: 0.45rem; }
.sd-badge {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.2);
  color: #e2e8f0;
}

/* Paneller */
.sd-panel {
  background: #f8fafc !important;
  border-radius: 12px !important;
  border: 1px solid #e2e8f0 !important;
  padding: 0.75rem !important;
  box-shadow: 0 1px 3px rgba(15,23,42,0.06) !important;
}
.sd-panel-title {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #64748b;
  margin: 0 0 0.5rem 0.25rem;
}

/* Sonuc karti */
.sd-result-wrap { min-height: 120px; }
.sd-result-card {
  background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  padding: 1.1rem 1.15rem 1rem;
  box-shadow: 0 4px 14px rgba(15,23,42,0.06);
}
.sd-result-kicker {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #64748b;
  margin-bottom: 0.35rem;
}
.sd-result-class {
  font-size: 1.45rem;
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.02em;
  word-break: break-word;
  line-height: 1.2;
  margin-bottom: 0.75rem;
}
.sd-bar-track {
  height: 8px;
  border-radius: 999px;
  background: #e2e8f0;
  overflow: hidden;
  margin-bottom: 0.4rem;
}
.sd-bar-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #0d9488, #14b8a6);
  transition: width 0.35s ease;
}
.sd-result-meta {
  font-size: 0.88rem;
  color: #475569;
  font-weight: 600;
}
.sd-result-hint {
  margin-top: 0.65rem;
  font-size: 0.8rem;
  color: #64748b;
  line-height: 1.4;
}
.sd-result-empty {
  color: #64748b;
  font-size: 0.95rem;
  padding: 1rem 0.25rem;
  line-height: 1.5;
}

/* Tablo basligi */
.sd-df-wrap .label-wrap span { font-weight: 700 !important; color: #334155 !important; }

/* Footer */
.sd-footer {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid #e2e8f0;
  font-size: 0.78rem;
  color: #94a3b8;
  text-align: center;
  line-height: 1.5;
}
.sd-footer code { background: #f1f5f9; padding: 0.1rem 0.35rem; border-radius: 4px; color: #475569; }
"""


def _default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent.parent
    run = root / "outputs" / "baseline_resnet18_20260501_164325"
    return run / "best_model.pt", run / "class_to_idx.json"


def _hero_html() -> str:
    return """
<div class="sd-hero-wrap">
  <div class="sd-hero">
    <h1>Çelik şerit yüzey kusuru analizi</h1>
    <p>
      NEU-DET veri kümesi üzerinde eğitilmiş derin öğrenme modeli ile tek görüntüden
      <strong>altı tipik kusur</strong> sınıfından birini seçer. Demo amaçlıdır; endüstriyel kullanım için doğrulama önerilir.
    </p>
    <div class="sd-badges">
      <span class="sd-badge">ResNet-18</span>
      <span class="sd-badge">Transfer learning</span>
      <span class="sd-badge">6 sınıf</span>
      <span class="sd-badge">Bilgisayar görüşü</span>
    </div>
  </div>
</div>
"""


def _empty_result_html() -> str:
    return """<div class="sd-result-wrap"><div class="sd-result-empty">
    Görüntü yükleyip <strong>Tahmin et</strong> düğmesine basın. JPEG veya PNG önerilir.
    </div></div>"""


def _result_html(class_name: str, confidence: float) -> str:
    safe = html.escape(class_name)
    pct = max(0.0, min(100.0, confidence * 100.0))
    pct_txt = f"{pct:.2f}".replace(".", ",")
    return f"""<div class="sd-result-wrap"><div class="sd-result-card">
  <div class="sd-result-kicker">En olası sınıf</div>
  <div class="sd-result-class">{safe}</div>
  <div class="sd-bar-track"><div class="sd-bar-fill" style="width:{pct:.2f}%"></div></div>
  <div class="sd-result-meta">Güven: %{pct_txt}</div>
  <div class="sd-result-hint">Sağdaki tabloda tüm sınıfların olasılık sıralamasını görebilirsiniz.</div>
</div></div>"""


class Predictor:
    def __init__(
        self,
        model_path: Path,
        class_map_path: Path,
        model_name: str = "resnet18",
        image_size: int = 224,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        with class_map_path.open("r", encoding="utf-8") as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {int(v): k for k, v in self.class_to_idx.items()}
        self.image_size = image_size
        self.model = build_model(model_name, num_classes=len(self.class_to_idx))
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def predict(self, image: Image.Image | None):
        if image is None:
            return _empty_result_html(), None
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        order = torch.argsort(probs, descending=True)
        top_idx = int(order[0].item())
        top_label = self.idx_to_class[top_idx]
        top_p = float(probs[top_idx].item())

        rows = []
        for j in order.tolist():
            rows.append([self.idx_to_class[int(j)], float(probs[int(j)].item())])
        return _result_html(top_label, top_p), rows


def main() -> None:
    default_model, default_map = _default_paths()
    parser = argparse.ArgumentParser(description="Gradio arayuzu: celik kusur siniflandirma")
    parser.add_argument("--model", type=str, default=str(default_model))
    parser.add_argument("--class-map", type=str, default=str(default_map))
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None, help="cpu veya cuda; boş ise otomatik")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gradio public link (gecici)")
    args = parser.parse_args()

    model_path = Path(args.model)
    class_map_path = Path(args.class_map)
    if not model_path.is_file():
        raise SystemExit(f"Model bulunamadı: {model_path}")
    if not class_map_path.is_file():
        raise SystemExit(f"class_to_idx.json bulunamadı: {class_map_path}")

    predictor = Predictor(
        model_path=model_path,
        class_map_path=class_map_path,
        model_name=args.model_name,
        image_size=args.image_size,
        device=args.device,
    )

    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_md,
    ).set(
        button_primary_background_fill="linear-gradient(90deg, #0d9488, #0f766e)",
        button_primary_background_fill_hover="linear-gradient(90deg, #14b8a6, #0d9488)",
        button_large_text_weight="600",
        block_title_text_weight="600",
    )

    url_hint = f"http://{args.host}:{args.port}"

    with gr.Blocks(
        theme=theme,
        title="Yüzey kusuru — demo",
        css=CUSTOM_CSS,
    ) as demo:
        gr.HTML(_hero_html())

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["sd-panel"]):
                gr.HTML('<p class="sd-panel-title">Girdi</p>')
                img_in = gr.Image(
                    type="pil",
                    image_mode="RGB",
                    label="Çelik yüzey görüntüsü",
                    height=360,
                )
                btn = gr.Button("Tahmin et", variant="primary", size="lg")

            with gr.Column(scale=1, elem_classes=["sd-panel"]):
                gr.HTML('<p class="sd-panel-title">Çıktı</p>')
                out_html = gr.HTML(value=_empty_result_html())
                out_df = gr.Dataframe(
                    headers=["Sınıf", "Olasılık"],
                    datatype=["str", "number"],
                    label="Tüm sınıflar (yüksekten düşüğe)",
                    interactive=False,
                    elem_classes=["sd-df-wrap"],
                    wrap=True,
                )

        def run_pred(im):
            return predictor.predict(im)

        btn.click(fn=run_pred, inputs=img_in, outputs=[out_html, out_df])

        gr.HTML(
            f"""
            <div class="sd-footer">
              Model dosyası: <code>{html.escape(model_path.name)}</code>
              &nbsp;·&nbsp; Cihaz: <code>{html.escape(str(predictor.device))}</code>
              <br/>
              Yerel adres: <code>{html.escape(url_hint)}</code>
              — Aynı bilgisayarda tarayıcıdan açın.
            </div>
            """
        )

    print(f"\n  Arayüz hazır: {url_hint}\n")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
