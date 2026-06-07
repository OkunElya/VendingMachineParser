# Vending Machine Parser

A computer-vision pipeline that turns a photo of a vending machine into a
structured product grid: it locates the machine, rectifies its display
window, detects every product on the shelves, matches each one against a
reference gallery, and returns a row/column table of product names and
confidence scores. A camera-based web app (FastAPI + Vite/TypeScript) sits on
top so it can be used straight from a phone.

## Documentation

- 📋 [Project report (English)](docs/REPORT_EN.md) — pipeline architecture,
  dataset creation & adaptation, pretrained models & fine-tuning, the web app,
  current abilities, full setup/preparation manual, and further work.
- 📋 [Отчёт по проекту (русский)](docs/REPORT_RU.md) — тот же отчёт на
  русском языке.
- 🖥️ [Web frontend README](web/README.md) — frontend stack, build/dev
  instructions, and usage notes.

## Quick start

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp tokens_example.yaml tokens.yaml   # then edit in real tokens
cd web && npm install && npm run build && cd ..
python3 api.py                       # serves API + web app on :8004
```

See the [project report](docs/REPORT_EN.md#project-setup--preparation-manual)
for the full setup manual, including where to get the datasets and model
weights.
