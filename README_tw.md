# pdf2ppt

[![CI](https://github.com/poirotw66/pdf2ppt/actions/workflows/ci.yml/badge.svg)](https://github.com/poirotw66/pdf2ppt/actions/workflows/ci.yml)

[English](README.md) | **繁體中文**

`pdf2ppt` 把 PDF 投影片轉成可編輯的 PowerPoint (`.pptx`)。面向簡報型文件：能抽就抽原生文字與版面，掃描或圖片頁交給 OCR，並只在需要時重建被抹掉文字後方的背景。

## 成果展示

<p align="center">
  <img src="image/example_pdf.png" alt="範例 PDF 投影片" width="48%" />
  <img src="image/example_ppt.png" alt="轉換後的 PPT 投影片" width="48%" />
</p>

<p align="center"><em>左側是原始 PDF，右側是轉換後可編輯的 PowerPoint。</em></p>

## 功能概要

- 優先原生 PDF 解析；掃描／圖片頁使用 PaddleOCR
- 同行文字框合併，減少標題碎框，並盡量維持多欄版面
- 可選版面分析：區域類型、閱讀順序、真實項目符號、標題 placeholder（[`docs/layout-analysis.md`](docs/layout-analysis.md)）
- 數位 PDF 上的簡單向量圖與有格線表格可轉成原生 PPT 形狀／表格（[`docs/vector-nativization.md`](docs/vector-nativization.md)）
- 背景引擎：`opencv-fast`（建議預設）、`white-box`、可選 `lama-onnx`／`lama-pytorch`，以及雲端生成式後端（[`docs/background-engines.md`](docs/background-engines.md)）
- 可選的確定性 render-compare-refine 迴圈（`--refine`）：轉檔後對投影片重繪、比對、規則修正
- JSON 轉換報告、debug 產物，以及可選的 OCR 審框 UI

## 安裝

需求：Python 3.11 以上，建議 Linux。OCR 請使用獨立環境並固定 `numpy<2`（PaddleOCR / PaddleX 較穩定）。

```bash
cd pdf2ppt
conda create -n ppocr python=3.12 numpy=1.26.4 -y
conda activate ppocr
# 或使用 venv：python -m venv .venv && source .venv/bin/activate
python -m pip install -e .
```

若已有環境：

```bash
python -m pip install "numpy<2"
python -m pip install -e .
```

請一律使用本 repo 的 editable 安裝（`pip install -e .`），不要把 `PYTHONPATH` 指到 `build/lib/`——那是過期的 setuptools 殘留，且已列入 `.gitignore`。

選用：測試可安裝 `python -m pip install pytest`。OCR 模型預設放在 `./model`（可用 `--ocr-model-root` 覆寫）。

## 快速開始

```bash
pdf2ppt input.pdf output.pptx
pdf2ppt input.pdf output.pptx --report output.report.json --debug-dir output_debug
pdf2ppt input.pdf output.pptx --inpaint-engine opencv-fast

# 可選 P4 refine（需要 LibreOffice；若無 soffice 會警告並略過，不中斷轉檔）
pdf2ppt input.pdf output.pptx --refine --refine-iters 2
```

### Review UI（選用）

```bash
# Terminal 1 — API
cd pdf2ppt
conda activate ppocr   # 或：source .venv/bin/activate
python -m uvicorn pdf2ppt.api:app --host 127.0.0.1 --port 8008

# Terminal 2 — 前端
cd pdf2ppt/frontend
npm install            # 第一次或前端依賴變更後
npm run dev
```

瀏覽器開啟 `http://127.0.0.1:5173`。完整流程、job 儲存與 API 說明見
[`docs/review-ui.md`](docs/review-ui.md)。

## 架構

```text
PDF → 頁面分類（digital / scanned / hybrid）
    → 原生抽取與／或 OCR（含行合併、可選版面分析）
    → 可選向量／表格原生化
    → 條件式背景（elements / overlay / full-page）
    → PPTX + JSON 報告
    → 可選 --refine（render-compare-fix；需 LibreOffice）
```

| 元件 | 角色 |
| --- | --- |
| `src/pdf2ppt/cli.py` | CLI 入口 |
| `src/pdf2ppt/pipeline.py` | 轉換編排 |
| `src/pdf2ppt/api.py` | Review UI 後端 |
| `frontend/` | OCR 審框前端 |

背景策略摘要：能保留原生元素就保留；overlay 頁在 `auto` 下優先 `opencv-fast`，偵測到 LaMa runtime 時改用 `lama-onnx`，否則回退 `white-box`。細節見
[`docs/background-engines.md`](docs/background-engines.md)。

## 更多文件

| 文件 | 內容 |
| --- | --- |
| [`docs/cli-reference.md`](docs/cli-reference.md) | 參數、模式、輸出、限制 |
| [`docs/background-engines.md`](docs/background-engines.md) | 背景引擎與 `auto` 路由 |
| [`docs/review-ui.md`](docs/review-ui.md) | Review UI 與 API |
| [`docs/layout-analysis.md`](docs/layout-analysis.md) | Heuristic／PP-Structure 版面 |
| [`docs/vector-nativization.md`](docs/vector-nativization.md) | 形狀與表格原生化 |
| [`docs/generative-inpaint.md`](docs/generative-inpaint.md) | 雲端生成式後端 |
| [`docs/slide-quality.md`](docs/slide-quality.md) | 畫質／可編輯性評估 |

CI（`.github/workflows/ci.yml`）會跑 pytest、ruff + mypy、前端測試與建置，以及僅報告用的 slide-quality bench。

## 授權

MIT，詳見 [`LICENSE`](LICENSE)。
