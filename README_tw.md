# pdf2ppt

`pdf2ppt` 用來把 PDF 轉成可編輯的 PowerPoint (`.pptx`)。

這個專案主要針對簡報型文件，採用混合式轉換流程：

- 能直接從 PDF 抽文字與版面的情況，優先使用原生解析。
- 掃描頁或圖片型頁面，再交給 PaddleOCR。
- 先判斷頁面類型（`digital`、`scanned`、`hybrid`），再決定最合適的輸出方式。
- 只有在需要時才做背景重建，而不是一律整頁抹字。
- OCR 文字會額外估算字級、字色與粗體。

## 主要功能

- 將 PDF 轉成可編輯的 PPTX。
- 優先保留原生 PDF 文字。
- 將掃描頁文字重建為 PowerPoint 文字方塊。
- 支援多種背景重建引擎：
  - `white-box`
  - `opencv-fast`
  - `diffusion-local`
  - `auto` 自動路由
- 每次轉換都可輸出 JSON 報告。
- 可輸出逐頁 debug 圖與分析檔，方便檢查 OCR 與背景處理結果。

## 環境需求

- Python 3.11 以上
- 建議使用 Linux
- 依賴套件定義於 `pyproject.toml`
- OCR 需要：
  - PaddleOCR 執行環境與模型下載
- 若要使用本地 diffusion inpainting，建議：
  - NVIDIA GPU
  - `iopaint` 或其他相容的本地後端

## 安裝方式

建立並啟用 Python 環境後，使用 editable mode 安裝：

```bash
python -m pip install -e .
```

如果你要使用本地 diffusion inpainting，請另外安裝並確認後端可正常執行。本專案目前已驗證過 `iopaint` 流程。

## 快速開始

基本轉換：

```bash
pdf2ppt input.pdf output.pptx
```

指定 report 輸出路徑：

```bash
pdf2ppt input.pdf output.pptx --report output.report.json
```

輸出 debug 檔案：

```bash
pdf2ppt input.pdf output.pptx --debug-dir output_debug
```

使用 OpenCV 快速背景重建：

```bash
pdf2ppt input.pdf output.pptx \
  --inpaint-engine opencv-fast \
  --report output.report.json \
  --debug-dir output_debug
```

使用本地 diffusion 後端：

```bash
pdf2ppt input.pdf output.pptx \
  --inpaint-engine diffusion-local \
  --diffusion-command iopaint \
  --diffusion-model runwayml/stable-diffusion-inpainting \
  --diffusion-device cuda \
  --report output.report.json
```

## CLI 參數

主要參數：

- `input_pdf`：輸入 PDF 路徑
- `output_pptx`：輸出 PPTX 路徑
- `--report`：JSON report 輸出路徑
- `--mode`：`editable`、`fidelity`、`fast`
- `--lang`：PaddleOCR 語言代碼，預設為 `ch`
- `--dpi`：OCR 與背景生成的渲染 DPI
- `--debug-dir`：逐頁 debug 圖與分析檔輸出資料夾
- `--enable-doc-unwarping`：啟用 PaddleOCR UVDoc 去扭曲

背景重建相關：

- `--inpaint-engine`：`auto`、`white-box`、`opencv-fast`、`diffusion-local`
- `--inpaint-padding-px`：在 inpainting 前擴張文字遮罩
- `--inpaint-max-area-ratio`：當遮罩面積太大時，強制改用 white-box

本地 diffusion 參數：

- `--diffusion-command`：呼叫後端的 CLI 指令，預設 `iopaint`
- `--diffusion-model`：傳給後端的模型名稱
- `--diffusion-device`：`cuda` 或 `cpu`
- `--diffusion-max-crop-edge`：送進後端的最大裁切邊長
- `--diffusion-complexity-threshold`：`auto` 模式下判斷複雜背景的門檻

## 轉換模式

- `editable`：在可編輯性與視覺相似度之間取得平衡
- `fidelity`：更保守，優先維持視覺接近原稿
- `fast`：用較低成本快速產出結果

## 背景重建策略

本專案不會一律對整頁做去字。

實際上會根據頁面狀況選擇不同模式：

- `elements`：盡量保留可編輯元素，不生成整頁背景圖
- `overlay`：只重建可編輯文字下方的背景
- `full-page`：當風險太高時，回退成整頁圖片

在 `overlay` 模式下，`auto` 會依條件選擇：

- `opencv-fast`：適合較簡單背景
- `diffusion-local`：適合較複雜背景，且需後端可用
- `white-box`：在遮罩過大或後端不可用時作為保底方案

## 輸出檔案

常見輸出如下：

- `output.pptx`：可編輯簡報
- `output.report.json`：結構化轉換報告
- `output_debug/`：可選的 debug 輸出

JSON report 會包含：

- 頁面類型判斷
- 背景模式
- 品質分數
- 實際使用的背景重建引擎
- OCR / 原生文字區塊與估算樣式

## 注意事項與限制

- OCR 頁面的重建本質上仍是估算，不是完整語意還原。
- 複雜圖表與向量圖目前較偏向保留外觀，而非完整還原成可編輯圖表物件。
- OCR 的粗體與顏色恢復屬 heuristic 判斷。
- 本地 diffusion 品質高度依賴後端可用性、GPU 記憶體與模型選擇。

## 開發

執行測試：

```bash
python -m pytest -q
```

主要檔案：

- CLI：`src/pdf2ppt/cli.py`
- 核心流程：`src/pdf2ppt/pipeline.py`
- 資料模型：`src/pdf2ppt/models.py`

## 文件語言版本

- English：`README.md`
- 繁體中文：`README_tw.md`
