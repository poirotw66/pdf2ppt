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
- 實驗性可選 VLM 版面規劃器（`--layout-planner gemini`）：結構化角色、段落合併、字體家族提示（[`docs/layout-planner.md`](docs/layout-planner.md)）
- 數位 PDF 上的簡單向量圖與有格線表格可轉成原生 PPT 形狀／表格（[`docs/vector-nativization.md`](docs/vector-nativization.md)）
- SlideIR agent 層：輸出版本化 JSON IR（`emit-ir`），再單獨從 IR 渲染 PPTX（`from-ir`），讓 agent 可 round-trip 而無須重跑 PDF pipeline（[`docs/slideir.md`](docs/slideir.md)）
- 背景引擎：`opencv-fast`（建議預設）、`white-box`、可選 `lama-onnx`／`lama-pytorch`，以及雲端生成式後端（[`docs/background-engines.md`](docs/background-engines.md)）
- 可選的確定性 render-compare-refine 迴圈（`--refine`，實驗性）：依預算選擇完整或部分 baseline、連續嚴重度排序、剩餘時間二次分配、接受修正後重設該元素嘗試次數。真實 soffice 的頁數／接受數會因機器而異。
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

# 實驗性可選 P4 refine（需要 LibreOffice；若無 soffice 會警告並略過，不中斷轉檔）
pdf2ppt input.pdf output.pptx --refine --refine-iters 2

# 實驗性可選 Phase-3 VLM 版面規劃器（需 GEMINI_API_KEY；失敗時逐頁退回幾何啟發式）
pdf2ppt input.pdf output.pptx --layout-planner gemini

# Agent / SlideIR round-trip（先不產生 PPTX，再用 from-ir 重建）
pdf2ppt emit-ir input.pdf deck.slideir.json
pdf2ppt from-ir deck.slideir.json rebuilt.pptx
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

### 轉換前預覽（Preview before convert）

在 Review UI 上傳 PDF 後切換到 **Preview before convert**：

1. 按 **Build Preview** — 執行分析並產生 SlideIR（預設關閉 layout planner，不需 Gemini API），以原始頁面圖疊加文字區塊（角色／閱讀順序）、表格／儲存格、圖片與向量形狀。
2. 可切換疊加層、排除個別元素、更改文字角色（`title`／`body`／`header`／`footer`／`page_number`），或將整頁標為 **raster fallback**。
3. 按 **Convert** — 先儲存編輯，再以 `from-ir` 產出 PPTX，預覽所見即下載所得。

相關 API：`POST /jobs/{id}/preview`、`GET /jobs/{id}/preview`、`PUT /jobs/{id}/preview-edits`、`POST /jobs/{id}/from-ir`。未使用此流程時，預設 CLI 輸出行為不變。

**限制說明**

- **排除 vs 烘進背景：** `overlay`／`full-page` 頁的照片與許多形狀已畫進背景圖。排除 IR 的 shape／image 只會拿掉原生物件，背景像素仍在。若要整頁點陣圖，請用 **raster fallback**。這類頁面的預覽會帶 `baked_background: true`。
- **重新分析會清空編輯：** 元素 id 依位置產生（`layout_text_1_N`）。再次 Build Preview 會刪除已存的 `preview_edits.json`，避免 OCR 框變更後 Convert 因未知 id 失敗。
- **旋轉頁（既有問題）：** 旋轉 90／180／270° 的頁，SlideIR bbox 可能與預覽圖對不齊（與 `main` 上 CLI PPTX 相同）。短期可對這類頁使用 raster fallback。

## SlideIR

SlideIR 是 pdf2ppt 的**版本化 JSON 中介表示**（schema `1.0`）。Agent 可把它當成確定性的證據層：對 PDF 只分析一次（`emit-ir`）、編輯結構欄位，再渲染（`from-ir`），不必重跑整條 PDF pipeline。Schema 檔：`src/pdf2ppt/schemas/slideir.schema.json`；序列化／反序列化：`pdf2ppt.ir_io`。完整說明見 [`docs/slideir.md`](docs/slideir.md)。

### 主要欄位

| 層級 | 欄位 | 說明 |
| --- | --- | --- |
| Deck | `schema_version` | 目前為 `"1.0"`；未知 major 應拒絕 |
| Deck | `slide_width_pt` / `slide_height_pt` | 投影片尺寸（PDF 點） |
| Deck | `assets_dir` | 相對 IR 檔的資產目錄名（預設 `{ir_filename}.assets`） |
| Deck | `source` | 可選來源路徑與頁數 |
| Deck | `plan_metadata` | 可選；只要啟用 `--layout-planner` 就會寫入（含頁面 fallback 時） |
| Page | `page_number` / `page_kind` / `background_mode` | 頁碼、digital／scanned／hybrid、背景模式 |
| Page | `text_blocks` | 文字框（`id`、`bbox`、`text`、樣式、`block_role`、`reading_order`、`paragraphs`…） |
| Page | `shapes` / `tables` / `images` | 原生形狀、表格、圖片裁切（圖片以相對路徑參照資產） |
| Page | `layout_regions` | 版面區域 |
| Page | `groups` | 可選元素分組（`id`、`member_ids`、可選 `bbox`／`confidence`）；只有加上 `--layout-planner-apply-grouping` 時才會寫入 |
| Page | `plan_metadata` | 可選頁面層級規劃器中繼資料（engine、model、accepted groups 等） |
| Page | `background` | 可選背景圖資產參照 |
| Page | `reading_order` | 可選的明確元素 id 順序 |

資產採**相對路徑、與 IR 同目錄**（非 base64）：

```text
deck.slideir.json
deck.slideir.json.assets/
  pages/
    001/
      background.png  # 可選；依影像資料可能是 .png／.jpg
      image_000.png
```

文字在 emit 時視為 ground truth（`locked: true`）。編輯時優先改 `block_role`、`reading_order`、渲染器已支援的樣式；不要虛構文字，也不要讓資產路徑指向 `assets_dir` 以外。啟用 `--layout-planner gemini` 且成功時，pipeline 會寫入 `pages[].plan_metadata`，並把角色與閱讀順序套用到 `text_blocks` 後再寫入 IR；段落合併與 `pages[].groups` 只有加上 `--layout-planner-apply-grouping` 時才會產生。`from-ir` 依 IR 內既有結構渲染 PPTX。

### 從 PDF 匯出 IR

```bash
# 只輸出 IR（不產生 PPTX）
pdf2ppt emit-ir input.pdf deck.slideir.json
# 等價寫法：
pdf2ppt input.pdf --emit-ir deck.slideir.json --ir-only

# 轉 PPTX 同時寫出 IR
pdf2ppt input.pdf output.pptx --emit-ir deck.slideir.json
```

分析相關旗標（`--vector-mode`、`--layout-engine`、OCR／inpaint 等）對 `emit-ir` 與一般 convert 相同。

### 編輯後重建 PPTX

```bash
# 子指令
pdf2ppt from-ir deck.slideir.json rebuilt.pptx
# 等價旗標寫法：
pdf2ppt --from-ir deck.slideir.json -o rebuilt.pptx
```

HTTP（Review UI 後端，預設 `http://127.0.0.1:8008`）：

| 端點 | 行為 |
| --- | --- |
| `POST /jobs/{id}/emit-ir` | 分析 → 只寫 SlideIR |
| `POST /jobs/{id}/preview` | 分析 → SlideIR（強制關閉 planner）＋疊加預覽 JSON |
| `GET /jobs/{id}/preview` | 依現有 SlideIR＋已存編輯回傳疊加預覽 |
| `PUT /jobs/{id}/preview-edits` | 儲存排除／角色／raster-fallback 編輯 |
| `POST /jobs/{id}/from-ir` | 套用編輯後，將 job 的 SlideIR 渲染成 PPTX |
| `GET /jobs/{id}/slideir.json` | 下載 IR JSON |
| `GET /jobs/{id}/slideir-assets.zip` | 下載資產目錄 zip |
| `POST /from-ir` | multipart：`ir_file`（必填）＋可選 `assets_zip` → 回傳 PPTX |

請先以 `POST /jobs`（multipart 上傳 PDF）建立 job；接著 `POST /jobs/{id}/emit-ir` 接受 JSON body（`{}` 即使用預設）。從 API 下載的 IR，其資產目錄名為 `output.slideir.json.assets`，因此若要用 CLI `from-ir`，必須解壓到同名資料夾（`POST /from-ir` 會自動處理）。

```bash
curl -o rebuilt.pptx \
  -F "ir_file=@deck.slideir.json;type=application/json" \
  -F "assets_zip=@deck.slideir.assets.zip;type=application/zip" \
  http://127.0.0.1:8008/from-ir
```

### 簡短範例

```bash
pdf2ppt emit-ir tests/fixtures/solid_background.pdf /tmp/deck.slideir.json
# 可編輯 /tmp/deck.slideir.json（例如調整 text_blocks[].block_role / reading_order）
pdf2ppt from-ir /tmp/deck.slideir.json /tmp/deck.pptx
```

## 架構

```text
PDF → 頁面分類（digital / scanned / hybrid）
    → 原生抽取與／或 OCR（含行合併、可選版面分析）
    → 可選向量／表格原生化
    → 條件式背景（elements / overlay / full-page）
    → PPTX + JSON 報告
    → 可選 --refine（實驗性 render-compare-fix；需 LibreOffice）
    → 可選 SlideIR emit / from-ir 渲染
```

| 元件 | 角色 |
| --- | --- |
| `src/pdf2ppt/cli.py` | CLI 入口 |
| `src/pdf2ppt/pipeline.py` | 轉換編排 |
| `src/pdf2ppt/ir_io.py` | SlideIR 序列化／反序列化 |
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
| [`docs/slideir.md`](docs/slideir.md) | SlideIR emit-ir／from-ir agent 層 |
| [`docs/generative-inpaint.md`](docs/generative-inpaint.md) | 雲端生成式後端 |
| [`docs/slide-quality.md`](docs/slide-quality.md) | 畫質／可編輯性評估 |

CI（`.github/workflows/ci.yml`）會跑 pytest、ruff + mypy、前端測試與建置，以及僅報告用的 slide-quality bench。

## 授權

MIT，詳見 [`LICENSE`](LICENSE)。
