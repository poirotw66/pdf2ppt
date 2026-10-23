# pdf2ppt

[![CI](https://github.com/poirotw66/pdf2ppt/actions/workflows/ci.yml/badge.svg)](https://github.com/poirotw66/pdf2ppt/actions/workflows/ci.yml)

[English](README.md) | **繁體中文**

把 PDF 投影片轉成**可編輯**的 PowerPoint (`.pptx`)。優先原生抽取，掃描頁走 OCR，可選版面／VLM 規劃，並只在抹字處重建背景。

## 成果展示

<p align="center">
  <img src="image/example_pdf.png" alt="範例 PDF 投影片" width="48%" />
  <img src="image/example_ppt.png" alt="轉換後的 PPT 投影片" width="48%" />
</p>

<p align="center"><em>左側是原始 PDF，右側是轉換後可編輯的 PowerPoint。</em></p>

## 功能

- 原生 PDF 抽取；掃描／圖片頁以 PaddleOCR 備援（需 `[ocr]`）
- 可選版面分析（heuristic／PP-Structure）與實驗性 Gemini 版面規劃器（美元上限、呼叫上限、斷路器、`--layout-planner-apply-grouping`）
- 向量／表格原生化：有格線表格、**無框線／斑馬紋**推斷、**疊在照片上的表格**（[`docs/vector-nativization.md`](docs/vector-nativization.md)）
- SlideIR agent 層：`emit-ir`／`from-ir` 與 HTTP（`POST /jobs`、`/emit-ir`、`/slideir.json`、`/slideir-assets.zip`、`POST /from-ir`）
- 可選 `--refine` render-compare 迴圈（需 LibreOffice）
- Review UI 支援**轉換前預覽**（疊加層、輕量編輯、縮圖、`baked_background`）
- 強化：`from-ir` 安全 zip 上限、分頁逾時（`PDF2PPT_PAGE_TIMEOUT_S`，預設 **120 秒**）、OCR 逾時通道降級、規劃器斷路器
- 資料夾批次：`pdf2ppt batch`（續跑、摘要、結束碼 **0／1／2**）
- 打包：pip／pipx extras `[ocr]` `[server]` `[dev]` `[all]`（paddlepaddle **3.2.0**）、`python -m pdf2ppt`、Docker Compose

文件索引：[`docs/README.md`](docs/README.md)。

## 快速開始

需求：Python **3.11** 以上，建議 Linux。使用 OCR 時請固定 `numpy<2`。

### pip／venv

```bash
cd pdf2ppt
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[ocr]"          # 轉檔 + OCR
# python -m pip install -e ".[dev]"    # 開發用：OCR + API + lint/pytest
pdf2ppt tests/fixtures/solid_background.pdf output.pptx
# 等價入口：python -m pdf2ppt …
```

### pipx

```bash
pipx install ".[ocr]"                   # 從本機 clone 安裝
pdf2ppt --help
```

### Extras

| Extra | 內容 |
| --- | --- |
| *（核心）* | 數位／原生 PDF → PPTX |
| `[ocr]` | PaddleOCR + **paddlepaddle==3.2.0**（3.3.x 會弄壞 CPU 上的 PP-OCRv5） |
| `[server]` | FastAPI Review UI 後端（`pdf2ppt-api`） |
| `[dev]` | `[ocr]` + `[server]` + ruff + mypy + pytest |
| `[all]` | `[ocr]` + `[server]` + CPU ONNX Runtime |
| `[cpu]`／`[gpu]` | `lama-onnx` 用的 ONNX Runtime |
| `[layout]` | 等同 `[ocr]`（PP-Structure 版面） |

OCR 模型預設在 `./model`（`--ocr-model-root`）。開發請用 `pip install -e ".[dev]"`，不要把 `PYTHONPATH` 指到 `build/lib/`。

### Docker

```bash
docker compose up --build
# Review UI：http://127.0.0.1:8080
# API health：http://127.0.0.1:8008/health
```

映像設定 `PDF2PPT_ROOT=/data`。OCR 模型使用具名 volume `pdf2ppt-models`（容器 **uid 10001** 可寫）。若改掛主機目錄：設 `PDF2PPT_MODEL_DIR=./model` 並 `chown 10001:10001`。Job 使用 `pdf2ppt-jobs`（沒有 `./output` bind）。

```bash
docker build -t pdf2ppt .
# 對齊主機 uid，才能寫入 bind mount（映像內 USER 為 10001）。
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/work" -w /work pdf2ppt \
  pdf2ppt tests/fixtures/solid_background.pdf /work/output.pptx
```

`GEMINI_API_KEY` 可選（僅規劃器／生成式 Gemini），必須執行時傳入（見 `.env.example`）。若要用 `--refine`，請以 `--build-arg INSTALL_LIBREOFFICE=1` 重建。

## 基本用法

```bash
pdf2ppt input.pdf output.pptx --report output.report.json --debug-dir output_debug
pdf2ppt input.pdf output.pptx --inpaint-engine opencv-fast

# 實驗性 refine（需 LibreOffice；若無 soffice 會警告並略過）
pdf2ppt input.pdf output.pptx --refine --refine-iters 2

# 實驗性 Gemini 版面規劃器（需 GEMINI_API_KEY；失敗時逐頁退回）
pdf2ppt input.pdf output.pptx --layout-planner gemini
pdf2ppt input.pdf output.pptx --layout-planner gemini --layout-planner-apply-grouping

# SlideIR round-trip
pdf2ppt emit-ir input.pdf deck.slideir.json
pdf2ppt from-ir deck.slideir.json rebuilt.pptx
```

完整旗標見 [`docs/cli-reference.md`](docs/cli-reference.md)。

### SlideIR（簡述）

Schema `1.0`；資產與 IR 同目錄，名為 `{ir}.assets/`。啟用 `--layout-planner gemini` 且成功時會套用角色／閱讀順序；**`groups` 只有加上 `--layout-planner-apply-grouping` 才會寫入**。HTTP（API `:8008`）：

| 端點 | 行為 |
| --- | --- |
| `POST /jobs` | 上傳 PDF／PNG／JPG → 建立 job |
| `POST /jobs/{id}/emit-ir` | 分析 → SlideIR |
| `GET /jobs/{id}/slideir.json` | 下載 IR |
| `GET /jobs/{id}/slideir-assets.zip` | 下載資產 zip |
| `POST /from-ir` | multipart：`ir_file`＋可選 `assets_zip` → PPTX |

細節與 zip 安全上限：[`docs/slideir.md`](docs/slideir.md)。

## 批次用法

```bash
pdf2ppt batch ./pdfs -o ./pptx_out
pdf2ppt batch ./pdfs -o ./pptx_out --recursive --resume --skip-existing
pdf2ppt batch ./pdfs -o ./pptx_out --fail-fast --jobs 1 --measure-quality
```

| 項目 | 行為 |
| --- | --- |
| 探索 | `*.pdf`／`*.PDF`；預設不遞迴；`--recursive` 鏡像子資料夾；略過隱藏名稱與已在輸出目錄下的 PDF |
| 隔離 | 單一壞檔不中止整批（除非 `--fail-fast`） |
| 續跑 | `--skip-existing`／`--resume` 保留有效 PPTX；中斷後請加 `--resume` 重跑 |
| 摘要 | `summary.csv`／`summary.json`／`summary.md`（每檔後原子寫入） |
| 平行 | `--jobs` 預設 **1**；`>1` 使用行程（OCR 很吃記憶體） |
| 結束碼 | **0** 全成功／略過 · **1** 至少一檔失敗 · **2** 用法／驗證錯誤 · **130** KeyboardInterrupt（部分摘要仍保留） |

單一 PDF 輸出與單檔模式一致，但同一行程連續轉多個 OCR 檔時，PaddleOCR 暖機可能造成文字色約 ±2/255 差異。**注意：** 原生 OCR 期間收到 SIGINT 時，行程可能以非零異常碼結束；`summary.*` 仍有效，請用 `--resume` 重跑。

## Review UI／API 概要

```bash
# Terminal 1
python -m pip install -e ".[ocr,server]"
pdf2ppt-api    # http://127.0.0.1:8008

# Terminal 2
cd frontend && npm install && npm run dev   # http://127.0.0.1:5173
```

或 `docker compose up --build` → UI 在 **8080**。

**轉換前預覽：** Build Preview → 編輯疊加層（排除／角色／raster fallback）→ 以 `from-ir` Convert。

| 項目 | 說明 |
| --- | --- |
| 端點 | `POST/GET …/preview`、`PUT …/preview-edits`、`POST …/from-ir`、縮圖 `GET …/pages/{n}.jpg?width=`（UI 用 **120**） |
| 規劃器 | Preview 強制關閉 layout planner（不需 Gemini） |
| `baked_background` | `overlay`／`full-page` 為 `true`；排除可能留下烘進像素 — 請用 raster fallback |
| **413** | preview-edits 本文超過 **1 MiB** |
| **409 `job-busy`** | Job 鎖約等 5 秒、最多 **4** 個等待者；UI 對 preview-edits 以 **1／2／4 秒**自動重試（不上傳／最終 convert／from-ir） |

完整流程：[`docs/review-ui.md`](docs/review-ui.md)。

## 設定（環境變數）

| 變數 | 預設 | 用途 |
| --- | --- | --- |
| `PDF2PPT_ROOT` | 自動（repo）／Docker `/data` | 專案根目錄（模型／相對路徑） |
| `PDF2PPT_JOB_ROOT` | `.pdf2ppt_jobs` | Review UI job 儲存 |
| `PDF2PPT_JOB_RETENTION_HOURS` | `24` | Job 保留時數（`<=0` 不逾期） |
| `PDF2PPT_CORS_ORIGINS` | Vite `5173` | 逗號分隔；`*` 會被拒絕（有 credentials）；僅逗號則回退預設 |
| `PDF2PPT_PAGE_TIMEOUT_S` | `120` | 分頁 OCR／版面逾時（`<=0` 關閉） |
| `PDF2PPT_ZIP_MAX_COMPRESSED_BYTES` | `50 MiB` | `from-ir` 資產 zip |
| `PDF2PPT_ZIP_MAX_UNCOMPRESSED_BYTES` | `200 MiB` | … |
| `PDF2PPT_ZIP_MAX_ENTRIES` | `10000` | … |
| `PDF2PPT_ZIP_MAX_ENTRY_BYTES` | `50 MiB` | … |
| `PDF2PPT_ZIP_MAX_COMPRESSION_RATIO` | `100` | … |
| `PDF2PPT_LAYOUT_PLANNER_MODEL` | `gemini-3.1-flash-lite` | 規劃器模型覆寫 |
| `PDF2PPT_LAYOUT_PLANNER_INPUT_USD_PER_MTOK`／`_OUTPUT_…` | `0.25`／`1.50` | 成本計算 |
| `GEMINI_API_KEY`／`GOOGLE_API_KEY` | 未設 | 規劃器／生成式 Gemini |
| `OPENAI_API_KEY` | 未設 | 生成式 OpenAI |
| `PDF2PPT_GEMINI_IMAGE_MODEL`／`PDF2PPT_OPENAI_IMAGE_MODEL` | 產品預設 | 生成式影像模型 |
| `PDF2PPT_API_PORT`／`PDF2PPT_UI_PORT` | `8008`／`8080` | Compose 主機埠 |
| `PDF2PPT_MODEL_DIR` | 具名 volume `pdf2ppt-models` | Compose 模型掛載 |

請勿提交真實金鑰。範本：[`.env.example`](.env.example)。

**版面規劃器煞車（CLI 預設）：** 最多呼叫 **30** 次、成本上限 **$1.0**、逾時 **45 秒**、並行 **2**、連續失敗 **2** 次斷路（認證失敗立即斷路）。細節：[`docs/layout-planner.md`](docs/layout-planner.md)。

## 開發／測試

```bash
python -m pip install -e ".[dev]"
make lint          # ruff + mypy
make test          # pytest
make test-frontend # 前端單元測試
make help          # 所有快捷指令
```

CI（`.github/workflows/ci.yml`）：pytest、ruff + mypy、前端測試／建置、wheel 煙霧測試、僅報告用的 slide-quality bench。

## 已知限制

- **旋轉頁（90／180／270°）：** SlideIR／疊加元素框使用未旋轉座標，可能與預覽圖對不齊（與 CLI PPTX 相同）。這類頁請優先用 raster fallback。
- **寬而不換行的文字：** LibreOffice 常忽略 placeholder 的 `wrap=none`；標題加寬與 OCR 單行啟發式可緩解，但無法完全對齊 PowerPoint 重排行為。
- **OCR 逾時通道：** 分頁逾時後會更換 OCR worker 執行緒；Paddle 具執行緒親和性，約 30–40% 的更換可能拋出原生 `std::exception`。pipeline 會對該頁降級為空 OCR／raster fallback，而不是整份失敗。建議維持預設 120 秒逾時。
- **Refine 時間預留：** 剩餘預算低於已核發的 soffice 預留（`max(5s, 3×` 最近單頁渲染））時不啟動 trial，避免極短殘餘時間拖垮健康轉檔。預設整份預算 **90 秒**；次數因機器而異。
- **批次在原生 OCR 期間收到 SIGINT：** 摘要檔仍有效；行程可能以非零異常碼結束 — 請用 `--resume` 重跑。

## 疑難排解／FAQ

| 狀況 | 處理 |
| --- | --- |
| `Failed to import PaddleOCR`／缺少 OCR extra | `pip install 'pdf2ppt[ocr]'`；維持 `numpy<2` |
| 升級 Paddle 後 OCR 損壞 | 固定 **paddlepaddle==3.2.0**（3.3.x 會弄壞 CPU 上的 PP-OCRv5） |
| `--refine` 被略過 | 安裝 LibreOffice Impress（`soffice` 需在 PATH）；Docker：`INSTALL_LIBREOFFICE=1` |
| 分頁／OCR 逾時、卡住後變空 OCR | 提高 `--page-timeout-s`／`PDF2PPT_PAGE_TIMEOUT_S`；見上方 OCR 通道限制 |
| `409 job-busy` | 另有請求佔用此 job；UI 會重試 preview-edits（1／2／4 秒）。請稍候或手動重試 Convert |
| preview-edits 回 `413` | 本文超過 **1 MiB** — 縮小編輯 payload |
| Docker 權限／模型目錄不可寫 | 用具名 volume `pdf2ppt-models`，或對主機 bind `chown 10001:10001`；CLI：`docker run --user "$(id -u):$(id -g)"` |
| `PDF2PPT_CORS_ORIGINS=*` | 啟用 credentials 時會被拒絕；請列出明確來源 |
| fastapi／pydantic 衝突 | 乾淨 venv + `[server]` |
| CJK／refine 指標異常 | 安裝 `fonts-noto-cjk` |

## 文件

| 文件 | 內容 |
| --- | --- |
| [`docs/README.md`](docs/README.md) | **所有文件索引** |
| [`docs/quickstart.md`](docs/quickstart.md) | 安裝、首次轉檔、Docker |
| [`docs/cli-reference.md`](docs/cli-reference.md) | 旗標、批次、結束碼、OCR 逾時 |
| [`docs/review-ui.md`](docs/review-ui.md) | Review UI + API + 預覽 |
| [`docs/slideir.md`](docs/slideir.md) | SlideIR emit-ir／from-ir |
| [`docs/layout-planner.md`](docs/layout-planner.md) | Gemini 規劃器上限／斷路器 |
| [`docs/layout-analysis.md`](docs/layout-analysis.md) | Heuristic／PP-Structure |
| [`docs/vector-nativization.md`](docs/vector-nativization.md) | 形狀與無框線／斑馬紋表格 |
| [`docs/background-engines.md`](docs/background-engines.md) | 背景引擎／`auto` |
| [`docs/generative-inpaint.md`](docs/generative-inpaint.md) | 雲端生成式後端 |
| [`docs/slide-quality.md`](docs/slide-quality.md) | 畫質／refine 評估 |
| [`CHANGELOG.md`](CHANGELOG.md) | 版本紀錄 |

## 授權

MIT，詳見 [`LICENSE`](LICENSE)。
