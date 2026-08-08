# pdf2ppt 優化路徑（Optimization Roadmap）

本文件規劃 v1 之後的優化順序。與 `spec.md`（產品規格）、`plan.md`（前後端工作流）互補：
spec 說「要做什麼」，本文件說「先做哪一個、怎麼驗收」。

---

## 1. 現況摘要

v1 功能已完整落地：CLI + FastAPI + React 審框前端，後端約 8,100 行 Python、167 個測試，
CI 同時跑後端 pytest 與前端 vitest + build。以下是本規劃要處理的實際缺口。

### 1.1 品質無法量化（最根本的問題）

`spec.md` §9 自承「沒有品質分級與回退設計」。目前**沒有任何客觀方式回答「這次改動有沒有讓輸出變好」**：

- `tests/fixtures/` 只有一個 README，沒有任何實際樣本 PDF
- 沒有 inpainting 品質度量，只有 `debug_artifacts.py` 產生的人眼檢查用圖
- `examples/outputs/` 下有 `rag_opencv.pptx` / `rag_opencv_v2.pptx` / `rag_diffusion.pptx`，
  但只是歷史產出，沒有對應的量化比較

**後果**：任何引擎替換、參數調整、模型升級都無法驗收，只能靠主觀感覺。
這是所有其他優化的前置條件。

### 1.2 生成式去字已經整合，但實質不可用

專案已有 LaMa（生成式 inpainting，非傳統演算法），但三個原因讓它幾乎不會被啟用：

| 問題 | 位置 | 影響 |
|------|------|------|
| `auto` 路由永遠不會選到 LaMa | `inpainting_overlay.py:938-953` | 預設使用者只會拿到 `opencv-fast` |
| 大遮罩退回 `white-box`（整塊塗白） | `inpainting_overlay.py:952` | 最需要生成模型的情境，反而得到最差結果 |
| ONNX 版硬性要求 CUDAExecutionProvider | `inpainting_engines.py:983` | 無 GPU 直接報錯，沒有 CPU 路徑 |
| PyTorch 版靠 subprocess 跨 conda env | `inpainting_engines.py:866-963`、`scripts/setup_lama_env.sh` | 需 clone 官方 repo、打 patch、建獨立環境，並釘死 `pytorch-lightning==1.2.9`、`kornia==0.5.0`、`albumentations==0.5.2`（2021 年版本） |

### 1.3 傳統演算法已接近調參極限

`OpenCvFastInpaintingEngine.__init__`（`inpainting_engines.py:166-190`）有 **20 個超參數**，
`inpainting_engines.py` 全檔 **2,605 行**。繼續往上疊啟發式的邊際效益正在遞減。

### 1.4 版面分析未實作

`spec.md:14` 標記 Paddle Layout Analysis「尚未實作」，規劃於 v1.5。
目前以 OCR det + 幾何行合併代替，因此：

- 複雜閱讀順序、表格／圖片區分能力有限
- `block_analysis.py:180` 對所有 OCR 區塊硬寫 `font_family=None`，
  最終全部退到 `ppt_render.py:67` 的 `default_font_family(script)` —— OCR 頁面完全失去字體識別

### 1.5 工程健康度

- **完全沒有靜態檢查**：`pyproject.toml` 無 ruff/mypy，`frontend/package.json` 無 eslint，
  CI 只跑 pytest + vitest + build
- 依賴釘死偏舊：`numpy<2`、`pydantic>=2.13.2,<2.14`

---

## 2. 排序原則

1. **先能衡量，再談改善**：無法驗收的優化不排進來
2. **先讓既有資產可用，再引入新資產**：LaMa 已經在 repo 裡，讓它能跑的成本遠低於接新模型
3. **不破壞離線可用性**：任何外部 API 都必須是 opt-in，未設定時走原本的本地路徑
4. **預設路徑保持確定性**：非決定性的生成模型不進預設 pipeline，只做顯式選用

---

## Phase 0 — 建立品質基準（前置，阻擋所有後續階段）

**目標**：讓「這次改動有沒有變好」變成可以跑指令回答的問題。

### 0.1 建立樣本語料

在 `tests/fixtures/` 下建立小型、可版控的樣本 PDF，涵蓋各引擎會分歧的情境：

| 類別 | 為什麼要 |
|------|---------|
| 純色背景 + 文字 | `opencv-fast` 的 prefill 路徑基準線 |
| 線性／放射漸層背景 | 測 `smooth_gradient_*` 那組參數 |
| 幾何色塊、圖表 | 測 `_restore_structural_line_regions` 的線條保護 |
| 照片背景 + 疊字 | LaMa 相對 Telea 的主要優勢場景 |
| 深色底淺色字 | 測 `white-box` 退化的嚴重程度 |
| 表格 | 版面分析階段（Phase 3）的基準 |
| 大遮罩頁（面積比 > 0.12） | 直接命中 `auto` 退回 white-box 的路徑 |

樣本需**體積小、可重現**，避免版控膨脹；`.gitignore` 已有 `!tests/fixtures/**` 白名單。

### 0.2 定義指標

實作 `scripts/bench_inpaint.py`，對每個樣本 × 每個引擎產出：

- **殘留文字偵測率**：修補後區域重跑一次 OCR det，理想值為 0。
  這是「去字乾淨度」最直接的代理指標，且可完全自動化。
- **遮罩邊界不連續度**：遮罩外緣 ring 與修補區交界的梯度／色差。
  偵測「補得出來但有明顯接縫」。
- **遮罩外像素零變動斷言**：非遮罩區必須逐像素等同原圖。
  這是回歸護欄，Phase 1 和 Phase 4 都會依賴它。
- **每頁耗時**：引擎替換的成本面。

輸出格式：JSON（進 CI 比較）+ 並排對照圖（人眼複核）。

### 0.3 CI 接上非阻斷式靜態檢查

同時把 ruff + mypy 加進 `pyproject.toml` 與 CI，但**先設為 report-only 不阻斷**，
並建立 baseline。目的是在 Phase 1 的大幅重構期間就開始收集訊號，而不是重構完才發現問題。
Phase 2 再收緊為阻斷式。

**驗收**：`python scripts/bench_inpaint.py` 能對全部樣本跑完 `white-box` / `opencv-fast`
兩個引擎並輸出可比較的 JSON。

**規模**：中。**風險**：低（純新增，不動既有邏輯）。

---

## Phase 1 — 讓生成式去字變成預設可用

**目標**：把 LaMa 從「理論上支援」變成「裝好就能用，且 auto 會選它」。
這是投入產出比最高的一階段。

### 1.1 ONNX 路徑放寬到 CPU

- `inpainting_engines.py:983` 目前 provider 不符即拋 `BackgroundInpaintingError`，
  改為允許回落 `CPUExecutionProvider`
- `pyproject.toml` 的 optional extras 目前只有 `onnxruntime-gpu`，補上 CPU 版
- 引擎命名相應調整（`lama-onnx-cuda` → `lama-onnx`，保留舊名為 alias 避免破壞既有指令）

### 1.2 in-process 取代 subprocess

用 TorchScript / ONNX 版 big-lama 在主環境內直接推論，移除：

- `scripts/setup_lama_env.sh`、`scripts/apply_lama_patches.sh`、`scripts/lama_patches/`
- `inpainting_engines.py` 中 `_get_lama_pytorch_worker` / `_read_lama_pytorch_response` /
  `_build_lama_subprocess_env` / `_validate_lama_pytorch_runtime` /
  `_resolve_lama_python_executable` / `_default_lama_python_candidates` 等一整套跨進程管理

安裝流程從「clone repo + 打 patch + 建 conda env」變成 `pip install`。
預估可移除 500–600 行純維運程式碼。

### 1.3 拆分 `inpainting_engines.py`

在 1.2 刪掉 subprocess 那一段之後再拆（先刪再拆，搬移量較小）：

```
inpainting/
  base.py        # BackgroundInpaintingEngine、BackgroundRenderResult、例外
  white_box.py
  opencv_fast.py # 含 20 個超參數與 prefill/residual 邏輯
  lama.py        # ONNX + TorchScript
  patching.py    # patch group 切分與合成（_build_lama_patch_groups 等）
  compositing.py # _composite_lama_restoration_* 系列
```

### 1.4 把 LaMa 納入 auto 路由

修改 `inpainting_overlay.py:938-953`：偵測到可用 LaMa runtime 時，
大遮罩情境走 LaMa 而非 `white-box`。無 runtime 時維持現有行為完全不變。

**驗收**（全部以 Phase 0 的指標為準）：

- 大遮罩樣本的殘留文字偵測率與邊界不連續度，相對 `white-box` 有可量化改善
- 無 GPU 環境下 `lama-onnx` 能跑完（允許較慢）
- 遮罩外像素零變動斷言全數通過
- 既有 167 個測試不回歸

**規模**：大。**風險**：中（動到核心路徑，但有 Phase 0 護欄）。

---

## Phase 2 — 工程健康度收尾

- ruff + mypy 從 report-only 轉為 CI 阻斷
- frontend 加 eslint（目前無任何 lint script）
- 評估解除 `numpy<2` 釘選（Phase 1 移除 LaMa 舊環境後，這個限制的主要來源就消失了）
- 前端 React 18 / Vite 5 升級評估

**規模**：中。**風險**：低。可與 Phase 3 並行。

---

## Phase 3 — 補上版面分析（v1.5）

**目標**：關閉 `spec.md:14` 的缺口。**對最終 PPTX 品質的影響可能大於 Phase 1**，
因為它直接決定文字結構與可編輯性，而非背景保真度。

由於本階段觸及的模組（`block_analysis.py`、`ocr.py`）與 Phase 1（`inpainting_*`）**不重疊**，
兩者可並行推進。

### 3.1 技術選型：Paddle Layout vs VLM

| 方案 | 優點 | 缺點 |
|------|------|------|
| Paddle Layout Analysis | 本地、確定性、與現有 PP-OCRv5 同生態 | 再多一組模型權重與依賴；類別體系固定 |
| VLM（如 Gemini 3.5 Flash-Lite 這類輕量理解型模型） | 輸出是結構化 JSON 可用 schema 驗證；標題／頁眉頁尾判斷彈性高；無需本地權重 | 需外部 API、需 opt-in、有延遲與成本 |

**建議**：以 VLM 先做 spike 驗證效果，但**必須設計成 opt-in**——未設 API key 時
完全走現有幾何啟發式路徑。若效果顯著再考慮是否補本地方案。

關鍵前提：本階段 VLM 的輸出是**結構化資料，不是像素**，因此不會有幻覺污染畫面、
不會有解析度損失，失敗時可優雅退回既有邏輯。這與 Phase 4 的性質完全不同。

### 3.2 順帶處理字體識別

`block_analysis.py:180` 的 `font_family=None` 在本階段一併解決——
版面分析階段已經在看頁面影像，可同時給出「黑體／明體／圓體」等級的字體家族判斷。

**驗收**：標題判斷準確率、頁眉頁尾過濾率、閱讀順序正確率（需在 Phase 0 語料上補標註）。

**規模**：大。**風險**：中。

---

## Phase 4 — 雲端影像模型作為手動重修（顯式 opt-in）

**目標**：處理 LaMa 也補不好的複雜照片背景，即目前退化為 `white-box` 的最差情境。

### 4.1 可行性結論

Gemini 3.1 Flash-Lite Image 這類模型**可以**接進來，因為既有架構已經解決了兩個主要障礙：

- **解析度**：`DEFAULT_LAMA_PATCH_MAX_SIDE_PX = 512`（`inpainting_engines.py:76`），
  patch 路徑本來就不送整頁，512px 的 crop 遠低於該模型 1K 的輸出上限
- **遮罩條件**：`_composite_lama_restoration_with_inference_mask`（`:1325`）
  已強制遮罩外像素取自原圖，即使模型重繪整個 crop，合成階段也會還原

實作上就是新增一個 `BackgroundInpaintingEngine`（介面僅 `inpaint(page_image, mask_image) -> Image`），
重用既有 patch + composite 機制。

### 4.2 但不能進預設 pipeline

| 限制 | 說明 |
|------|------|
| 成本 | 一頁約 10 個 patch group，30 頁簡報 ≈ 300 次呼叫。以 Flash Image 1K $0.067/張為上界估算約 $20/份（Lite 更便宜，但確切價格需查證） |
| 延遲 | 300 次呼叫即使並行也是分鐘級，對照 `opencv-fast` 的秒級 |
| patch 間不一致 | 生成模型非決定性，相鄰 patch 可能對背景給出不同解釋而產生接縫；官方亦說明該模型「未針對多輪連續編輯最佳化」 |
| 幻覺 | 300 次呼叫即 300 次把挖掉的文字「補」回去的機會 |
| 隱私 | 專案目前零外部依賴，逐頁上傳第三方是政策級決定 |

### 4.3 設計約束

- 與 `lama-*` 同級的顯式 opt-in，**不進 `auto`、不進預設**
- 主要入口是**審框 UI 的單頁「高品質重修」按鈕**，人在迴路裡即時複核幻覺與接縫
- 提供呼叫次數上限參數作為成本煞車
- 未設 API key 時對現有行為零影響

如此三個主要限制都被壓在可接受範圍：成本從「每份 $20」變成「使用者主動點的那幾頁」。

**規模**：中。**風險**：中（外部依賴 + 成本）。

---

## Phase 5 — 模型升級探索

在 Phase 0 的度量到位、Phase 1 的引擎介面拆乾淨之後，替換模型才是低成本操作：

- **MI-GAN**（ICCV 2023）：專為端上設計，比 LaMa 快數倍、有現成 ONNX，品質相當或更好，
  是目前最務實的 LaMa 替代品
- **LaMa refinement**：高解析度精修流程，對投影片這種大尺寸圖有感

**驗收**：直接沿用 Phase 0 的指標做 A/B。

---

## 6. 已評估但不採用的選項

記錄於此以免重複討論。

| 選項 | 不採用的理由 |
|------|------------|
| Diffusion 作為預設去字引擎 | README 記載 `diffusion-local` 曾實作後移除（太慢、不一致）。更關鍵的是**幻覺文字**——在挖空的文字區生成出看似文字的內容，對轉檔工具是災難級錯誤；Telea/LaMa 最差只是糊掉，不會無中生有。且投影片背景多為純色／漸層／幾何色塊，diffusion 的生成力用不上卻要付 1–2 個數量級的時間成本 |
| 整頁送雲端影像模型 | 會重繪圖表、logo、照片等本應保留的元素，且 144 DPI 下 16:9 版面為 1920×1080，超過該類模型 1K 輸出上限。patch 路徑（Phase 4）才是正確接法 |
| 繼續加 `opencv-fast` 啟發式 | 已有 20 個超參數，邊際效益遞減。應讓 learned model 成為預設（Phase 1）而非繼續疊規則 |

---

## 7. 里程碑總表

| Phase | 內容 | 規模 | 風險 | 相依 |
|-------|------|------|------|------|
| 0 | 品質基準：樣本語料 + `bench_inpaint.py` + 非阻斷 lint | 中 | 低 | — |
| 1 | 生成式去字可用化：CPU ONNX、in-process LaMa、拆檔、納入 auto | 大 | 中 | Phase 0 |
| 2 | 工程健康度：lint 轉阻斷、依賴升級 | 中 | 低 | Phase 0（可與 3 並行） |
| 3 | 版面分析 v1.5 + 字體家族識別 | 大 | 中 | Phase 0（可與 1 並行） |
| 4 | 雲端影像模型手動重修 | 中 | 中 | Phase 0、Phase 1 |
| 5 | 模型升級（MI-GAN 等） | 中 | 低 | Phase 0、Phase 1 |

**建議起點**：Phase 0。在它完成之前，Phase 1 與 Phase 4 都沒有驗收標準，
等於是在無法證明有效的前提下改動核心路徑。
