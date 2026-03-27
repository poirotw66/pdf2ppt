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
- CLI 會顯示逐頁轉換進度條。

## 環境需求

- Python 3.11 以上
- 建議使用 Linux
- 依賴套件定義於 `pyproject.toml`
- 若要使用 OCR，建議使用獨立的 Conda 環境並固定 `numpy<2`
- OCR 需要：
  - PaddleOCR 執行環境與模型下載
- 若要使用本地 diffusion inpainting，建議：
  - NVIDIA GPU
  - `iopaint` 或其他相容的本地後端

## 安裝方式

建議的 OCR 執行環境：

```bash
conda create -n ppocr python=3.12 numpy=1.26.4 -y
conda activate ppocr
python -m pip install -e .
```

建議這樣安裝的原因：

- `PaddleOCR` / `PaddleX` 目前在 `numpy<2` 的環境下較穩定。
- 使用獨立 Conda 環境，可降低 `pyarrow`、`scikit-learn` 等全域套件相依衝突。

如果你已經有現成環境，至少請先確認符合 `pyproject.toml` 中的 NumPy 限制：

```bash
python -m pip install "numpy<2"
python -m pip install -e .
```

如果你要執行測試，請另外安裝：

```bash
python -m pip install pytest
```

如果你要使用本地 diffusion inpainting，請另外安裝並確認後端可正常執行。本專案目前已驗證過 `iopaint` 流程。

快速確認環境是否正常：

```bash
python - <<'PY'
import numpy
print(numpy.__version__)
PY
python -m pdf2ppt input.pdf output.pptx
```

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

### `opencv-fast` 的技術原理

`opencv-fast` 是本專案在 `overlay` 頁面上使用的輕量級背景重建路徑。

它的設計目標是：

- 不需要下載模型
- 不需要 GPU
- 直接在本機用 OpenCV 完成
- 對純色、漸層、簡單紋理的投影片背景通常效果最好

技術流程如下：

1. 先找出之後要重建成可編輯 PowerPoint 文字的文字區塊。
2. 透過 `build_text_mask_image()` 把這些文字區域轉成二值遮罩。
3. 可利用 `--inpaint-padding-px` 擴張遮罩，蓋住抗鋸齒邊緣與 OCR 框略小的情況。
4. `OpenCvFastInpaintingEngine` 會把頁面影像轉成 NumPy / OpenCV 格式。
5. 接著呼叫 `cv2.inpaint(..., cv2.INPAINT_TELEA)`，用小半徑做局部修補。
6. 修補後的影像成為背景，再把可編輯文字方塊疊回 PowerPoint。

實作細節：

- 修補演算法：OpenCV Telea 方法（`cv2.INPAINT_TELEA`）
- 預設半徑：`3.0`
- 遮罩格式：8-bit 單通道二值 mask
- 影像流程：PIL RGB -> OpenCV BGR -> Telea 修補 -> PIL RGB

為什麼它很快：

- 它是傳統影像處理，不是生成式模型。
- 核心做法是從遮罩邊界往內推估周邊顏色與結構。
- 主要成本來自影像尺寸與遮罩大小，不需要模型載入與神經網路推論。

適合的情境：

- 純色背景簡報
- 輕微漸層背景
- 文字後方只有簡單紋理或幾何圖形
- 想快速迭代、快速預覽轉換結果

效果較差的情境：

- 文字後方是密集插圖或照片
- 遮罩區域很大
- 複雜圖案無法只靠鄰近像素合理補回
- 被移除的文字剛好壓在重要邊線、圖示或細線圖表上

它與 `auto` 路由的關係：

- 如果遮罩覆蓋面積太大，`auto` 會為了安全改用 `white-box`
- 如果背景複雜度較低，`auto` 會優先選 `opencv-fast`
- 複雜度會依據遮罩周圍區域的灰階變異與邊緣密度估算
- 如果複雜度高，且本地 diffusion 後端可用，`auto` 會改選 `diffusion-local`

常用參數：

- `--inpaint-engine opencv-fast`：強制指定使用此引擎
- `--inpaint-padding-px`：先擴張文字遮罩再修補
- `--inpaint-max-area-ratio`：當遮罩過大時避免使用局部修補
- `--debug-dir`：輸出 mask 與背景決策結果方便檢查

實務建議：

- 一般簡報 PDF 可以先從 `opencv-fast` 開始
- 如果文字邊緣殘留白邊或光暈，可小幅提高 `--inpaint-padding-px`
- 只有在背景真的夠複雜時，再切換到 `diffusion-local`

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
- `--ocr-det-thresh`：PaddleOCR 文字偵測門檻，可選；省略時使用 PaddleOCR 官方預設
- `--ocr-det-box-thresh`：PaddleOCR 偵測框門檻，可選；省略時使用 PaddleOCR 官方預設
- `--ocr-drop-score`：PaddleOCR 辨識分數門檻，可選；省略時使用 PaddleOCR 官方預設
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
