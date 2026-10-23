# Architecture

```text
PDF → page classify (digital / scanned / hybrid)
    → native extract and/or OCR (+ line merge, optional layout)
    → optional vector/table nativization
    → conditional background (elements / overlay / full-page)
    → PPTX + JSON report
    → optional --refine (experimental render-compare-fix; LibreOffice)
    → optional SlideIR emit / from-ir render
```

| Piece | Role |
| --- | --- |
| `src/pdf2ppt/cli.py` | CLI entry |
| `src/pdf2ppt/batch.py` | Batch folder convert |
| `src/pdf2ppt/pipeline.py` | Conversion orchestration |
| `src/pdf2ppt/ir_io.py` | SlideIR serialize / deserialize |
| `src/pdf2ppt/api.py` | Review UI backend |
| `frontend/` | OCR box review UI |

Background strategy in short: keep native elements when safe; for overlay
pages, `auto` prefers `opencv-fast`, then `lama-onnx` when a LaMa runtime is
available, otherwise `white-box`. Details:
[background-engines.md](background-engines.md).
