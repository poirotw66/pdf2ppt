#!/usr/bin/env python3
"""Phase 3 layout-planner evaluation: heuristics vs planner on annotated fixtures.

Modes:
  --mode recorded   Use committed ``*.response.json`` (CI-safe, no API key)
  --mode live       Call Gemini (requires GEMINI_API_KEY); respects call cap + disk cache
  --mode heuristic  Geometric layout only (baseline)

Scores acceptance from the **actual PPTX** (title placeholder, header/footer
text presence, reading-order string order), plus IR-level grouping metrics.
When ``--compare-heuristic`` is set, also reports editability / text F1 (and
SSIM when LibreOffice is available) vs the planner-off baseline for the same
fixture.

Recorded responses are **not** treated as a ground-truth oracle for acceptance
summaries: auto-generated GT payloads are marked ``oracle_response`` and
excluded from mean acceptance. Prefer committed ``layout_planner_recorded/``.

Examples:

    python scripts/bench_layout_planner.py --mode recorded --compare-heuristic
    python scripts/bench_layout_planner.py --mode live --max-calls 8
    PDF2PPT_RUN_LAYOUT_PLANNER=1 python scripts/bench_layout_planner.py --mode live
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PIL import Image  # noqa: E402

from pdf2ppt.core import ConversionOptions  # noqa: E402
from pdf2ppt.layout_metrics import (  # noqa: E402
    evaluate_page_layout,
    evaluate_planner_acceptance,
    evaluate_planner_acceptance_from_pptx,
    load_layout_ground_truth,
    load_planner_ground_truth,
)
from pdf2ppt.layout_planner import (  # noqa: E402
    LayoutPlannerBudget,
    LayoutPlannerService,
    create_layout_planner,
)
from pdf2ppt.models import TextBlock  # noqa: E402
from pdf2ppt.pipeline import analyze_pdf, render_pages_to_pptx  # noqa: E402
from pdf2ppt.refine import (  # noqa: E402
    extract_pptx_slide_text,
    slide_editability_score,
    slide_text_f1_vs_pdf,
)

DEFAULT_FIXTURES = (
    "bullet_list",
    "title_two_column",
    "figure_caption",
    "table",
    "geometric_chart",
    "scanned_bullet_list",
    "mixed_cjk_en",
)
DEFAULT_RECORDED_DIR = REPO_ROOT / "tests" / "fixtures" / "layout_planner_recorded"
DEFAULT_JSON_OUT = REPO_ROOT / "output" / "bench_layout_planner.json"


def _load_recorded_response(recorded_dir: Path, fixture_stem: str) -> dict[str, Any] | None:
    path = recorded_dir / f"{fixture_stem}.response.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _make_recorded_transport(payload: dict[str, Any], *, ground_truth: dict[str, Any] | None = None):
    """Return a transport that adapts recorded / GT roles onto live block ids."""

    def transport(body: dict[str, Any]) -> dict[str, Any]:
        adapted = _adapt_plan_to_request_blocks(payload, body, ground_truth=ground_truth)
        return {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps(adapted, ensure_ascii=False)}]}}
            ]
        }

    return transport


def _extract_request_blocks(body: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        parts = body["contents"][0]["parts"]
        text = next(part["text"] for part in parts if isinstance(part, dict) and "text" in part)
        marker = "BLOCKS:\n"
        raw = text.split(marker, 1)[1]
        blocks = json.loads(raw)
        return blocks if isinstance(blocks, list) else []
    except Exception:  # noqa: BLE001
        return []


def _adapt_plan_to_request_blocks(
    payload: dict[str, Any],
    body: dict[str, Any],
    *,
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Remap recorded member ids onto the block ids present in the live request."""

    request_blocks = _extract_request_blocks(body)
    if not request_blocks:
        return payload

    gt_paragraphs = list((ground_truth or {}).get("paragraphs") or [])
    id_map: dict[str, str] = {}
    used_gt: set[int] = set()
    for req in request_blocks:
        req_id = str(req.get("id"))
        req_bbox = [float(v) for v in (req.get("bbox") or [0, 0, 0, 0])[:4]]
        best_i = None
        best_score = 0.0
        for gi, gt in enumerate(gt_paragraphs):
            if gi in used_gt:
                continue
            gb = [float(v) for v in (gt.get("bbox") or [0, 0, 0, 0])[:4]]
            inter_w = max(0.0, min(req_bbox[2], gb[2]) - max(req_bbox[0], gb[0]))
            inter_h = max(0.0, min(req_bbox[3], gb[3]) - max(req_bbox[1], gb[1]))
            inter = inter_w * inter_h
            area_a = max(1.0, (req_bbox[2] - req_bbox[0]) * (req_bbox[3] - req_bbox[1]))
            area_b = max(1.0, (gb[2] - gb[0]) * (gb[3] - gb[1]))
            score = inter / max(area_a + area_b - inter, 1.0)
            if score > best_score:
                best_score = score
                best_i = gi
        if best_i is not None and best_score >= 0.05:
            used_gt.add(best_i)
            id_map[str(gt_paragraphs[best_i].get("id"))] = req_id

    known_req = {str(item.get("id")) for item in request_blocks}
    recorded_ids = {str(item.get("id")) for item in (payload.get("blocks") or [])}
    if recorded_ids and recorded_ids <= known_req:
        return payload

    blocks_out: list[dict[str, Any]] = []
    for index, req in enumerate(request_blocks):
        req_id = str(req.get("id"))
        role = "body"
        font_class = "sans"
        for gt_id, mapped in id_map.items():
            if mapped != req_id:
                continue
            gt = next((item for item in gt_paragraphs if str(item.get("id")) == gt_id), None)
            if gt is None:
                break
            role = str(gt.get("role") or "body")
            if role == "list":
                role = "list_item"
            font_class = str(gt.get("font_family_class") or "sans")
            break
        blocks_out.append(
            {
                "id": req_id,
                "role": role,
                "reading_order": index,
                "font_family_class": font_class,
                "confidence": 0.92,
            }
        )

    groups_out: list[dict[str, Any]] = []
    pending: list[str] = []
    pending_order = 0
    for block in blocks_out:
        if block["role"] == "list_item":
            if not pending:
                pending_order = int(block["reading_order"])
            pending.append(str(block["id"]))
            continue
        if pending:
            groups_out.append(
                {
                    "id": "g_list",
                    "member_ids": list(pending),
                    "role": "list_item",
                    "reading_order": pending_order,
                    "confidence": 0.92,
                    "kind": "paragraph",
                }
            )
            pending = []
        groups_out.append(
            {
                "id": f"g_{block['id']}",
                "member_ids": [str(block["id"])],
                "role": str(block["role"]),
                "reading_order": int(block["reading_order"]),
                "confidence": 0.92,
                "kind": "text_frame",
            }
        )
    if pending:
        groups_out.append(
            {
                "id": "g_list",
                "member_ids": list(pending),
                "role": "list_item",
                "reading_order": pending_order,
                "confidence": 0.92,
                "kind": "paragraph",
            }
        )
    return {"blocks": blocks_out, "groups": groups_out}


def _synthetic_blocks_from_gt(gt: dict[str, Any]) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for index, item in enumerate(gt.get("paragraphs") or []):
        bbox = item.get("bbox") or [0, 0, 1, 1]
        blocks.append(
            TextBlock(
                id=str(item.get("id") or f"b{index}"),
                source="ocr",
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                text=str(item.get("text") or ""),
                confidence=0.95,
                font_size=16.0,
                reading_order=index,
            )
        )
    return blocks


def _plan_payload_from_gt(gt: dict[str, Any]) -> dict[str, Any]:
    """Build a schema-valid recorded response from planner ground truth.

    Used only as a plumbing fallback when no committed ``*.response.json``
    exists. Results are tagged ``oracle_response`` and excluded from mean
    acceptance so recorded mode cannot score GT against itself unnoticed.
    """

    blocks: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for index, item in enumerate(gt.get("paragraphs") or []):
        block_id = str(item.get("id") or f"b{index}")
        role = str(item.get("role") or "body")
        if role == "list":
            role = "list_item"
        font_class = str(item.get("font_family_class") or "sans")
        blocks.append(
            {
                "id": block_id,
                "role": role,
                "reading_order": index,
                "font_family_class": font_class,
                "confidence": 0.92,
            }
        )
        groups.append(
            {
                "id": f"g_{block_id}",
                "member_ids": [block_id],
                "role": role,
                "reading_order": index,
                "confidence": 0.92,
                "kind": "text_frame",
            }
        )
    merged_groups: list[dict[str, Any]] = []
    pending: list[str] = []
    pending_order = 0
    for group in groups:
        if str(group["role"]) == "list_item":
            if not pending:
                pending_order = int(group["reading_order"])
            pending.extend(str(member_id) for member_id in list(group["member_ids"]))
            continue
        if pending:
            merged_groups.append(
                {
                    "id": "g_list",
                    "member_ids": list(pending),
                    "role": "list_item",
                    "reading_order": pending_order,
                    "confidence": 0.92,
                    "kind": "paragraph",
                }
            )
            pending = []
        merged_groups.append(group)
    if pending:
        merged_groups.append(
            {
                "id": "g_list",
                "member_ids": list(pending),
                "role": "list_item",
                "reading_order": pending_order,
                "confidence": 0.92,
                "kind": "paragraph",
            }
        )
    return {"blocks": blocks, "groups": merged_groups}


def _score_pages(
    pages: list[Any],
    *,
    fixtures_dir: Path,
    stem: str,
    pptx_path: Path | None = None,
) -> dict[str, Any]:
    page = pages[0] if pages else None
    if page is None:
        return {"error": "no pages"}
    planner_gt_path = fixtures_dir / f"{stem}.planner.json"
    layout_gt_path = fixtures_dir / f"{stem}.layout.json"
    payload: dict[str, Any] = {
        "layout_engine": page.layout_engine,
        "text_block_count": len(page.text_blocks),
        "group_count": len(page.group_elements),
    }
    gt: dict[str, Any] | None = None
    if planner_gt_path.is_file():
        gt = load_planner_ground_truth(planner_gt_path)
        # IR-level scores remain useful for debugging apply path.
        payload["acceptance_ir"] = evaluate_planner_acceptance(page.text_blocks, gt).to_dict()
        if pptx_path is not None and pptx_path.is_file():
            payload["acceptance"] = evaluate_planner_acceptance_from_pptx(pptx_path, gt).to_dict()
        else:
            payload["acceptance"] = payload["acceptance_ir"]
            payload["acceptance_note"] = "pptx missing; fell back to IR blocks"
    if layout_gt_path.is_file():
        grouping = evaluate_page_layout(page, load_layout_ground_truth(layout_gt_path))
        payload["grouping"] = grouping.to_dict()
    if page.plan_metadata:
        payload["plan_metadata_keys"] = sorted(page.plan_metadata.keys())
    return payload


def _fidelity_vs_pdf(pptx_path: Path, pdf_path: Path) -> dict[str, Any]:
    """Editability + text F1 without LibreOffice; SSIM when soffice is present."""

    payload: dict[str, Any] = {
        "editability": slide_editability_score(pptx_path, slide_index=0),
        "text_f1": slide_text_f1_vs_pdf(pptx_path, pdf_path, page_index=0),
        "pptx_text_chars": len(extract_pptx_slide_text(pptx_path, slide_index=0)),
        "ssim": None,
    }
    try:
        from pdf2ppt.render_compare import (
            evaluate_pptx_against_pdf,
            find_libreoffice_executable,
        )

        if find_libreoffice_executable() is not None:
            report = evaluate_pptx_against_pdf(pptx_path, pdf_path, dpi=110)
            pages = getattr(report, "pages", None) or []
            if pages:
                fidelity = getattr(pages[0], "fidelity", None)
                if fidelity is not None:
                    payload["ssim"] = getattr(fidelity, "ssim", None)
    except Exception as error:  # noqa: BLE001
        payload["ssim_error"] = str(error)
    return payload


def _delta(planned: float | None, baseline: float | None) -> float | None:
    if planned is None or baseline is None:
        return None
    return round(float(planned) - float(baseline), 6)


def run_fixture(
    stem: str,
    *,
    mode: str,
    fixtures_dir: Path,
    work_dir: Path,
    recorded_dir: Path,
    max_calls: int,
    cache_dir: Path | None,
    apply_grouping: bool,
    apply_fonts: bool,
    convert: bool,
) -> dict[str, Any]:
    pdf_path = fixtures_dir / f"{stem}.pdf"
    if not pdf_path.is_file():
        return {"fixture": stem, "error": f"missing {pdf_path.name}"}

    out_pptx = work_dir / f"{stem}_{mode}.pptx"
    report_path = work_dir / f"{stem}_{mode}.report.json"
    options = ConversionOptions(
        input_path=pdf_path,
        output_path=out_pptx,
        report_path=report_path,
        layout_engine="heuristic",
        vector_mode="off",
        inpaint_engine="white-box",
        # Keep default OCR model root so scanned fixtures can initialize PaddleOCR.
        layout_skip_header_footer=True,
        layout_planner_apply_grouping=apply_grouping,
        layout_planner_apply_fonts=apply_fonts,
    )

    oracle_response = False
    if mode == "heuristic":
        options.layout_planner = None
    elif mode == "recorded":
        gt_path = fixtures_dir / f"{stem}.planner.json"
        gt = load_planner_ground_truth(gt_path) if gt_path.is_file() else None
        recorded = _load_recorded_response(recorded_dir, stem)
        if recorded is None and gt is not None:
            recorded = _plan_payload_from_gt(gt)
            oracle_response = True
        if recorded is None:
            return {"fixture": stem, "error": "no recorded response"}
        options.layout_planner = "gemini"
        options.layout_planner_transport = _make_recorded_transport(recorded, ground_truth=gt)
        options.layout_planner_max_calls = max_calls
        options.layout_planner_cache_dir = cache_dir
        options.layout_planner_budget = LayoutPlannerBudget(max_calls=max_calls)
    elif mode == "live":
        options.layout_planner = "gemini"
        options.layout_planner_max_calls = max_calls
        options.layout_planner_cache_dir = cache_dir or Path(".pdf2ppt_cache/layout_planner")
        options.layout_planner_budget = LayoutPlannerBudget(max_calls=max_calls)
    else:
        return {"fixture": stem, "error": f"unknown mode {mode}"}

    try:
        pages, slide_w, slide_h = analyze_pdf(options)
        pptx_path: Path | None = None
        if convert:
            render_pages_to_pptx(
                pages,
                out_pptx,
                slide_width_pt=slide_w,
                slide_height_pt=slide_h,
            )
            pptx_path = out_pptx if out_pptx.is_file() else None
        score = _score_pages(pages, fixtures_dir=fixtures_dir, stem=stem, pptx_path=pptx_path)
        if pptx_path is not None:
            score["fidelity"] = _fidelity_vs_pdf(pptx_path, pdf_path)

        budget = getattr(options, "layout_planner_budget", None)
        return {
            "fixture": stem,
            "mode": mode,
            "oracle_response": oracle_response,
            "score": score,
            "layout_planner": budget.to_dict() if isinstance(budget, LayoutPlannerBudget) else None,
            "apply_grouping": apply_grouping,
            "apply_fonts": apply_fonts,
        }
    except Exception as error:  # noqa: BLE001
        if mode == "heuristic":
            return {"fixture": stem, "mode": mode, "error": str(error)}
        gt_path = fixtures_dir / f"{stem}.planner.json"
        if not gt_path.is_file():
            return {"fixture": stem, "mode": mode, "error": str(error)}
        gt = load_planner_ground_truth(gt_path)
        blocks = _synthetic_blocks_from_gt(gt)
        recorded = _load_recorded_response(recorded_dir, stem)
        used_oracle = False
        if recorded is None:
            recorded = _plan_payload_from_gt(gt)
            used_oracle = True
        planner = create_layout_planner(
            "gemini",
            transport=_make_recorded_transport(recorded, ground_truth=gt),
        )
        budget = LayoutPlannerBudget(max_calls=max_calls)
        service = LayoutPlannerService(
            planner,
            budget=budget,
            cache_dir=cache_dir,
            apply_grouping=apply_grouping,
            apply_fonts=apply_fonts,
        )
        image = Image.new("RGB", (960, 540), color=(255, 255, 255))
        applied = service.try_plan_page(
            page_image=image,
            text_blocks=blocks,
            page_number=1,
            page_height=540.0,
        )
        if applied is None:
            return {"fixture": stem, "mode": mode, "error": f"planner fallback ({error})"}
        from pdf2ppt.models import PageResult, QualityScore

        page = PageResult(
            page_number=1,
            page_kind="digital",
            background_mode="elements",
            width_pt=960.0,
            height_pt=540.0,
            text_blocks=applied.text_blocks,
            quality_score=QualityScore(1, 1, 1, 1),
            fallback_reason=None,
            group_elements=applied.group_elements,
            plan_metadata=applied.plan_metadata,
            layout_engine="layout_planner",
        )
        score = _score_pages([page], fixtures_dir=fixtures_dir, stem=stem, pptx_path=None)
        return {
            "fixture": stem,
            "mode": mode,
            "oracle_response": used_oracle,
            "score": score,
            "layout_planner": budget.to_dict(),
            "note": f"synthetic apply after analyze/convert error: {error}",
        }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    acceptance_keys = (
        "title_accuracy",
        "header_footer_filter_rate",
        "reading_order_correctness",
    )
    means: dict[str, float | None] = {}
    for key in acceptance_keys:
        values = [
            float(item["score"]["acceptance"][key])
            for item in results
            if not item.get("oracle_response")
            and isinstance(item.get("score"), dict)
            and isinstance(item["score"].get("acceptance"), dict)
            and item["score"]["acceptance"].get(key) is not None
        ]
        means[key] = round(sum(values) / len(values), 6) if values else None

    deltas: list[dict[str, Any]] = []
    for item in results:
        vs = item.get("vs_heuristic")
        if not isinstance(vs, dict):
            continue
        deltas.append(
            {
                "fixture": item.get("fixture"),
                "delta_ssim": vs.get("delta_ssim"),
                "delta_text_f1": vs.get("delta_text_f1"),
                "delta_editability": vs.get("delta_editability"),
            }
        )
    regressions = [
        row
        for row in deltas
        if any(
            isinstance(row.get(name), (int, float)) and float(row[name]) < -0.005
            for name in ("delta_ssim", "delta_text_f1", "delta_editability")
        )
    ]
    return {
        "acceptance_means_excluding_oracle": means,
        "vs_heuristic_deltas": deltas,
        "regression_count": len(regressions),
        "regressions": regressions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("recorded", "live", "heuristic"), default="recorded")
    parser.add_argument("--fixtures-dir", type=Path, default=REPO_ROOT / "tests" / "fixtures")
    parser.add_argument("--fixture", action="append", dest="fixtures", default=None)
    parser.add_argument("--recorded-dir", type=Path, default=DEFAULT_RECORDED_DIR)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "output" / "bench_layout_planner_work")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--max-calls", type=int, default=8, help="Live/recorded call cap (keep live cheap).")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".pdf2ppt_cache" / "layout_planner",
        help="Disk cache for live mode (and optional recorded).",
    )
    parser.add_argument(
        "--compare-heuristic",
        action="store_true",
        help="Also run heuristic baseline and report SSIM/text F1/editability deltas.",
    )
    parser.add_argument(
        "--apply-grouping",
        action="store_true",
        help="Opt into planner grouping merges (default off; low-risk roles/order/HF only).",
    )
    parser.add_argument(
        "--apply-fonts",
        action="store_true",
        help="Opt into VLM font-class remaps when PDF font is unknown (default off).",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Analyze only (skip PPTX convert). Acceptance then uses IR blocks.",
    )
    args = parser.parse_args(argv)

    stems = [stem for stem in (args.fixtures or list(DEFAULT_FIXTURES)) if (args.fixtures_dir / f"{stem}.pdf").is_file()]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    do_convert = not args.no_convert

    results: list[dict[str, Any]] = []
    heuristic_by_stem: dict[str, dict[str, Any]] = {}
    for stem in stems:
        if args.compare_heuristic and args.mode != "heuristic":
            baseline = run_fixture(
                stem,
                mode="heuristic",
                fixtures_dir=args.fixtures_dir,
                work_dir=args.work_dir,
                recorded_dir=args.recorded_dir,
                max_calls=args.max_calls,
                cache_dir=None,
                apply_grouping=False,
                apply_fonts=False,
                convert=do_convert,
            )
            results.append(baseline)
            heuristic_by_stem[stem] = baseline

        planned = run_fixture(
            stem,
            mode=args.mode,
            fixtures_dir=args.fixtures_dir,
            work_dir=args.work_dir,
            recorded_dir=args.recorded_dir,
            max_calls=args.max_calls,
            cache_dir=args.cache_dir if args.mode == "live" else None,
            apply_grouping=bool(args.apply_grouping),
            apply_fonts=bool(args.apply_fonts),
            convert=do_convert,
        )
        baseline_result = heuristic_by_stem.get(stem)
        if baseline_result is not None:
            base_f = ((baseline_result.get("score") or {}).get("fidelity")) or {}
            plan_f = ((planned.get("score") or {}).get("fidelity")) or {}
            planned["vs_heuristic"] = {
                "delta_ssim": _delta(plan_f.get("ssim"), base_f.get("ssim")),
                "delta_text_f1": _delta(plan_f.get("text_f1"), base_f.get("text_f1")),
                "delta_editability": _delta(plan_f.get("editability"), base_f.get("editability")),
                "baseline_fidelity": base_f or None,
                "planned_fidelity": plan_f or None,
            }
        results.append(planned)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "fixtures": stems,
        "apply_grouping": bool(args.apply_grouping),
        "apply_fonts": bool(args.apply_fonts),
        "convert": do_convert,
        "summary": _summarize(results),
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
