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

Recorded mode requires committed **live** Gemini captures under
``layout_planner_recorded/*.response.json`` (real block ids). Missing or
GT-copied files are marked ``oracle_response`` / skipped and **excluded** from
acceptance means — never reported as planner metrics.

Capture with::

    python scripts/bench_layout_planner.py --mode live --record-responses --max-calls 8

Examples:

    python scripts/bench_layout_planner.py --mode recorded --compare-heuristic
    python scripts/bench_layout_planner.py --mode live --max-calls 8 --record-responses
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

from pdf2ppt.core import ConversionOptions  # noqa: E402
from pdf2ppt.layout_metrics import (  # noqa: E402
    evaluate_page_layout,
    evaluate_planner_acceptance,
    evaluate_planner_acceptance_from_pptx,
    load_layout_ground_truth,
    load_planner_ground_truth,
)
from pdf2ppt.layout_planner import LayoutPlannerBudget  # noqa: E402
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    # Ignore empty placeholders / markers.
    if not payload.get("blocks") and not payload.get("groups"):
        return None
    if payload.get("_comment") and not payload.get("blocks"):
        return None
    return payload


def _looks_like_ground_truth_oracle(
    recorded: dict[str, Any],
    ground_truth: dict[str, Any] | None,
) -> bool:
    """True when recorded ids match planner.json GT ids (not live native_* ids)."""

    if ground_truth is None:
        return False
    gt_ids = {str(item.get("id")) for item in (ground_truth.get("paragraphs") or []) if item.get("id")}
    recorded_ids = {str(item.get("id")) for item in (recorded.get("blocks") or []) if item.get("id")}
    if not recorded_ids or not gt_ids:
        return False
    return recorded_ids <= gt_ids


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
    """Remap recorded member ids onto the block ids present in the live request.

    Never rebuilds roles from ground truth — that would score an oracle.
    When ids already match the request, return the payload unchanged. Otherwise
    remap by reading-order index from the recorded plan onto request blocks.
    """

    del ground_truth  # Explicitly unused: do not GT-oracle remap.
    request_blocks = _extract_request_blocks(body)
    if not request_blocks:
        return payload

    known_req = {str(item.get("id")) for item in request_blocks}
    recorded_blocks = list(payload.get("blocks") or [])
    recorded_ids = {str(item.get("id")) for item in recorded_blocks}
    if recorded_ids and recorded_ids <= known_req:
        return payload

    # Index-align recorded annotations onto live request blocks (plumbing only).
    blocks_out: list[dict[str, Any]] = []
    for index, req in enumerate(request_blocks):
        src = recorded_blocks[index] if index < len(recorded_blocks) else {}
        role = str(src.get("role") or "body")
        if role == "list":
            role = "list_item"
        blocks_out.append(
            {
                "id": str(req.get("id")),
                "role": role,
                "reading_order": index,
                "font_family_class": str(src.get("font_family_class") or "sans"),
                "confidence": float(src.get("confidence") or 0.9),
            }
        )
    groups_out = [
        {
            "id": f"g_{block['id']}",
            "member_ids": [str(block["id"])],
            "role": str(block["role"]),
            "reading_order": int(block["reading_order"]),
            "confidence": float(block["confidence"]),
            "kind": "text_frame",
        }
        for block in blocks_out
    ]
    return {"blocks": blocks_out, "groups": groups_out}


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


def _save_recorded_plan(recorded_dir: Path, stem: str, page: Any) -> Path | None:
    """Persist the live plan (with real block ids) for later --mode recorded."""

    meta = getattr(page, "plan_metadata", None) or {}
    plan = meta.get("plan")
    if not isinstance(plan, dict) or not plan.get("blocks"):
        return None
    recorded_dir.mkdir(parents=True, exist_ok=True)
    out = recorded_dir / f"{stem}.response.json"
    payload = {
        "blocks": plan.get("blocks") or [],
        "groups": plan.get("groups") or [],
        "model": plan.get("model"),
        "prompt_version": plan.get("prompt_version"),
        "recorded_from": "live",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


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
    record_responses: bool = False,
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
        if recorded is None:
            return {
                "fixture": stem,
                "mode": mode,
                "oracle_response": True,
                "skipped": True,
                "error": (
                    "no real recorded response "
                    f"(expected {recorded_dir / (stem + '.response.json')}); "
                    "not scoring GT-oracle. Capture with --mode live --record-responses."
                ),
            }
        if _looks_like_ground_truth_oracle(recorded, gt):
            return {
                "fixture": stem,
                "mode": mode,
                "oracle_response": True,
                "skipped": True,
                "error": (
                    "recorded response ids match ground-truth labels (oracle); "
                    "excluded from averages. Replace with a live Gemini capture."
                ),
            }
        options.layout_planner = "gemini"
        # Do not pass ground_truth into the transport — that would rebuild roles
        # from GT when ids fail to match live blocks.
        options.layout_planner_transport = _make_recorded_transport(recorded, ground_truth=None)
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

        recorded_path = None
        if record_responses and mode == "live" and pages:
            recorded_path = _save_recorded_plan(recorded_dir, stem, pages[0])

        budget = getattr(options, "layout_planner_budget", None)
        return {
            "fixture": stem,
            "mode": mode,
            "oracle_response": oracle_response,
            "score": score,
            "layout_planner": budget.to_dict() if isinstance(budget, LayoutPlannerBudget) else None,
            "apply_grouping": apply_grouping,
            "apply_fonts": apply_fonts,
            "recorded_response_path": str(recorded_path) if recorded_path else None,
        }
    except Exception as error:  # noqa: BLE001
        return {"fixture": stem, "mode": mode, "error": str(error)}


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
    oracle_count = sum(1 for item in results if item.get("oracle_response") or item.get("skipped"))
    return {
        "acceptance_means_excluding_oracle": means,
        "oracle_or_skipped_count": oracle_count,
        "note": (
            "Acceptance means exclude oracle/skipped fixtures. "
            "Recorded mode without real Gemini captures does not claim planner metrics."
        ),
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
    parser.add_argument(
        "--record-responses",
        action="store_true",
        help="With --mode live, write real plan JSON (live block ids) into --recorded-dir.",
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
            record_responses=bool(args.record_responses),
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
