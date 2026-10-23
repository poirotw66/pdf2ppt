import type { PreviewBlockRole, PreviewEdits, PreviewPage, PreviewPageEdits } from "../types";

export function getPageEdits(edits: PreviewEdits, pageNumber: number): PreviewPageEdits {
  const existing = edits.pages.find((page) => page.page === pageNumber);
  if (existing) {
    return {
      page: existing.page,
      excluded_ids: [...existing.excluded_ids],
      role_overrides: { ...existing.role_overrides },
      raster_fallback: existing.raster_fallback,
    };
  }
  return {
    page: pageNumber,
    excluded_ids: [],
    role_overrides: {},
    raster_fallback: false,
  };
}

export function upsertPageEdits(edits: PreviewEdits, pageEdits: PreviewPageEdits): PreviewEdits {
  const pages = edits.pages.filter((page) => page.page !== pageEdits.page);
  const cleaned: PreviewPageEdits = {
    page: pageEdits.page,
    excluded_ids: [...new Set(pageEdits.excluded_ids)].sort(),
    role_overrides: { ...pageEdits.role_overrides },
    raster_fallback: pageEdits.raster_fallback,
  };
  const isEmpty =
    cleaned.excluded_ids.length === 0 &&
    Object.keys(cleaned.role_overrides).length === 0 &&
    !cleaned.raster_fallback;
  if (!isEmpty) {
    pages.push(cleaned);
  }
  pages.sort((a, b) => a.page - b.page);
  return { pages };
}

export function toggleExcluded(edits: PreviewEdits, pageNumber: number, elementId: string): PreviewEdits {
  const pageEdits = getPageEdits(edits, pageNumber);
  const excluded = new Set(pageEdits.excluded_ids);
  if (excluded.has(elementId)) {
    excluded.delete(elementId);
  } else {
    excluded.add(elementId);
  }
  return upsertPageEdits(edits, { ...pageEdits, excluded_ids: [...excluded] });
}

export function setBlockRole(
  edits: PreviewEdits,
  pageNumber: number,
  blockId: string,
  role: PreviewBlockRole,
  originalRole: string,
): PreviewEdits {
  const pageEdits = getPageEdits(edits, pageNumber);
  const role_overrides = { ...pageEdits.role_overrides };
  if (role === originalRole) {
    delete role_overrides[blockId];
  } else {
    role_overrides[blockId] = role;
  }
  return upsertPageEdits(edits, { ...pageEdits, role_overrides });
}

export function setRasterFallback(edits: PreviewEdits, pageNumber: number, enabled: boolean): PreviewEdits {
  const pageEdits = getPageEdits(edits, pageNumber);
  return upsertPageEdits(edits, { ...pageEdits, raster_fallback: enabled });
}

export function applyEditsToPreviewPage(page: PreviewPage, edits: PreviewEdits): PreviewPage {
  const pageEdits = getPageEdits(edits, page.page);
  const excluded = new Set(pageEdits.excluded_ids);
  return {
    ...page,
    raster_fallback: pageEdits.raster_fallback,
    text_blocks: page.text_blocks.map((block) => ({
      ...block,
      block_role: pageEdits.role_overrides[block.id] ?? block.block_role,
      excluded: excluded.has(block.id),
    })),
    tables: page.tables.map((table) => ({ ...table, excluded: excluded.has(table.id) })),
    images: page.images.map((image) => ({ ...image, excluded: excluded.has(image.id) })),
    shapes: page.shapes.map((shape) => ({ ...shape, excluded: excluded.has(shape.id) })),
  };
}

export function bboxToPercentStyle(
  bbox: number[],
  widthPt: number,
  heightPt: number,
): { left: string; top: string; width: string; height: string } {
  const [x0, y0, x1, y1] = bbox;
  const w = Math.max(widthPt, 1);
  const h = Math.max(heightPt, 1);
  return {
    left: `${(x0 / w) * 100}%`,
    top: `${(y0 / h) * 100}%`,
    width: `${(Math.max(0, x1 - x0) / w) * 100}%`,
    height: `${(Math.max(0, y1 - y0) / h) * 100}%`,
  };
}
