import type { MouseEvent as ReactMouseEvent } from "react";

import type { Box, BoxGroup, BoxGroupEntry, BoxSort, ResizeHandle } from "../types";

export const resizeHandles: ResizeHandle[] = ["nw", "ne", "se", "sw"];

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function normalizeRect(x0: number, y0: number, x1: number, y1: number): [number, number, number, number] {
  return [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
}

export function normalizeAndClampRect(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  maxWidth: number,
  maxHeight: number,
): [number, number, number, number] {
  const [nx0, ny0, nx1, ny1] = normalizeRect(
    clamp(x0, 0, maxWidth),
    clamp(y0, 0, maxHeight),
    clamp(x1, 0, maxWidth),
    clamp(y1, 0, maxHeight),
  );
  const minSize = 4;
  return [nx0, ny0, Math.max(nx0 + minSize, nx1), Math.max(ny0 + minSize, ny1)];
}

export function rectToPolygon(x0: number, y0: number, x1: number, y1: number): number[][] {
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ];
}

export function rectToStyle([x0, y0, x1, y1]: [number, number, number, number]) {
  return {
    left: x0,
    top: y0,
    width: x1 - x0,
    height: y1 - y0,
  };
}

export function rectanglesIntersect(
  [leftX0, leftY0, leftX1, leftY1]: [number, number, number, number],
  [rightX0, rightY0, rightX1, rightY1]: [number, number, number, number],
): boolean {
  return leftX0 <= rightX1 && leftX1 >= rightX0 && leftY0 <= rightY1 && leftY1 >= rightY0;
}

export function getLocalPoint(event: ReactMouseEvent<HTMLElement>, element: HTMLDivElement, scale: number) {
  const rect = element.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / (rect.width / element.offsetWidth)) / scale,
    y: ((event.clientY - rect.top) / (rect.height / element.offsetHeight)) / scale,
  };
}

export function resizeBBox(
  origin: [number, number, number, number],
  handle: ResizeHandle,
  pointerX: number,
  pointerY: number,
  maxWidth: number,
  maxHeight: number,
): [number, number, number, number] {
  let [x0, y0, x1, y1] = origin;
  if (handle.includes("n")) {
    y0 = pointerY;
  }
  if (handle.includes("s")) {
    y1 = pointerY;
  }
  if (handle.includes("w")) {
    x0 = pointerX;
  }
  if (handle.includes("e")) {
    x1 = pointerX;
  }
  return normalizeAndClampRect(x0, y0, x1, y1, maxWidth, maxHeight);
}

export function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  if (target.isContentEditable) {
    return true;
  }
  const tagName = target.tagName.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
}

export function sortBoxes(boxes: Box[], sortBy: BoxSort): Box[] {
  const nextBoxes = [...boxes];
  nextBoxes.sort((left, right) => {
    if (sortBy === "confidence-asc") {
      return left.confidence - right.confidence || compareReadingOrder(left, right);
    }
    if (sortBy === "confidence-desc") {
      return right.confidence - left.confidence || compareReadingOrder(left, right);
    }
    if (sortBy === "source") {
      return left.source.localeCompare(right.source) || compareReadingOrder(left, right);
    }
    return compareReadingOrder(left, right);
  });
  return nextBoxes;
}

export function groupBoxes(boxes: Box[], groupBy: BoxGroup): BoxGroupEntry[] {
  if (groupBy === "none") {
    return [{ key: "all", label: "All Boxes", boxes }];
  }
  const grouped = new Map<string, Box[]>();
  for (const box of boxes) {
    const key = box.source || "unknown";
    const existing = grouped.get(key);
    if (existing) {
      existing.push(box);
    } else {
      grouped.set(key, [box]);
    }
  }
  return Array.from(grouped.entries()).map(([key, groupBoxesValue]) => ({
    key,
    label: key,
    boxes: groupBoxesValue,
  }));
}

function compareReadingOrder(left: Box, right: Box): number {
  const yDiff = left.bbox[1] - right.bbox[1];
  if (Math.abs(yDiff) > 6) {
    return yDiff;
  }
  return left.bbox[0] - right.bbox[0];
}