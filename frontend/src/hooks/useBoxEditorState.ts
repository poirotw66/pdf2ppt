import { ChangeEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import type {
  Box,
  BoxFilter,
  BoxGroup,
  BoxSort,
  DragState,
  EditorTool,
  PagePayload,
  ResizeHandle,
} from "../types";
import {
  clamp,
  getLocalPoint,
  groupBoxes,
  isEditableTarget,
  normalizeAndClampRect,
  normalizeRect,
  rectToPolygon,
  rectanglesIntersect,
  resizeBBox,
  sortBoxes,
} from "../utils/geometry";

const lowConfidenceDefault = 0.85;

type UseBoxEditorStateOptions = {
  setStatusText: (value: string) => void;
};

export function useBoxEditorState({ setStatusText }: UseBoxEditorStateOptions) {
  const [pages, setPages] = useState<PagePayload[]>([]);
  const [selectedPageIndex, setSelectedPageIndex] = useState(0);
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [selectedBoxIds, setSelectedBoxIds] = useState<string[]>([]);
  const [dragState, setDragState] = useState<DragState>(null);
  const [zoom, setZoom] = useState(1);
  const [boxFilter, setBoxFilter] = useState<BoxFilter>("all");
  const [boxSort, setBoxSort] = useState<BoxSort>("reading-order");
  const [boxGroup, setBoxGroup] = useState<BoxGroup>("none");
  const [editorTool, setEditorTool] = useState<EditorTool>("select");
  const [lowConfidenceThreshold, setLowConfidenceThreshold] = useState(lowConfidenceDefault);
  const editorRef = useRef<HTMLDivElement>(null);
  const editorViewportRef = useRef<HTMLDivElement>(null);
  const [editorViewportSize, setEditorViewportSize] = useState({ width: 0, height: 0 });

  const selectedPage = pages[selectedPageIndex] ?? null;
  const selectedBox = useMemo(
    () => selectedPage?.boxes.find((box) => box.id === selectedBoxId) ?? null,
    [selectedBoxId, selectedPage],
  );
  const filteredBoxes = useMemo(() => {
    if (!selectedPage) {
      return [];
    }
    return selectedPage.boxes.filter((box) => {
      if (boxFilter === "empty-text") {
        return !(box.text ?? "").trim();
      }
      if (boxFilter === "low-confidence") {
        return box.confidence < lowConfidenceThreshold;
      }
      return true;
    });
  }, [boxFilter, lowConfidenceThreshold, selectedPage]);
  const orderedBoxes = useMemo(() => sortBoxes(filteredBoxes, boxSort), [boxSort, filteredBoxes]);
  const groupedBoxes = useMemo(() => groupBoxes(orderedBoxes, boxGroup), [boxGroup, orderedBoxes]);
  const fitScale = useMemo(() => {
    if (!selectedPage || editorViewportSize.width <= 0 || editorViewportSize.height <= 0) {
      return 1;
    }
    const widthScale = editorViewportSize.width / selectedPage.width;
    const heightScale = editorViewportSize.height / selectedPage.height;
    return Math.min(widthScale, heightScale);
  }, [editorViewportSize.height, editorViewportSize.width, selectedPage]);
  const previewScale = selectedPage ? fitScale * zoom : 1;

  useEffect(() => {
    function onWindowKeyDown(event: KeyboardEvent) {
      if (!selectedPage || !selectedBoxId) {
        return;
      }
      if (dragState || isEditableTarget(event.target)) {
        return;
      }
      const step = event.shiftKey ? 10 : 1;
      switch (event.key) {
        case "Delete":
        case "Backspace":
          event.preventDefault();
          deleteSelectedBox();
          return;
        case "ArrowUp":
          event.preventDefault();
          nudgeSelectedBox(0, -step);
          return;
        case "ArrowDown":
          event.preventDefault();
          nudgeSelectedBox(0, step);
          return;
        case "ArrowLeft":
          event.preventDefault();
          nudgeSelectedBox(-step, 0);
          return;
        case "ArrowRight":
          event.preventDefault();
          nudgeSelectedBox(step, 0);
          return;
        case "[":
        case "j":
        case "J":
          event.preventDefault();
          selectRelativeBox(-1);
          return;
        case "]":
        case "k":
        case "K":
          event.preventDefault();
          selectRelativeBox(1);
          return;
        default:
          return;
      }
    }

    window.addEventListener("keydown", onWindowKeyDown);
    return () => window.removeEventListener("keydown", onWindowKeyDown);
  }, [dragState, orderedBoxes, selectedBoxId, selectedPage]);

  useEffect(() => {
    if (!editorViewportRef.current) {
      return;
    }
    const element = editorViewportRef.current;
    const updateSize = () => {
      setEditorViewportSize({
        width: Math.max(element.clientWidth - 36, 1),
        height: Math.max(element.clientHeight - 36, 1),
      });
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setDragState(null);
  }, [selectedPageIndex]);

  function resetPages() {
    setPages([]);
    setSelectedPageIndex(0);
    setSelectedBoxId(null);
    setSelectedBoxIds([]);
  }

  function loadDetectedPages(nextPages: PagePayload[]) {
    setPages(nextPages);
    setSelectedPageIndex(0);
    setSelectedBoxId(nextPages[0]?.boxes[0]?.id ?? null);
    setSelectedBoxIds(nextPages[0]?.boxes[0]?.id ? [nextPages[0].boxes[0].id] : []);
  }

  function selectPage(index: number) {
    const page = pages[index];
    setSelectedPageIndex(index);
    setSelectedBoxId(page?.boxes[0]?.id ?? null);
    setSelectedBoxIds(page?.boxes[0]?.id ? [page.boxes[0].id] : []);
    if (page) {
      setStatusText(`Viewing page ${page.page}.`);
    }
  }

  function selectBox(boxId: string) {
    setSelectedBoxId(boxId);
    setSelectedBoxIds([boxId]);
  }

  function updateCurrentPage(mutator: (page: PagePayload) => PagePayload) {
    setPages((currentPages) =>
      currentPages.map((page, index) => (index === selectedPageIndex ? mutator(page) : page)),
    );
  }

  function updateSelectedBox(mutator: (box: Box, page: PagePayload) => Box) {
    if (!selectedBoxId) {
      return;
    }
    updateCurrentPage((page) => ({
      ...page,
      boxes: page.boxes.map((box) => (box.id === selectedBoxId ? mutator(box, page) : box)),
    }));
  }

  function deleteSelectedBox() {
    if (!selectedPage || (selectedBoxIds.length === 0 && !selectedBoxId)) {
      return;
    }
    const idsToDelete = new Set(selectedBoxIds.length > 0 ? selectedBoxIds : selectedBoxId ? [selectedBoxId] : []);
    updateCurrentPage((page) => ({
      ...page,
      boxes: page.boxes.filter((box) => !idsToDelete.has(box.id)),
    }));
    setSelectedBoxId(null);
    setSelectedBoxIds([]);
  }

  function onEditorMouseDown(event: React.MouseEvent<HTMLDivElement>) {
    if (!selectedPage || !editorRef.current) {
      return;
    }
    const target = event.target as HTMLElement;
    if (target.closest(".ocr-box") || target.closest(".resize-handle")) {
      return;
    }
    const point = getLocalPoint(event, editorRef.current, previewScale);
    setSelectedBoxId(null);
    setSelectedBoxIds([]);
    setDragState(
      editorTool === "create"
        ? { kind: "create", startX: point.x, startY: point.y, currentX: point.x, currentY: point.y }
        : { kind: "select", startX: point.x, startY: point.y, currentX: point.x, currentY: point.y },
    );
  }

  function onEditorMouseMove(event: React.MouseEvent<HTMLDivElement>) {
    if (!dragState || !editorRef.current) {
      return;
    }
    const point = getLocalPoint(event, editorRef.current, previewScale);
    if (dragState.kind === "create" || dragState.kind === "select") {
      setDragState({ ...dragState, currentX: point.x, currentY: point.y });
      return;
    }
    if (dragState.kind === "move") {
      const dx = point.x - dragState.startX;
      const dy = point.y - dragState.startY;
      updateCurrentPage((page) => ({
        ...page,
        boxes: page.boxes.map((box) => {
          if (box.id !== dragState.boxId) {
            return box;
          }
          const [x0, y0, x1, y1] = dragState.origin;
          const width = x1 - x0;
          const height = y1 - y0;
          const nextX0 = clamp(x0 + dx, 0, page.width - width);
          const nextY0 = clamp(y0 + dy, 0, page.height - height);
          return {
            ...box,
            bbox: [nextX0, nextY0, nextX0 + width, nextY0 + height],
            polygon: rectToPolygon(nextX0, nextY0, nextX0 + width, nextY0 + height),
          };
        }),
      }));
      return;
    }
    if (dragState.kind === "resize") {
      updateCurrentPage((page) => ({
        ...page,
        boxes: page.boxes.map((box) => {
          if (box.id !== dragState.boxId) {
            return box;
          }
          const nextBbox = resizeBBox(dragState.origin, dragState.handle, point.x, point.y, page.width, page.height);
          return {
            ...box,
            bbox: nextBbox,
            polygon: rectToPolygon(...nextBbox),
          };
        }),
      }));
    }
  }

  function onEditorMouseUp() {
    if (!selectedPage || !dragState) {
      setDragState(null);
      return;
    }
    if (dragState.kind === "create") {
      const bbox = normalizeRect(dragState.startX, dragState.startY, dragState.currentX, dragState.currentY);
      const [x0, y0, x1, y1] = bbox;
      if (x1 - x0 >= 8 && y1 - y0 >= 8) {
        const newId = `user_${selectedPage.page}_${Date.now()}`;
        updateCurrentPage((page) => ({
          ...page,
          boxes: [
            ...page.boxes,
            {
              id: newId,
              source: "ocr-user",
              bbox,
              polygon: rectToPolygon(x0, y0, x1, y1),
              text: "",
              confidence: 1,
            },
          ],
        }));
        setSelectedBoxId(newId);
        setSelectedBoxIds([newId]);
      }
    }
    if (dragState.kind === "select") {
      const selectionRect = normalizeRect(dragState.startX, dragState.startY, dragState.currentX, dragState.currentY);
      const matchedIds = selectedPage.boxes.filter((box) => rectanglesIntersect(selectionRect, box.bbox)).map((box) => box.id);
      setSelectedBoxIds(matchedIds);
      setSelectedBoxId(matchedIds[0] ?? null);
      setStatusText(matchedIds.length > 0 ? `Selected ${matchedIds.length} box(es).` : "No boxes selected.");
    }
    setDragState(null);
  }

  function startMoveBox(event: React.MouseEvent<HTMLDivElement>, box: Box) {
    event.stopPropagation();
    if (!editorRef.current) {
      return;
    }
    const point = getLocalPoint(event, editorRef.current, previewScale);
    setSelectedBoxId(box.id);
    setSelectedBoxIds([box.id]);
    setDragState({ kind: "move", boxId: box.id, startX: point.x, startY: point.y, origin: box.bbox });
  }

  function startResizeBox(event: React.MouseEvent<HTMLButtonElement>, box: Box, handle: ResizeHandle) {
    event.stopPropagation();
    if (!editorRef.current) {
      return;
    }
    const point = getLocalPoint(event, editorRef.current, previewScale);
    setSelectedBoxId(box.id);
    setSelectedBoxIds([box.id]);
    setDragState({ kind: "resize", boxId: box.id, handle, startX: point.x, startY: point.y, origin: box.bbox });
  }

  function updateSelectedBoxText(text: string) {
    updateSelectedBox((box) => ({ ...box, text }));
  }

  function updateSelectedBoxMeta(field: "source" | "confidence", value: string) {
    updateSelectedBox((box) => {
      if (field === "confidence") {
        const confidence = Number.parseFloat(value);
        return { ...box, confidence: Number.isFinite(confidence) ? clamp(confidence, 0, 1) : box.confidence };
      }
      return { ...box, source: value };
    });
  }

  function updateSelectedBBox(index: number, value: string) {
    updateSelectedBox((box, page) => {
      const parsed = Number.parseFloat(value);
      if (!Number.isFinite(parsed)) {
        return box;
      }
      const next = [...box.bbox] as [number, number, number, number];
      next[index] = parsed;
      const normalized = normalizeAndClampRect(next[0], next[1], next[2], next[3], page.width, page.height);
      return {
        ...box,
        bbox: normalized,
        polygon: rectToPolygon(...normalized),
      };
    });
  }

  function nudgeSelectedBox(dx: number, dy: number) {
    updateSelectedBox((box, page) => {
      const [x0, y0, x1, y1] = box.bbox;
      const width = x1 - x0;
      const height = y1 - y0;
      const nextX0 = clamp(x0 + dx, 0, page.width - width);
      const nextY0 = clamp(y0 + dy, 0, page.height - height);
      const normalized = [nextX0, nextY0, nextX0 + width, nextY0 + height] as [number, number, number, number];
      return {
        ...box,
        bbox: normalized,
        polygon: rectToPolygon(...normalized),
      };
    });
  }

  function expandSelectedBox(delta: number) {
    updateSelectedBox((box, page) => {
      const [x0, y0, x1, y1] = box.bbox;
      const normalized = normalizeAndClampRect(x0 - delta, y0 - delta, x1 + delta, y1 + delta, page.width, page.height);
      return {
        ...box,
        bbox: normalized,
        polygon: rectToPolygon(...normalized),
      };
    });
  }

  function duplicateSelectedBox() {
    if (!selectedBox) {
      return;
    }
    const newId = `${selectedBox.id}_copy_${Date.now()}`;
    const duplicated = {
      ...selectedBox,
      id: newId,
      bbox: [
        selectedBox.bbox[0] + 8,
        selectedBox.bbox[1] + 8,
        selectedBox.bbox[2] + 8,
        selectedBox.bbox[3] + 8,
      ] as [number, number, number, number],
    };
    duplicated.polygon = rectToPolygon(...duplicated.bbox);
    updateCurrentPage((page) => ({
      ...page,
      boxes: [...page.boxes, duplicated],
    }));
    setSelectedBoxId(newId);
    setSelectedBoxIds([newId]);
  }

  function onZoomChange(event: ChangeEvent<HTMLInputElement>) {
    setZoom(Number.parseInt(event.target.value, 10) / 100);
  }

  function onThresholdChange(event: ChangeEvent<HTMLInputElement>) {
    const next = Number.parseFloat(event.target.value);
    if (!Number.isFinite(next)) {
      return;
    }
    setLowConfidenceThreshold(clamp(next, 0, 1));
  }

  function onSortChange(event: ChangeEvent<HTMLSelectElement>) {
    setBoxSort(event.target.value as BoxSort);
  }

  function onGroupChange(event: ChangeEvent<HTMLSelectElement>) {
    setBoxGroup(event.target.value as BoxGroup);
  }

  function onBoxListKeyDown(event: ReactKeyboardEvent<HTMLButtonElement>, boxId: string) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setSelectedBoxId(boxId);
    }
  }

  function selectRelativeBox(direction: -1 | 1) {
    if (orderedBoxes.length === 0) {
      return;
    }
    const currentIndex = orderedBoxes.findIndex((box) => box.id === selectedBoxId);
    const nextIndex = currentIndex === -1
      ? direction > 0 ? 0 : orderedBoxes.length - 1
      : clamp(currentIndex + direction, 0, orderedBoxes.length - 1);
    const nextId = orderedBoxes[nextIndex]?.id ?? null;
    setSelectedBoxId(nextId);
    setSelectedBoxIds(nextId ? [nextId] : []);
  }

  return {
    boxFilter,
    boxGroup,
    boxSort,
    dragState,
    duplicateSelectedBox,
    editorRef,
    editorTool,
    editorViewportRef,
    expandSelectedBox,
    groupedBoxes,
    loadDetectedPages,
    lowConfidenceThreshold,
    nudgeSelectedBox,
    onBoxListKeyDown,
    onEditorMouseDown,
    onEditorMouseMove,
    onEditorMouseUp,
    onGroupChange,
    onSortChange,
    onThresholdChange,
    onZoomChange,
    orderedBoxes,
    pages,
    previewScale,
    resetPages,
    selectBox,
    selectPage,
    selectedBox,
    selectedBoxId,
    selectedBoxIds,
    selectedPage,
    selectedPageIndex,
    setBoxFilter,
    setBoxGroup,
    setBoxSort,
    setEditorTool,
    setPages,
    setSelectedBoxIds,
    setSelectedPageIndex,
    startMoveBox,
    startResizeBox,
    updateSelectedBBox,
    updateSelectedBoxMeta,
    updateSelectedBoxText,
    zoom,
    deleteSelectedBox,
  };
}