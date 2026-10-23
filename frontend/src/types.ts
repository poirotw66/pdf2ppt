export type JobResponse = {
  job_id: string;
  status: string;
  original_filename: string;
  page_count: number;
  created_at: string;
  updated_at: string;
  detection_path?: string | null;
  approved_boxes_path?: string | null;
  output_pptx_path?: string | null;
  report_path?: string | null;
};

export type Box = {
  id: string;
  source: string;
  bbox: [number, number, number, number];
  polygon?: number[][] | null;
  text?: string | null;
  confidence: number;
};

export type PagePayload = {
  page: number;
  image_url: string;
  width: number;
  height: number;
  boxes: Box[];
};

export type DetectResponse = {
  job_id: string;
  status: string;
  pages: PagePayload[];
};

export type ConvertResponse = {
  job_id: string;
  status: string;
  output_pptx_path: string;
  report_path: string;
  page_count: number;
};

export type WorkflowMode = "ocr" | "preview";

export type PreviewBlockRole = "title" | "body" | "header" | "footer" | "page_number";

export type OverlayVisibility = {
  text: boolean;
  tables: boolean;
  images: boolean;
  shapes: boolean;
};

export type PreviewTextBlock = {
  id: string;
  kind: "text";
  bbox: [number, number, number, number] | number[];
  text: string;
  block_role: string;
  reading_order: number;
  excluded: boolean;
};

export type PreviewTableCell = {
  row: number;
  col: number;
  text: string;
  bbox: number[];
};

export type PreviewTable = {
  id: string;
  kind: "table";
  bbox: number[];
  row_count: number;
  col_count: number;
  cells: PreviewTableCell[];
  excluded: boolean;
};

export type PreviewImage = {
  id: string;
  kind: "image";
  bbox: number[];
  excluded: boolean;
};

export type PreviewShape = {
  id: string;
  kind: "shape";
  shape_kind: string;
  bbox: number[];
  excluded: boolean;
};

export type PreviewPage = {
  page: number;
  image_url: string;
  width_pt: number;
  height_pt: number;
  page_kind: string;
  background_mode: string;
  raster_fallback: boolean;
  text_blocks: PreviewTextBlock[];
  tables: PreviewTable[];
  images: PreviewImage[];
  shapes: PreviewShape[];
};

export type PreviewPageEdits = {
  page: number;
  excluded_ids: string[];
  role_overrides: Record<string, PreviewBlockRole>;
  raster_fallback: boolean;
};

export type PreviewEdits = {
  pages: PreviewPageEdits[];
};

export type PreviewResponse = {
  job_id: string;
  status: string;
  page_count: number;
  slide_width_pt: number;
  slide_height_pt: number;
  pages: PreviewPage[];
  edits: PreviewEdits;
  slideir_path?: string | null;
  slideir_assets_dir?: string | null;
};

export type FromIrJobResponse = {
  job_id: string;
  status: string;
  output_pptx_path: string;
  report_path: string;
  page_count: number;
  slideir_path?: string | null;
  edited_slideir_path?: string | null;
};

export type PreviewElementRef =
  | { kind: "text"; id: string }
  | { kind: "table"; id: string }
  | { kind: "image"; id: string }
  | { kind: "shape"; id: string }
  | null;

export type ApiErrorDetail = {
  code: string;
  message: string;
  page?: number | null;
};

export type ApiErrorResponse = {
  detail: ApiErrorDetail;
};

export type ResizeHandle = "nw" | "ne" | "se" | "sw";
export type BoxFilter = "all" | "empty-text" | "low-confidence";
export type BoxSort = "reading-order" | "confidence-asc" | "confidence-desc" | "source";
export type BoxGroup = "none" | "source";
export type EditorTool = "select" | "create";

export type DragState =
  | { kind: "move"; boxId: string; startX: number; startY: number; origin: [number, number, number, number] }
  | {
      kind: "resize";
      boxId: string;
      handle: ResizeHandle;
      startX: number;
      startY: number;
      origin: [number, number, number, number];
    }
  | { kind: "create"; startX: number; startY: number; currentX: number; currentY: number }
  | { kind: "select"; startX: number; startY: number; currentX: number; currentY: number }
  | null;

export type BoxGroupEntry = {
  key: string;
  label: string;
  boxes: Box[];
};