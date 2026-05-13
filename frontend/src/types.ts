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