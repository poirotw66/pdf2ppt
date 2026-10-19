export const detectConfidenceThreshold = 0.75;

export const INPAINT_ENGINE_OPTIONS = [
  { value: "opencv-fast", label: "OpenCV Fast (recommended)" },
  { value: "auto", label: "Auto routing" },
  { value: "white-box", label: "White box" },
  { value: "lama-pytorch-hybrid", label: "LaMa PyTorch Hybrid (GPU, recommended)" },
  { value: "lama-pytorch", label: "LaMa PyTorch (GPU)" },
  { value: "generative-openai", label: "Generative OpenAI (opt-in, API key)" },
  { value: "generative-gemini", label: "Generative Gemini / Nano Banana (opt-in, API key)" },
] as const;

export type InpaintEngine = (typeof INPAINT_ENGINE_OPTIONS)[number]["value"];

export const defaultInpaintEngine: InpaintEngine = "lama-pytorch";

export const lamaInpaintPaddingPx = 8;

export const generativeInpaintPaddingPx = 8;

export type ConvertRequestPayload = {
  write_debug_artifacts: boolean;
  inpaint_engine: InpaintEngine;
  inpaint_padding_px?: number;
  inpaint_model_root?: string;
  inpaint_lama_repo_root?: string;
  inpaint_lama_device?: string;
  generative_max_calls?: number;
  generative_max_cost_usd?: number;
  generative_verify_ocr?: boolean;
};

export function buildConvertRequestPayload(
  inpaintEngine: InpaintEngine,
  writeDebugArtifacts = false,
): ConvertRequestPayload {
  const payload: ConvertRequestPayload = {
    write_debug_artifacts: writeDebugArtifacts,
    inpaint_engine: inpaintEngine,
  };
  if (inpaintEngine === "lama-pytorch" || inpaintEngine === "lama-pytorch-hybrid") {
    payload.inpaint_padding_px = lamaInpaintPaddingPx;
    payload.inpaint_model_root = "lama/big-lama";
    payload.inpaint_lama_repo_root = "lama";
    payload.inpaint_lama_device = "cuda";
  }
  if (inpaintEngine === "generative-openai" || inpaintEngine === "generative-gemini") {
    // Cloud generative engines benefit from slightly wider masks so OCR under-boxes
    // and anti-aliased edges are covered; keys come from server env vars.
    payload.inpaint_padding_px = generativeInpaintPaddingPx;
    payload.generative_verify_ocr = true;
  }
  return payload;
}
