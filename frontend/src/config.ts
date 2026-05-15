export const detectConfidenceThreshold = 0.75;

export const INPAINT_ENGINE_OPTIONS = [
  { value: "opencv-fast", label: "OpenCV Fast (recommended)" },
  { value: "auto", label: "Auto routing" },
  { value: "white-box", label: "White box" },
  { value: "lama-onnx-cuda", label: "LaMa ONNX (GPU)" },
  { value: "lama-pytorch", label: "LaMa PyTorch (GPU)" },
] as const;

export type InpaintEngine = (typeof INPAINT_ENGINE_OPTIONS)[number]["value"];

export const defaultInpaintEngine: InpaintEngine = "opencv-fast";

export type ConvertRequestPayload = {
  write_debug_artifacts: boolean;
  inpaint_engine: InpaintEngine;
  inpaint_model_root?: string;
  inpaint_lama_repo_root?: string;
  inpaint_lama_device?: string;
};

export function buildConvertRequestPayload(
  inpaintEngine: InpaintEngine,
  writeDebugArtifacts = false,
): ConvertRequestPayload {
  const payload: ConvertRequestPayload = {
    write_debug_artifacts: writeDebugArtifacts,
    inpaint_engine: inpaintEngine,
  };
  if (inpaintEngine === "lama-pytorch") {
    payload.inpaint_model_root = "lama/big-lama";
    payload.inpaint_lama_repo_root = "lama";
    payload.inpaint_lama_device = "cuda";
  } else if (inpaintEngine === "lama-onnx-cuda") {
    payload.inpaint_model_root = "model/lama";
  }
  return payload;
}
