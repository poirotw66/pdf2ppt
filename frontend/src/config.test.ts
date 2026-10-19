import { describe, expect, it } from "vitest";

import { buildConvertRequestPayload, lamaInpaintPaddingPx } from "./config";

describe("buildConvertRequestPayload", () => {
  it("requests wider inpaint padding for LaMa engines", () => {
    expect(buildConvertRequestPayload("lama-pytorch").inpaint_padding_px).toBe(lamaInpaintPaddingPx);
    expect(buildConvertRequestPayload("lama-pytorch-hybrid").inpaint_padding_px).toBe(lamaInpaintPaddingPx);
    expect(buildConvertRequestPayload("opencv-fast").inpaint_padding_px).toBeUndefined();
  });

  it("sends distinct inpaint_engine values for pure and hybrid LaMa modes", () => {
    expect(buildConvertRequestPayload("lama-pytorch").inpaint_engine).toBe("lama-pytorch");
    expect(buildConvertRequestPayload("lama-pytorch-hybrid").inpaint_engine).toBe("lama-pytorch-hybrid");
  });

  it("enables OCR verification and wider padding for generative engines", () => {
    expect(buildConvertRequestPayload("generative-openai").generative_verify_ocr).toBe(true);
    expect(buildConvertRequestPayload("generative-openai").inpaint_padding_px).toBe(8);
    expect(buildConvertRequestPayload("generative-gemini").inpaint_engine).toBe("generative-gemini");
    expect(buildConvertRequestPayload("opencv-fast").generative_verify_ocr).toBeUndefined();
  });
});
