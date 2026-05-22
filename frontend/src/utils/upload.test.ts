import { describe, expect, it } from "vitest";

import { isSupportedUpload } from "./upload";

describe("isSupportedUpload", () => {
  it("accepts pdf, png, and jpg files by extension", () => {
    expect(isSupportedUpload(new File(["pdf"], "deck.pdf", { type: "application/pdf" }))).toBe(true);
    expect(isSupportedUpload(new File(["png"], "slide.PNG", { type: "image/png" }))).toBe(true);
    expect(isSupportedUpload(new File(["jpg"], "photo.JPG", { type: "image/jpeg" }))).toBe(true);
    expect(isSupportedUpload(new File(["jpeg"], "photo.jpeg", { type: "image/jpeg" }))).toBe(true);
  });

  it("rejects unsupported file types", () => {
    expect(isSupportedUpload(new File(["txt"], "notes.txt", { type: "text/plain" }))).toBe(false);
  });
});
