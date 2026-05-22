export const UPLOAD_ACCEPT = "application/pdf,image/png,image/jpeg,.pdf,.png,.jpg,.jpeg";

const SUPPORTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg"] as const;

const SUPPORTED_MIME_TYPES = new Set(["application/pdf", "image/png", "image/jpeg"]);

export function isSupportedUpload(file: File): boolean {
  const lowerName = file.name.toLowerCase();
  if (SUPPORTED_EXTENSIONS.some((extension) => lowerName.endsWith(extension))) {
    return true;
  }
  return file.type !== "" && SUPPORTED_MIME_TYPES.has(file.type);
}
