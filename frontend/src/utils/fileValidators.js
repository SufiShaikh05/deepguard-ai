const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 // 10 MB - matches the backend limit

export function validateImageFile(file) {
  if (!file) return { valid: false, error: 'No file selected.' }

  if (!ALLOWED_TYPES.includes(file.type)) {
    return {
      valid: false,
      error: `Unsupported file type. Please upload a JPEG, PNG, or WEBP image instead of a ${file.type || 'this'} file.`,
    }
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      valid: false,
      error: `That file is ${(file.size / 1024 / 1024).toFixed(1)}MB - the limit is 10MB.`,
    }
  }

  return { valid: true }
}
