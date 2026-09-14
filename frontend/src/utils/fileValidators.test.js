import { describe, expect, it } from 'vitest'
import { validateImageFile } from './fileValidators'

function makeFile(name, type, sizeInBytes) {
  return new File([new Uint8Array(sizeInBytes)], name, { type })
}

describe('validateImageFile', () => {
  it('accepts a normal JPEG', () => {
    expect(validateImageFile(makeFile('photo.jpg', 'image/jpeg', 1024)).valid).toBe(true)
  })

  it('rejects a PDF disguised as an upload', () => {
    const result = validateImageFile(makeFile('document.pdf', 'application/pdf', 1024))
    expect(result.valid).toBe(false)
    expect(result.error).toMatch(/JPEG, PNG, or WEBP/)
  })

  it('rejects a file over the 10MB limit', () => {
    const result = validateImageFile(makeFile('huge.jpg', 'image/jpeg', 11 * 1024 * 1024))
    expect(result.valid).toBe(false)
    expect(result.error).toMatch(/10MB/)
  })
})
