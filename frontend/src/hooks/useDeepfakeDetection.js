import { useCallback, useState } from 'react'
import { analyzeImage } from '../services/api'
import { validateImageFile } from '../utils/fileValidators'

export function useDeepfakeDetection() {
  const [status, setStatus] = useState('idle') // 'idle' | 'scanning' | 'success'
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [errorMessage, setErrorMessage] = useState(null)

  const submitImage = useCallback(async (file) => {
    const validation = validateImageFile(file)
    if (!validation.valid) {
      setErrorMessage(validation.error)
      return
    }

    setErrorMessage(null)
    setPreviewUrl(URL.createObjectURL(file))
    setStatus('scanning')

    try {
      const analysis = await analyzeImage(file)
      setResult(analysis)
      setStatus('success')
    } catch (error) {
      console.error('Deepfake detection error:', error)
      const message =
        error.response?.data?.message || 'The scan failed. Check that the backend is running and try again.'
      setErrorMessage(message)
      setStatus('idle')
    }
  }, [])

  const reset = useCallback(() => {
    setStatus('idle')
    setResult(null)
    setErrorMessage(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
  }, [previewUrl])

  return { status, previewUrl, result, errorMessage, submitImage, reset }
}
