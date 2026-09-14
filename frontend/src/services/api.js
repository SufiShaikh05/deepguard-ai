import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({ baseURL: API_BASE_URL, timeout: 30000 })

export async function analyzeImage(imageFile) {
  const formData = new FormData()
  formData.append('file', imageFile) // must match FastAPI's `file: UploadFile` param name exactly

  const response = await apiClient.post('/api/v1/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })

  const data = response.data
  return {
    label: data.label,
    confidence: data.confidence,
    probabilities: data.probabilities,
    heatmapImage: data.heatmap_image,
  }
}
