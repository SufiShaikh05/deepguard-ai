import Header from './components/Header'
import UploadZone from './components/UploadZone'
import ScanningStage from './components/ScanningStage'
import ResultsPanel from './components/ResultsPanel'
import ErrorBanner from './components/ErrorBanner'
import { useDeepfakeDetection } from './hooks/useDeepfakeDetection'

export default function App() {
  const { status, previewUrl, result, errorMessage, submitImage, reset } = useDeepfakeDetection()

  return (
    <div className="min-h-screen bg-void font-body text-ink">
      <Header />
      <main className="mx-auto max-w-3xl px-6 py-16 sm:px-10">
        {errorMessage && <ErrorBanner message={errorMessage} />}
        {status === 'idle' && <UploadZone onFileSelected={submitImage} />}
        {status === 'scanning' && <ScanningStage previewUrl={previewUrl} />}
        {status === 'success' && <ResultsPanel previewUrl={previewUrl} result={result} onReset={reset} />}
      </main>
    </div>
  )
}
