import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { ImagePlus } from 'lucide-react'
import ScanFrame from './ScanFrame'

const ACCEPTED_TYPES = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/webp': ['.webp'],
}

export default function UploadZone({ onFileSelected }) {
  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles.length > 0) onFileSelected(acceptedFiles[0])
    },
    [onFileSelected],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    multiple: false,
  })

  return (
    <ScanFrame tone={isDragActive ? 'scan' : 'neutral'}>
      <div
        {...getRootProps()}
        className="flex h-80 w-full cursor-pointer flex-col items-center justify-center gap-4 rounded-lg border border-dashed border-line bg-surface/50 px-6 text-center transition-colors hover:border-scan/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-scan"
      >
        <input {...getInputProps()} />
        <ImagePlus className="h-10 w-10 text-ink-muted" strokeWidth={1.5} />
        <div>
          <p className="font-display text-lg font-medium text-ink">
            {isDragActive ? 'Drop it to scan' : 'Drop an image to scan'}
          </p>
          <p className="mt-1 text-sm text-ink-muted">or click to browse</p>
        </div>
        <p className="font-mono text-xs uppercase tracking-wide text-ink-muted">JPEG · PNG · WEBP · up to 10MB</p>
      </div>
    </ScanFrame>
  )
}
