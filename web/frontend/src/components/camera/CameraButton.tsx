import { useState, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { Camera, Loader2 } from 'lucide-react'
import { CameraDialog } from './CameraDialog'
import { apiKeyStorage } from '../../utils/storage'

interface CameraButtonProps {
  onCapture: (blob: Blob) => Promise<void>
  disabled?: boolean
  className?: string
}

export function CameraButton({ onCapture, disabled = false, className = '' }: CameraButtonProps) {
  const { t } = useTranslation()
  const [showCamera, setShowCamera] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)

  const hasApiKey = apiKeyStorage.hasAnyApiKey()

  const handleCapture = useCallback(
    async (blob: Blob) => {
      setIsProcessing(true)
      try {
        await onCapture(blob)
        // Close camera dialog after successful capture
        setShowCamera(false)
      } catch (err) {
        console.error('[CameraButton] Capture error:', err)
        // Keep dialog open on error so user can retry
      } finally {
        setIsProcessing(false)
      }
    },
    [onCapture]
  )

  // Hide button if no API key is configured
  if (!hasApiKey) {
    return null
  }

  return (
    <>
      <button
        onClick={() => setShowCamera(true)}
        disabled={disabled || isProcessing}
        className={`p-2 hover:bg-[var(--muted)] rounded-[var(--radius-sm)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
        title={t('camera.capture')}
      >
        {isProcessing ? (
          <Loader2 size={20} className="animate-spin" />
        ) : (
          <Camera size={20} />
        )}
      </button>

      <CameraDialog
        isOpen={showCamera}
        onClose={() => setShowCamera(false)}
        onCapture={handleCapture}
        isProcessing={isProcessing}
      />
    </>
  )
}
