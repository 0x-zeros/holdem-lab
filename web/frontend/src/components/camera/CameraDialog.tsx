import { useRef, useState, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { X, Camera, SwitchCamera, Loader2 } from 'lucide-react'

interface CameraDialogProps {
  isOpen: boolean
  onClose: () => void
  onCapture: (imageBlob: Blob) => void
  isProcessing?: boolean
}

export function CameraDialog({
  isOpen,
  onClose,
  onCapture,
  isProcessing = false,
}: CameraDialogProps) {
  const { t } = useTranslation()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment')
  const [isStarting, setIsStarting] = useState(false)

  const startCamera = useCallback(async () => {
    setIsStarting(true)
    setError(null)

    try {
      // Stop existing stream first
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      })

      setStream(mediaStream)

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
    } catch (err) {
      console.error('[Camera] Error starting camera:', err)
      if (err instanceof DOMException) {
        if (err.name === 'NotAllowedError') {
          setError(t('camera.permissionDenied'))
        } else if (err.name === 'NotFoundError') {
          setError(t('camera.notFound'))
        } else {
          setError(t('camera.failed'))
        }
      } else {
        setError(t('camera.failed'))
      }
    } finally {
      setIsStarting(false)
    }
  }, [facingMode, stream, t])

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
  }, [stream])

  // Start camera when dialog opens
  useEffect(() => {
    if (isOpen) {
      startCamera()
    } else {
      stopCamera()
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }
    }
  }, [isOpen]) // eslint-disable-line react-hooks/exhaustive-deps

  // Handle facing mode change
  useEffect(() => {
    if (isOpen && stream) {
      startCamera()
    }
  }, [facingMode]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleCapture = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || isProcessing) return

    const video = videoRef.current
    const canvas = canvasRef.current

    // Set canvas size to video size
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw video frame to canvas
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.drawImage(video, 0, 0)

    // Convert to blob
    canvas.toBlob(
      (blob) => {
        if (blob) {
          console.log('[Camera] Captured image:', Math.round(blob.size / 1024), 'KB')
          onCapture(blob)
        }
      },
      'image/jpeg',
      0.9
    )
  }, [isProcessing, onCapture])

  const toggleFacingMode = useCallback(() => {
    setFacingMode((prev) => (prev === 'user' ? 'environment' : 'user'))
  }, [])

  const handleClose = useCallback(() => {
    stopCamera()
    onClose()
  }, [stopCamera, onClose])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/90 flex flex-col z-50">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-black/50">
        <h2 className="text-white text-lg font-medium">{t('camera.capture')}</h2>
        <button
          onClick={handleClose}
          className="p-2 text-white hover:bg-white/10 rounded-full transition-colors"
          disabled={isProcessing}
        >
          <X size={24} />
        </button>
      </div>

      {/* Camera preview */}
      <div className="flex-1 relative flex items-center justify-center overflow-hidden">
        {error ? (
          <div className="text-center p-4">
            <p className="text-red-400 mb-4">{error}</p>
            <button
              onClick={startCamera}
              className="px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
            >
              {t('camera.retry')}
            </button>
          </div>
        ) : isStarting ? (
          <div className="text-white flex items-center gap-2">
            <Loader2 className="animate-spin" size={24} />
            <span>{t('camera.starting')}</span>
          </div>
        ) : (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="max-h-full max-w-full object-contain"
          />
        )}

        {/* Processing overlay */}
        {isProcessing && (
          <div className="absolute inset-0 bg-black/70 flex items-center justify-center">
            <div className="text-white flex flex-col items-center gap-3">
              <Loader2 className="animate-spin" size={40} />
              <span className="text-lg">{t('camera.recognizing')}</span>
            </div>
          </div>
        )}

        {/* Hidden canvas for capture */}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-8 py-6 bg-black/50">
        {/* Switch camera button */}
        <button
          onClick={toggleFacingMode}
          disabled={isProcessing || isStarting || !!error}
          className="p-4 text-white hover:bg-white/10 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title={t('camera.switchCamera')}
        >
          <SwitchCamera size={28} />
        </button>

        {/* Capture button */}
        <button
          onClick={handleCapture}
          disabled={isProcessing || isStarting || !!error || !stream}
          className="p-6 bg-white rounded-full hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title={t('camera.takePhoto')}
        >
          <Camera size={32} className="text-gray-900" />
        </button>

        {/* Spacer for symmetry */}
        <div className="w-[60px]" />
      </div>
    </div>
  )
}
