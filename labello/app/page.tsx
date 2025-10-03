"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { RefreshCw, Settings, Play, Pause, Camera, ChevronDown, X } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface Rectangle {
  x: number
  y: number
  width: number
  height: number
  label: string
  confidence?: number
}

interface DetectionResponse {
  rectangles: Rectangle[]
  timestamp: string
  fps: number
  model: string
}

export default function CameraPage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [facingMode, setFacingMode] = useState<"user" | "environment">("environment")
  const [rectangle, setRectangle] = useState<Rectangle>({ x: 20, y: 20, width: 120, height: 120, label: "Scene" })
  const [error, setError] = useState<string>("")
  const [fps, setFps] = useState<number>(0)
  const [isRunning, setIsRunning] = useState<boolean>(false)
  const [showSettings, setShowSettings] = useState<boolean>(false)
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false)
  const [selectedModel, setSelectedModel] = useState<string>("scene-v1")
  const [confidence, setConfidence] = useState<number>(75)
  const [showLabels, setShowLabels] = useState<boolean>(true)
  const [updateInterval, setUpdateInterval] = useState<number>(500)
  const [wsConnected, setWsConnected] = useState<boolean>(false)
  const [backendError, setBackendError] = useState<string>("")

  useEffect(() => {
    const startCamera = async () => {
      try {
        if (stream) {
          stream.getTracks().forEach((track) => track.stop())
        }

        // Check if mediaDevices is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("UNSUPPORTED_BROWSER")
        }

        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: facingMode,
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        })

        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream
        }
        setStream(mediaStream)
        setError("")
      } catch (err) {
        console.error("Error accessing camera:", err)
        
        // Provide more specific error messages
        if (err instanceof Error) {
          if (err.message === "UNSUPPORTED_BROWSER") {
            setError("Navigation non supportée. Utilisez HTTPS pour accéder à la caméra.")
          } else if (err.name === "NotAllowedError") {
            setError("Permission caméra refusée. Veuillez autoriser l'accès.")
          } else if (err.name === "NotFoundError") {
            setError("Aucune caméra trouvée sur cet appareil.")
          } else if (err.name === "NotReadableError") {
            setError("Caméra déjà utilisée par une autre application.")
          } else if (err.name === "OverconstrainedError") {
            setError("Contraintes caméra non supportées.")
          } else if (err.name === "SecurityError") {
            setError("Accès caméra bloqué. Utilisez HTTPS.")
          } else {
            setError("Impossible d'accéder à la caméra: " + err.message)
          }
        } else {
          setError("Impossible d'accéder à la caméra")
        }
      }
    }

    startCamera()

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }
    }
  }, [facingMode])

  // Connexion WebSocket
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws/detect')
        wsRef.current = ws

        ws.onopen = () => {
          console.log('WebSocket connecté')
          setWsConnected(true)
          setBackendError("")
        }

        ws.onmessage = (event) => {
          try {
            const response: DetectionResponse = JSON.parse(event.data)
            if (response.rectangles && response.rectangles.length > 0) {
              setRectangle(response.rectangles[0])
              setFps(response.fps)
            }
          } catch (err) {
            console.error('Erreur parsing message WebSocket:', err)
          }
        }

        ws.onclose = () => {
          console.log('WebSocket déconnecté')
          setWsConnected(false)
          setBackendError("Connexion au backend perdue")
          // Tentative de reconnexion après 3 secondes
          setTimeout(connectWebSocket, 3000)
        }

        ws.onerror = (error) => {
          console.error('Erreur WebSocket:', error)
          setBackendError("Impossible de se connecter au backend (port 8000)")
          setWsConnected(false)
        }
      } catch (err) {
        console.error('Erreur création WebSocket:', err)
        setBackendError("Erreur de connexion WebSocket")
      }
    }

    connectWebSocket()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  // Gestion du démarrage/arrêt de la détection via WebSocket
  useEffect(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return
    }

    if (isRunning && containerRef.current) {
      const containerWidth = containerRef.current.clientWidth
      const containerHeight = containerRef.current.clientHeight

      const message = {
        type: "start",
        width: containerWidth,
        height: containerHeight,
        model: selectedModel,
        interval: updateInterval
      }

      wsRef.current.send(JSON.stringify(message))
    } else {
      const message = { type: "stop" }
      wsRef.current.send(JSON.stringify(message))
      setFps(0)
    }
  }, [isRunning, selectedModel, updateInterval])

  // Mise à jour de la configuration WebSocket
  useEffect(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || !isRunning) {
      return
    }

    if (containerRef.current) {
      const containerWidth = containerRef.current.clientWidth
      const containerHeight = containerRef.current.clientHeight

      const message = {
        type: "config",
        width: containerWidth,
        height: containerHeight,
        model: selectedModel,
        interval: updateInterval
      }

      wsRef.current.send(JSON.stringify(message))
    }
  }, [selectedModel, updateInterval])

  const switchCamera = () => {
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"))
  }

  const toggleVision = () => {
    setIsRunning((prev) => !prev)
  }

  return (
    <div className="flex h-screen w-screen flex-col bg-background">
      <header className="flex items-center justify-between border-b border-border bg-card/50 px-4 py-3 backdrop-blur-sm md:px-6 md:py-4">
        <div className="flex items-center gap-2 md:gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary md:h-10 md:w-10">
            <Camera className="h-4 w-4 text-primary-foreground md:h-6 md:w-6" />
          </div>
          <div>
            <h1 className="text-base font-semibold text-foreground md:text-lg">Vision AI</h1>
            <p className="hidden text-xs text-muted-foreground md:block">Scene Labelling</p>
          </div>
        </div>
        <div className="flex items-center gap-2 md:gap-3">
          {isRunning && (
            <div className="flex items-center gap-2 rounded-lg bg-muted px-2 py-1 md:px-3 md:py-1.5">
              <div className="h-2 w-2 animate-pulse rounded-full bg-green-500" />
              <span className="text-xs text-muted-foreground md:text-sm">{fps} FPS</span>
            </div>
          )}
          
          <div className="flex items-center gap-2 rounded-lg bg-muted px-2 py-1 md:px-3 md:py-1.5">
            <div className={`h-2 w-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-xs text-muted-foreground md:text-sm">
              {wsConnected ? 'Backend' : 'Offline'}
            </span>
          </div>
          
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 md:h-9 md:w-9"
            onClick={() => setShowSettings(!showSettings)}
          >
            {showSettings ? <X className="h-4 w-4 md:h-5 md:w-5" /> : <Settings className="h-4 w-4 md:h-5 md:w-5" />}
          </Button>
        </div>
      </header>

      <div className="relative flex flex-1 overflow-hidden">
        <div className="relative flex-1 overflow-hidden bg-black p-2 md:p-4">
          <Card className="relative h-full overflow-hidden border-2 border-border bg-black">
            <div ref={containerRef} className="relative h-full w-full">
              <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-contain" />

              {isRunning && (
                <div
                  className="absolute border-2 border-red-500 shadow-lg shadow-red-500/50 transition-all duration-300 ease-out"
                  style={{
                    left: `${rectangle.x}px`,
                    top: `${rectangle.y}px`,
                    width: `${rectangle.width}px`,
                    height: `${rectangle.height}px`,
                  }}
                >
                  {showLabels && (
                    <div className="absolute -top-6 left-0 rounded bg-red-500 px-2 py-0.5 text-xs font-medium text-white">
                      {rectangle.label}
                    </div>
                  )}
                </div>
              )}

              {(error || backendError) && (
                <div className="absolute left-1/2 top-4 -translate-x-1/2 rounded-lg bg-destructive/90 px-4 py-2 text-xs text-destructive-foreground backdrop-blur-sm md:px-6 md:py-3 md:text-sm">
                  <div className="text-center">
                    <div className="font-medium">{error || backendError}</div>
                    {error && error.includes("HTTPS") && (
                      <div className="mt-1 text-xs opacity-90">
                        Pour mobile: utilisez https://192.168.1.79:3000
                      </div>
                    )}
                    {backendError && (
                      <div className="mt-1 text-xs opacity-90">
                        Assurez-vous que le backend est démarré sur le port 8000
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-2 md:gap-3">
                <Button
                  onClick={toggleVision}
                  size="icon"
                  className={`h-12 w-12 rounded-full shadow-lg md:h-14 md:w-14 ${
                    isRunning ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"
                  }`}
                >
                  {isRunning ? <Pause className="h-5 w-5 md:h-6 md:w-6" /> : <Play className="h-5 w-5 md:h-6 md:w-6" />}
                </Button>

                <Button
                  onClick={switchCamera}
                  size="icon"
                  className="h-10 w-10 rounded-full bg-card/80 shadow-lg backdrop-blur-md hover:bg-card md:h-12 md:w-12"
                >
                  <RefreshCw className="h-4 w-4 md:h-5 md:w-5" />
                </Button>

                <div className="hidden rounded-full bg-card/60 px-3 py-1.5 text-xs text-muted-foreground backdrop-blur-md md:block md:px-4 md:py-2">
                  {facingMode === "user" ? "Avant" : "Arrière"}
                </div>
              </div>
            </div>
          </Card>
        </div>

        {showSettings && (
          <>
            <div
              className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm md:hidden"
              onClick={() => setShowSettings(false)}
            />

            <Card className="fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-[90vw] max-w-md -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-2xl border-2 border-border p-5 shadow-2xl md:relative md:left-auto md:top-auto md:z-0 md:max-h-none md:w-80 md:translate-x-0 md:translate-y-0 md:rounded-none md:border-l-2 md:border-t-0 md:p-6">
              <button
                onClick={() => setShowSettings(false)}
                className="absolute right-4 top-4 rounded-full p-1 hover:bg-muted md:hidden"
              >
                <X className="h-5 w-5" />
              </button>

              <div className="space-y-6">
                <div>
                  <h2 className="mb-4 text-lg font-semibold text-foreground">Paramètres</h2>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="model" className="text-sm font-medium">
                        Modèle de détection
                      </Label>
                      <Select value={selectedModel} onValueChange={setSelectedModel}>
                        <SelectTrigger id="model" className="w-full">
                          <SelectValue placeholder="Sélectionner un modèle" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="scene-v1">Scene Recognition v1</SelectItem>
                          <SelectItem value="places365">Places365 (MIT)</SelectItem>
                          <SelectItem value="urban-v2">Urban Scenes v2</SelectItem>
                          <SelectItem value="nature-v1">Nature Classifier</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">Choisissez le modèle adapté à votre environnement</p>
                    </div>

                    <div className="flex items-center justify-between rounded-lg border border-border p-3">
                      <Label htmlFor="labels" className="text-sm">
                        Afficher les labels
                      </Label>
                      <Switch id="labels" checked={showLabels} onCheckedChange={setShowLabels} />
                    </div>
                  </div>
                </div>

                <div className="border-t border-border pt-4">
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex w-full items-center justify-between text-sm font-medium text-foreground hover:text-foreground/80"
                  >
                    <span>Paramètres avancés</span>
                    <ChevronDown className={`h-4 w-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
                  </button>

                  {showAdvanced && (
                    <div className="mt-4 space-y-4">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label htmlFor="confidence" className="text-xs text-muted-foreground">
                            Seuil de confiance
                          </Label>
                          <span className="text-xs font-medium text-muted-foreground">{confidence}%</span>
                        </div>
                        <Slider
                          id="confidence"
                          min={0}
                          max={100}
                          step={5}
                          value={[confidence]}
                          onValueChange={(value) => setConfidence(value[0])}
                          className="w-full"
                        />
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label htmlFor="interval" className="text-xs text-muted-foreground">
                            Intervalle de mise à jour
                          </Label>
                          <span className="text-xs font-medium text-muted-foreground">{updateInterval}ms</span>
                        </div>
                        <Slider
                          id="interval"
                          min={100}
                          max={2000}
                          step={100}
                          value={[updateInterval]}
                          onValueChange={(value) => setUpdateInterval(value[0])}
                          className="w-full"
                        />
                      </div>

                      <div className="rounded-lg bg-muted/50 p-3">
                        <p className="text-xs text-muted-foreground">
                          Ces paramètres sont destinés aux utilisateurs avancés pour ajuster les performances et la
                          précision du modèle.
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="border-t border-border pt-4">
                  <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Informations
                  </h3>
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center justify-between rounded-lg bg-muted/50 p-2">
                      <span className="text-muted-foreground">Statut</span>
                      <span className={`font-medium ${isRunning ? "text-green-500" : "text-muted-foreground"}`}>
                        {isRunning ? "Actif" : "Inactif"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg bg-muted/50 p-2">
                      <span className="text-muted-foreground">Backend</span>
                      <span className={`font-medium ${wsConnected ? "text-green-500" : "text-red-500"}`}>
                        {wsConnected ? "Connecté" : "Déconnecté"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg bg-muted/50 p-2">
                      <span className="text-muted-foreground">Caméra</span>
                      <span className="font-medium text-foreground">{facingMode === "user" ? "Avant" : "Arrière"}</span>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
