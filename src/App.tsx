import { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Users, 
  AlertTriangle, 
  Trash2, 
  Settings, 
  Wifi, 
  Accessibility,
  Volume2,
  VolumeX,
  Loader2
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { HandLandmarker, FilesetResolver, HandLandmarkerResult } from '@mediapipe/tasks-vision';

export default function App() {
  const [isTracking, setIsTracking] = useState(true);
  const [detectedWord, setDetectedWord] = useState('');
  const [isSpeechEnabled, setIsSpeechEnabled] = useState(true);
  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastVideoTimeRef = useRef(-1);
  const requestRef = useRef<number>(null);
  const detectedWordRef = useRef('');

  // Initialize MediaPipe Hand Landmarker
  useEffect(() => {
    async function initMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });
        setHandLandmarker(landmarker);
        setIsLoading(false);
      } catch (err) {
        console.error("Error initializing MediaPipe:", err);
        setIsLoading(false);
      }
    }
    initMediaPipe();
  }, []);

  // Setup Camera
  useEffect(() => {
    let currentStream: MediaStream | null = null;
    async function setupCamera() {
      try {
        currentStream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            facingMode: 'user', 
            width: { ideal: 1280 }, 
            height: { ideal: 720 } 
          },
          audio: false 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = currentStream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
          };
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    }
    setupCamera();
    return () => {
      if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Speech Synthesis
  const speak = useCallback((text: string) => {
    if (!isSpeechEnabled || !text) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  }, [isSpeechEnabled]);

  // Robust Gesture Recognition Logic
  const recognizeGesture = (result: HandLandmarkerResult) => {
    if (!result.landmarks || result.landmarks.length === 0) return null;

    const landmarks = result.landmarks[0];
    
    // Finger indices: 
    // Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
    
    const isFingerUp = (tip: number, pip: number, mcp: number) => {
      // More robust: tip must be significantly above pip and mcp
      return landmarks[tip].y < landmarks[pip].y && landmarks[tip].y < landmarks[mcp].y;
    };

    const indexUp = isFingerUp(8, 6, 5);
    const middleUp = isFingerUp(12, 10, 9);
    const ringUp = isFingerUp(16, 14, 13);
    const pinkyUp = isFingerUp(20, 18, 17);
    
    // Thumb is tricky, use horizontal distance relative to index MCP
    const thumbUp = Math.abs(landmarks[4].x - landmarks[5].x) > 0.1;

    // Gesture definitions
    if (indexUp && middleUp && ringUp && pinkyUp && thumbUp) return "HELLO";
    if (indexUp && middleUp && ringUp && !pinkyUp) return "WATER";
    if (indexUp && !middleUp && !ringUp && !pinkyUp) return "POINT";
    if (!indexUp && !middleUp && !ringUp && !pinkyUp) return "YES"; // Fist
    if (indexUp && pinkyUp && !middleUp && !ringUp) return "LOVE"; // ILY sign
    if (thumbUp && !indexUp && !middleUp && !ringUp && !pinkyUp) return "GOOD"; // Thumbs up
    
    return null;
  };

  // Detection Loop
  const predictWebcam = useCallback(() => {
    if (!videoRef.current || !handLandmarker || !isTracking) {
      requestRef.current = requestAnimationFrame(predictWebcam);
      return;
    }

    if (videoRef.current.readyState >= 2) {
      const startTimeMs = performance.now();
      if (lastVideoTimeRef.current !== videoRef.current.currentTime) {
        lastVideoTimeRef.current = videoRef.current.currentTime;
        const result = handLandmarker.detectForVideo(videoRef.current, startTimeMs);
        
        const gesture = recognizeGesture(result);
        if (gesture && gesture !== detectedWordRef.current) {
          detectedWordRef.current = gesture;
          setDetectedWord(gesture);
          speak(gesture);
        } else if (!gesture && detectedWordRef.current !== '') {
          // Optional: clear after some time of no gesture
          // For now, keep the last detected word
        }

        // Draw landmarks on canvas
        if (canvasRef.current && result.landmarks) {
          const ctx = canvasRef.current.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            
            result.landmarks.forEach(hand => {
              // Draw connections
              ctx.strokeStyle = "#f9f506";
              ctx.lineWidth = 3;
              ctx.lineJoin = "round";
              
              const drawPath = (indices: number[]) => {
                ctx.beginPath();
                indices.forEach((idx, i) => {
                  const x = hand[idx].x * canvasRef.current!.width;
                  const y = hand[idx].y * canvasRef.current!.height;
                  if (i === 0) ctx.moveTo(x, y);
                  else ctx.lineTo(x, y);
                });
                ctx.stroke();
              };

              // Thumb
              drawPath([0, 1, 2, 3, 4]);
              // Fingers
              drawPath([0, 5, 6, 7, 8]);
              drawPath([0, 9, 10, 11, 12]);
              drawPath([0, 13, 14, 15, 16]);
              drawPath([0, 17, 18, 19, 20]);
              // Palm base
              drawPath([5, 9, 13, 17]);

              // Draw points
              ctx.fillStyle = "#007aff";
              hand.forEach(landmark => {
                const x = landmark.x * canvasRef.current!.width;
                const y = landmark.y * canvasRef.current!.height;
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, 2 * Math.PI);
                ctx.fill();
              });
            });
          }
        }
      }
    }
    requestRef.current = requestAnimationFrame(predictWebcam);
  }, [handLandmarker, isTracking, speak]);

  useEffect(() => {
    requestRef.current = requestAnimationFrame(predictWebcam);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [predictWebcam]);

  const toggleTracking = () => setIsTracking(!isTracking);
  const clearText = () => {
    detectedWordRef.current = '';
    setDetectedWord('');
  };
  const toggleSpeech = () => setIsSpeechEnabled(!isSpeechEnabled);

  return (
    <div className="min-h-screen flex flex-col bg-background-dark text-white font-display overflow-hidden">
      {/* Header */}
      <header className="w-full border-b border-white/10 bg-background-dark/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-primary p-2 rounded-lg">
              <Users className="text-background-dark w-6 h-6" strokeWidth={3} />
            </div>
            <h1 className="text-xl md:text-2xl font-bold tracking-tight">
              The Communication Bridge
            </h1>
          </div>
          <button className="bg-sos-red hover:bg-red-600 text-white px-6 md:px-8 py-2 md:py-3 rounded-xl font-bold text-lg md:text-xl shadow-lg transition-all active:scale-95 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 md:w-6 md:h-6" />
            SOS
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center p-4 md:p-6 max-w-5xl mx-auto w-full gap-6 md:gap-8">
        
        {/* Viewfinder */}
        <div className="relative w-full aspect-video bg-black rounded-2xl overflow-hidden shadow-2xl border-4 border-white/5 group">
          {/* Camera Feed */}
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            className="w-full h-full object-cover opacity-80 scale-x-[-1]"
          />
          
          {/* Hand Tracking Canvas Overlay */}
          <canvas 
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none scale-x-[-1]"
            width={1280}
            height={720}
          />
          
          {/* Loading Overlay */}
          {isLoading && (
            <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center gap-4 z-20">
              <Loader2 className="w-12 h-12 text-primary animate-spin" />
              <p className="text-primary font-bold tracking-widest uppercase text-sm">Initializing AI Models...</p>
            </div>
          )}

          {/* Tracking Status Tag */}
          <div className="absolute top-4 left-4 md:top-6 md:left-6 flex items-center gap-2 bg-black/60 backdrop-blur-md px-3 md:px-4 py-1.5 md:py-2 rounded-full border border-white/20 z-10">
            <motion.div 
              animate={{ opacity: [1, 0.4, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              className={`w-2 h-2 rounded-full ${isTracking && handLandmarker ? 'bg-green-500' : 'bg-red-500'}`} 
            />
            <span className="text-white text-[10px] md:text-xs font-bold uppercase tracking-widest">
              {isLoading ? 'Loading...' : isTracking ? 'AI Tracking Active' : 'Tracking Paused'}
            </span>
          </div>

          {/* Waveform Animation */}
          <div className="absolute bottom-4 md:bottom-8 left-1/2 -translate-x-1/2 flex items-end gap-1.5 h-10 md:h-14 z-10">
            {[0.4, 0.7, 1, 0.6, 0.9, 0.5].map((opacity, i) => (
              <motion.div
                key={i}
                className="w-1.5 bg-accent-blue rounded-full"
                animate={{ 
                  height: isTracking ? [10, 30 + (i * 5), 15] : 4 
                }}
                transition={{ 
                  duration: 0.6 + (i * 0.1), 
                  repeat: Infinity, 
                  ease: "easeInOut" 
                }}
                style={{ opacity }}
              />
            ))}
          </div>
        </div>

        {/* Translation Box */}
        <motion.div 
          layout
          className="w-full bg-black rounded-2xl p-8 md:p-12 flex flex-col items-center justify-center shadow-xl border-t-4 border-primary relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none" />
          <span className="text-white/40 text-xs md:text-sm font-bold uppercase tracking-[0.3em] mb-4 relative z-10">
            Detected Word
          </span>
          <AnimatePresence mode="wait">
            <motion.div 
              key={detectedWord}
              initial={{ opacity: 0, scale: 0.8, filter: 'blur(10px)' }}
              animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
              exit={{ opacity: 0, scale: 1.2, filter: 'blur(10px)' }}
              className="text-primary text-6xl md:text-9xl font-black tracking-tighter leading-none relative z-10 text-center drop-shadow-[0_0_30px_rgba(249,245,6,0.3)]"
            >
              {detectedWord || '...'}
            </motion.div>
          </AnimatePresence>
        </motion.div>

        {/* Controls */}
        <div className="grid grid-cols-3 gap-4 md:gap-12 w-full max-w-3xl">
          {/* Toggle Speech */}
          <div className="flex flex-col items-center gap-3 md:gap-4 group">
            <button 
              onClick={toggleSpeech}
              className={`w-16 h-16 md:w-24 md:h-24 rounded-full flex items-center justify-center shadow-lg transition-all active:scale-90 ${
                isSpeechEnabled ? 'bg-primary shadow-primary/20' : 'bg-white/10'
              }`}
            >
              {isSpeechEnabled ? (
                <Volume2 className="w-8 h-8 md:w-12 md:h-12 font-bold text-background-dark" />
              ) : (
                <VolumeX className="w-8 h-8 md:w-12 md:h-12 font-bold text-white" />
              )}
            </button>
            <span className="text-[10px] md:text-xs font-bold uppercase tracking-widest text-white/50 text-center">
              {isSpeechEnabled ? 'Speech ON' : 'Speech OFF'}
            </span>
          </div>

          {/* Clear Text */}
          <div className="flex flex-col items-center gap-3 md:gap-4 group">
            <button 
              onClick={clearText}
              className="w-16 h-16 md:w-24 md:h-24 rounded-full bg-white/5 hover:bg-primary transition-all active:scale-90 flex items-center justify-center shadow-lg group-hover:shadow-primary/20 group"
            >
              <Trash2 className="w-8 h-8 md:w-12 md:h-12 text-white group-hover:text-background-dark transition-colors" />
            </button>
            <span className="text-[10px] md:text-xs font-bold uppercase tracking-widest text-white/50 text-center">
              Clear Text
            </span>
          </div>

          {/* Settings / Tracking Toggle */}
          <div className="flex flex-col items-center gap-3 md:gap-4 group">
            <button 
              onClick={toggleTracking}
              className={`w-16 h-16 md:w-24 md:h-24 rounded-full flex items-center justify-center shadow-lg transition-all active:scale-90 ${
                isTracking ? 'bg-white/5 hover:bg-primary' : 'bg-red-500/20 border border-red-500/50'
              }`}
            >
              <Settings className={`w-8 h-8 md:w-12 md:h-12 ${isTracking ? 'text-white group-hover:text-background-dark' : 'text-red-500'}`} />
            </button>
            <span className="text-[10px] md:text-xs font-bold uppercase tracking-widest text-white/50 text-center">
              {isTracking ? 'Tracking ON' : 'Tracking OFF'}
            </span>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="w-full px-6 py-4 flex flex-col md:flex-row justify-between items-center gap-4 text-white/40 bg-black/20 border-t border-white/5">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Wifi className="text-green-500 w-4 h-4" />
            <span className="text-[10px] font-bold uppercase tracking-wider">System Online</span>
          </div>
          <div className="flex items-center gap-2">
            <Accessibility className="text-white/40 w-4 h-4" />
            <span className="text-[10px] font-bold uppercase tracking-wider">Accessibility: High</span>
          </div>
        </div>
        <div className="text-[10px] font-bold uppercase tracking-widest">
          Version 2.4.0-Stable
        </div>
      </footer>
    </div>
  );
}
