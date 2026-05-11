<template>
  <div class="vision-section">
    <div class="controls">
      <button @click="startCapture" v-if="!isCapturing" class="btn-primary">选择窗口流</button>
      <template v-else>
        <button @click="toggleCalibration" :class="{ 'btn-danger': isCalibrating }">
          {{ isCalibrating ? '取消标定' : '手动标定' }}
        </button>
        
        <label class="switch-container">
          <input type="checkbox" v-model="showGrid">
          <span class="switch-text">显示网格</span>
        </label>

        <div class="status-badge">
            <span class="dot" :class="{ 'pulse': currentActiveSeat !== null }"></span>
            {{ currentActiveSeat ? currentActiveSeat + '号发言' : '无人发言' }}
        </div>
      </template>
    </div>

    <div class="canvas-wrapper">
      <video ref="videoRef" autoplay muted playsinline style="display: none;"></video>
      <canvas ref="canvasRef" width="1280" height="720"></canvas>

      <div v-if="showGrid && !isCalibrating" class="preview-layer">
        <div 
          v-for="(box, index) in previewRois.seats" 
          :key="'seat-' + index"
          class="grid-box seat-box"
          :class="{ 'active-glow': currentActiveSeat === index + 1 }"
          :style="getBoxStyle(box)"
        >
          <span class="box-tag">{{ index + 1 }}</span>
        </div>
        
        <div 
          v-if="previewRois.speaker_ui" 
          class="grid-box ui-box" 
          :style="getBoxStyle(previewRois.speaker_ui)"
        >
          <span class="box-tag">UI区</span>
        </div>
      </div>

      <div v-if="isCalibrating" class="calibration-overlay" @mousedown="onMouseDown" @mousemove="onMouseMove" @mouseup="onMouseUp">
        <div class="guide-banner">
          <span v-if="!hasDrawn">步骤 {{ calStep + 1 }}/4: 框选 {{ stepLabels[calStep] }}</span>
          <span v-else class="confirm-text">已选定，请确认位置</span>
        </div>
        <div v-if="isDrawing" class="drawing-rect" :style="drawingBoxStyle"></div>
        <div v-if="hasDrawn && tempBox" class="temp-rect" :style="getBoxStyle(tempBox)">
          <div class="confirm-actions">
            <button class="btn-confirm" @click.stop="confirmStep">确认</button>
            <button class="btn-retry" @click.stop="retryStep">重画</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onUnmounted } from 'vue';

const emit = defineEmits(['speech-data']);

const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const isCapturing = ref(false);
const currentActiveSeat = ref<number | null>(null);
let socket: WebSocket | null = null;
let animationId: number | null = null;
let lastSendTime = 0;

const showGrid = ref(true); 
const previewRois = ref({ seats: [], speaker_ui: null });

const isCalibrating = ref(false);
const calStep = ref(0);
const stepLabels = ['1号位', '2号位', '4号位', '提示区'];
const keys = ['seat_1', 'seat_2', 'seat_4', 'speaker_ui'];
const results = ref<Record<string, number[]>>({});

const isDrawing = ref(false);
const hasDrawn = ref(false);
const tempBox = ref<number[] | null>(null);
const startPos = { x: 0, y: 0 };
const currentPos = ref({ x: 0, y: 0 });

const getBoxStyle = (box: number[]) => ({
  left: box[0] + 'px',
  top: box[1] + 'px',
  width: box[2] + 'px',
  height: box[3] + 'px'
});

const syncCurrentConfig = async () => {
  try {
    const res = await fetch('http://localhost:8000/api/v1/calibration/current');
    const data = await res.json();
    if (data.status === 'success') previewRois.value = data.config_preview;
  } catch (err) { console.error("配置同步失败", err); }
};

const toggleCalibration = () => { isCalibrating.value = !isCalibrating.value; resetCalibState(); };
const resetCalibState = () => { calStep.value = 0; results.value = {}; hasDrawn.value = false; tempBox.value = null; };

const onMouseDown = (e: MouseEvent) => {
  if (hasDrawn.value) return; 
  isDrawing.value = true;
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
  startPos.x = e.clientX - rect.left; startPos.y = e.clientY - rect.top;
  currentPos.value = { ...startPos };
};

const onMouseMove = (e: MouseEvent) => {
  if (!isDrawing.value) return;
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
  currentPos.value.x = e.clientX - rect.left; currentPos.value.y = e.clientY - rect.top;
};

const onMouseUp = () => {
  if (!isDrawing.value) return;
  isDrawing.value = false;
  const x = Math.min(startPos.x, currentPos.value.x);
  const y = Math.min(startPos.y, currentPos.value.y);
  const w = Math.abs(currentPos.value.x - startPos.x);
  const h = Math.abs(currentPos.value.y - startPos.y);
  if (w > 5 && h > 5) {
    tempBox.value = [Math.round(x), Math.round(y), Math.round(w), Math.round(h)];
    hasDrawn.value = true;
  }
};

const confirmStep = async () => {
  // 1. 先获取 key 并确保它是有效的
  const currentKey = keys[calStep.value]
  
  // 2. 增加安全检查，确保 key 和 tempBox 都存在
  if (currentKey && tempBox.value) {
    results.value[currentKey] = [...tempBox.value]
  }

  hasDrawn.value = false
  tempBox.value = null

  if (calStep.value < 3) {
    calStep.value++
  } else {
    await submitCalibration()
  }
}

const retryStep = () => { hasDrawn.value = false; tempBox.value = null; };

const submitCalibration = async () => {
  const res = await fetch('http://localhost:8000/api/v1/calibrate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(results.value)
  });
  const data = await res.json();
  if (data.status === 'success') {
    previewRois.value = data.config_preview;
    isCalibrating.value = false;
  }
};

const drawingBoxStyle = computed(() => ({
  left: Math.min(startPos.x, currentPos.value.x) + 'px',
  top: Math.min(startPos.y, currentPos.value.y) + 'px',
  width: Math.abs(currentPos.value.x - startPos.x) + 'px',
  height: Math.abs(currentPos.value.y - startPos.y) + 'px'
}));

const startCapture = async () => {
  await syncCurrentConfig();
  const stream = await navigator.mediaDevices.getDisplayMedia({ video: { width: 1280, height: 720 } });
  if (videoRef.value) {
    videoRef.value.srcObject = stream;
    isCapturing.value = true;
    renderLoop();
    socket = new WebSocket("ws://localhost:8000/api/v1/ws/stream");
    socket.onmessage = (e) => {
      const res = JSON.parse(e.data);
      if (res.active_seat !== undefined) currentActiveSeat.value = res.active_seat;
      if (res.content) emit('speech-data', res); // 发送语音识别结果给 App.vue
    };
  }
};

const renderLoop = () => {
  if (!videoRef.value || !canvasRef.value) return;
  const ctx = canvasRef.value.getContext('2d');
  if (ctx) {
    ctx.drawImage(videoRef.value, 0, 0, 1280, 720);
    const now = Date.now();
    if (!isCalibrating.value && now - lastSendTime > 400 && socket?.readyState === WebSocket.OPEN) {
      canvasRef.value.toBlob((blob) => { if (blob) socket?.send(blob); }, 'image/jpeg', 0.7);
      lastSendTime = now;
    }
  }
  animationId = requestAnimationFrame(renderLoop);
};

onUnmounted(() => { if (animationId) cancelAnimationFrame(animationId); socket?.close(); });
</script>

<style scoped>
.vision-section { flex: 1; display: flex; flex-direction: column; background: #121212; }
.controls { display: flex; gap: 15px; padding: 15px; background: #1a1a1a; border-bottom: 1px solid #333; }
.canvas-wrapper { position: relative; width: 1280px; height: 720px; overflow: hidden; }
.preview-layer { position: absolute; inset: 0; pointer-events: none; }
.grid-box { position: absolute; border: 1px solid rgba(66, 184, 131, 0.3); background: rgba(66, 184, 131, 0.05); }
.box-tag { position: absolute; bottom: 2px; right: 4px; font-size: 10px; color: #42b883; font-weight: 800; }
.active-glow { border: 2px solid #00ff00 !important; background: rgba(0, 255, 0, 0.1) !important; box-shadow: 0 0 15px rgba(0,255,0,0.2); }
.status-badge { margin-left: auto; display: flex; align-items: center; gap: 8px; color: #42b883; font-size: 14px; font-weight: bold; }
.dot { width: 8px; height: 8px; background: #444; border-radius: 50%; }
.pulse { background: #00ff00; box-shadow: 0 0 8px #00ff00; animation: blink 1s infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

/* 标定层 */
.calibration-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.6); cursor: crosshair; z-index: 100; }
.guide-banner { position: absolute; top: 20px; left: 50%; transform: translateX(-50%); background: #42b883; color: white; padding: 10px 25px; border-radius: 50px; font-weight: bold; }
.drawing-rect { position: absolute; border: 2px dashed #00ff00; background: rgba(0, 255, 0, 0.1); pointer-events: none; }
.temp-rect { position: absolute; border: 2px solid #f39c12; background: rgba(243, 156, 18, 0.2); }
.saved-rect { position: absolute; border: 2px solid #ff4444; background: rgba(255, 68, 68, 0.1); }
.saved-rect .box-tag { color: #ff4444; }

.confirm-actions { position: absolute; bottom: -40px; left: 50%; transform: translateX(-50%); display: flex; gap: 8px; }
.btn-confirm { background: #2ecc71; color: white; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.btn-retry { background: #e74c3c; color: white; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }

button { padding: 8px 18px; border-radius: 6px; border: none; font-weight: bold; cursor: pointer; }
.btn-primary { background: #42b883; color: white; }
.btn-danger { background: #e74c3c; color: white; }
.switch-container { display: flex; align-items: center; gap: 8px; cursor: pointer; color: #bbb; font-size: 14px; }
</style>