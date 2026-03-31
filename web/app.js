const logEl = document.getElementById("log");
const previewEl = document.getElementById("preview");
const btnPick = document.getElementById("btnPick");
const btnConnect = document.getElementById("btnConnect");
const btnStart = document.getElementById("btnStart");
const btnAnalyze = document.getElementById("btnAnalyze");
const btnStop = document.getElementById("btnStop");

let stream = null;
let ws = null;
let sessionId = "";
let analysisBuffer = "";
let frameTimer = null;
let audioRecorders = [];
let micStream = null;

const WS_URL = "ws://127.0.0.1:8000/ws/analyze";
const screenshotIntervalMs = 2000;

function log(message, obj = null) {
  const line = obj ? `${message} ${JSON.stringify(obj)}` : message;
  logEl.textContent = `${new Date().toISOString()} ${line}\n${logEl.textContent}`;
}

function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i += 1) binary += String.fromCharCode(bytes[i]);
  return window.btoa(binary);
}

function pickSupportedAudioMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
  ];
  for (const mimeType of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(mimeType)) {
      return mimeType;
    }
  }
  return "";
}

async function pickWindow() {
  stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: true
  });
  previewEl.srcObject = stream;
  log("已选择屏幕/窗口");
}

function connectWs() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => log("WS 已连接");
  ws.onclose = () => log("WS 已关闭");
  ws.onerror = (e) => log("WS 错误", { error: String(e) });
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "session") {
      sessionId = msg.session_id;
      log("收到会话ID", { sessionId });
      return;
    }

    if (msg.type === "analysis_chunk") {
      // 流式文本片段（如果后端仍走 chunk 模式）
      const content = typeof msg.content === "string" ? msg.content : String(msg.content ?? "");
      analysisBuffer += content;
      log("分析中...", { delta: content });
      return;
    }

    if (msg.type === "analysis_done") {
      // 非流式最终输出，或 chunk 模式的收尾
      const full = typeof msg.full_result === "string" ? msg.full_result : (msg.full_result ?? analysisBuffer);
      analysisBuffer = "";
      try {
        const parsed = typeof full === "string" ? JSON.parse(full) : full;
        log("分析完成（JSON）", parsed);
      } catch (e) {
        log("分析完成（非JSON原文）", { raw: full });
      }
      return;
    }

    if (msg.type === "analysis_result") {
      // README 协议：一次性返回结构化 result
      log("分析完成（result）", msg.result ?? msg);
      return;
    }

    log("收到服务端消息", msg);
  };
}

function sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !stream) return;
  const video = previewEl;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/png");
  const imageBase64 = dataUrl.split(",")[1];
  ws.send(JSON.stringify({ type: "frame", image_base64: imageBase64 }));
}

function startAudioSend() {
  if (!stream || !ws || ws.readyState !== WebSocket.OPEN) return;
  const startRecorder = (sourceStream, sourceName) => {
    const mimeType = pickSupportedAudioMimeType();
    const options = mimeType ? { mimeType } : {};
    const recorder = new MediaRecorder(sourceStream, options);
    recorder.ondataavailable = async (e) => {
      if (!e.data || !e.data.size) return;
      const buffer = await e.data.arrayBuffer();
      const audioBase64 = arrayBufferToBase64(buffer);
      ws.send(JSON.stringify({
        type: "audio",
        audio_base64: audioBase64,
        mime_type: e.data.type || mimeType || "audio/webm",
        source: sourceName
      }));
      log("已发送音频块", { source: sourceName, size: e.data.size, mime_type: e.data.type || mimeType || "audio/webm" });
    };
    recorder.start(4000);
    audioRecorders.push(recorder);
    log("音频采集已启动", { source: sourceName, mime_type: mimeType || "browser-default" });
  };

  // 1) 同步采集窗口/系统音频（若浏览器和权限允许）
  const displayAudioTracks = stream.getAudioTracks();
  if (displayAudioTracks.length) {
    const track = displayAudioTracks[0];
    log("检测到屏幕音频轨道", { label: track.label || "unknown" });
    const audioStream = new MediaStream([track]);
    startRecorder(audioStream, "display");
  } else {
    log("未检测到屏幕音频轨道（可能是浏览器/权限限制）");
  }

  // 2) 同步采集麦克风音频（与窗口音频并行）
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then((micInputStream) => {
      micStream = micInputStream;
      startRecorder(micStream, "microphone");
    })
    .catch((err) => {
      log("麦克风采集启动失败", { error: String(err) });
    });
}

function startCapture() {
  if (!stream || !ws || ws.readyState !== WebSocket.OPEN) {
    log("请先选择窗口并连接 WS");
    return;
  }
  frameTimer = setInterval(sendFrame, screenshotIntervalMs);
  log("采集已开始（音频由后端采集）");
}

function analyzeNow() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "analyze" }));
  log("已触发分析", { sessionId });
}

function stopAll() {
  if (frameTimer) clearInterval(frameTimer);
  frameTimer = null;
  audioRecorders.forEach((recorder) => {
    if (recorder && recorder.state !== "inactive") recorder.stop();
  });
  audioRecorders = [];
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
    previewEl.srcObject = null;
  }
  log("已停止采集");
}

btnPick.onclick = async () => {
  try { await pickWindow(); } catch (e) { log("选择窗口失败", { error: String(e) }); }
};
btnConnect.onclick = connectWs;
btnStart.onclick = startCapture;
btnAnalyze.onclick = analyzeNow;
btnStop.onclick = stopAll;
