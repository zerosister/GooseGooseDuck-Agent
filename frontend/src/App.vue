<script setup lang="ts">
import { ref } from 'vue'
import ScreenCapture from './components/ScreenCapture.vue'
import SpeechLog from './components/SpeechLog.vue'

// 使用 InstanceType 获取组件的公共实例类型
const speechLogRef = ref<InstanceType<typeof SpeechLog> | null>(null)

const onSpeechMessage = (data: any) => {
  // 此时 TypeScript 知道 speechLogRef 包含 handleIncomingSpeech 方法
  if (speechLogRef.value) {
    speechLogRef.value.handleIncomingSpeech(data)
  }
}
</script>

<template>
  <div class="app-container">
    <header class="main-header">
      <div class="logo">GGD <span>Vision Admin</span></div>
      <p class="desc">5x3 阵位自适应感知控制台</p>
    </header>

    <main class="content-area">
      <div class="main-layout">
        <ScreenCapture @speech-data="onSpeechMessage" />
        <SpeechLog ref="speechLogRef" />
      </div>
    </main>

    <footer class="footer">
      <div class="tip">💡 提示：系统将自动通过 HSV 检测边框，并利用 ASR 整合相同发言人的内容。</div>
    </footer>
  </div>
</template>

<style>
:root {
  --bg-color: #0a0a0a;
  --accent-color: #42b883;
}

body {
  margin: 0;
  background-color: var(--bg-color);
  color: #fff;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.app-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  min-height: 100vh;
}

.main-header { text-align: center; margin-bottom: 20px; }
.logo { font-size: 2.4rem; font-weight: 800; color: var(--accent-color); letter-spacing: -1.5px; }
.logo span { color: #fff; font-weight: 200; }

/* 核心布局：左右分栏 */
.main-layout {
  display: flex;
  flex-direction: row;
  align-items: flex-start;
  gap: 0; /* 紧贴布局 */
  background: #121212;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid #333;
}

.content-area {
  box-shadow: 0 30px 60px rgba(0,0,0,0.6);
}

.footer { margin-top: 20px; color: #666; font-size: 0.9rem; }
.tip { background: #1a1a1a; padding: 10px 20px; border-radius: 8px; border-left: 4px solid var(--accent-color); }
</style>