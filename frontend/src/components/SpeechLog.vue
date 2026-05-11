<template>
  <div class="speech-log-panel">
    <div class="log-header">
      <span>实时发言记录</span>
      <button class="btn-clear" @click="clearLogs">清空</button>
    </div>
    
    <div class="log-scroll-area">
      <transition-group name="list">
        <div v-for="log in logs" :key="log.id" class="log-card">
          <div class="log-meta">
            <span class="seat-num" v-if="log.seat_id">{{ log.seat_id }}号</span>
            <span class="player-name">{{ log.name }}</span>
            <span class="log-time">{{ formatTime(log.timestamp) }}</span>
          </div>
          <div class="log-content">{{ log.content }}</div>
        </div>
      </transition-group>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const logs = ref([]);

const handleIncomingSpeech = (data) => {
  if (data.type === 'new') {
    logs.value.unshift({ ...data }); // 最新消息在顶端
    if (logs.value.length > 50) logs.value.pop();
  } else if (data.type === 'update' && logs.value.length > 0) {
    logs.value[0].content = data.content;
  }
};

const clearLogs = () => logs.value = [];
const formatTime = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

defineExpose({ handleIncomingSpeech });
</script>

<style scoped>
.speech-log-panel { 
  width: 350px; 
  height: 720px; /* 匹配视频高度 */
  display: flex; 
  flex-direction: column; 
  background: #181818; 
  border-left: 1px solid #333; 
}
.log-header { 
  padding: 15px; 
  background: #222; 
  display: flex; 
  justify-content: space-between; 
  align-items: center; 
  border-bottom: 1px solid #333;
}
.btn-clear { background: #333; color: #888; border: none; padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; }
.btn-clear:hover { background: #444; color: #fff; }

.log-scroll-area { flex: 1; overflow-y: auto; padding: 12px; }
.log-card { 
  margin-bottom: 12px; 
  padding: 12px; 
  background: #252525; 
  border-radius: 8px; 
  border-left: 4px solid #42b883;
  box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.log-meta { font-size: 12px; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; color: #aaa; }
.seat-num { background: #42b883; color: #fff; padding: 1px 5px; border-radius: 3px; font-weight: bold; }
.player-name { color: #fff; font-weight: 600; }
.log-time { color: #555; margin-left: auto; }
.log-content { 
  font-size: 13px; 
  color: #ccc; 
  line-height: 1.6; 
  white-space: pre-wrap; /* 允许空格换行并保留多个空格 */
  word-break: break-all; 
}

/* 动画 */
.list-enter-active, .list-leave-active { transition: all 0.3s ease; }
.list-enter-from { opacity: 0; transform: translateY(-10px); }
</style>