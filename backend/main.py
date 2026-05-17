import sys
import os
# 将当前执行脚本的祖父目录（项目根目录）加入搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.my_router import router as my_router
from backend.app.services.audio_service.audio_capture import AudioCaptureService
from backend.app.services.video_frame_capture import VideoFrameCaptureService
from backend.app.services.vision_ocr import GGDVisionService
from backend.app.core.ggd_coordinator import GGDCoordinator
from backend.utils.logger import log
from backend.utils.config_loader import config

# 1. 定义生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 【启动阶段】 ---
    log.success("=== GGD Agent Starting Up ===")
    
    # 初始化视觉分析服务
    vision_service = GGDVisionService()
    app.state.vision_service = vision_service

    # 初始化视频帧捕获服务
    frame_capture_service = VideoFrameCaptureService(
        config.vision.mode, 
        config.vision.target, 
        vision_service=vision_service,
        fps_limit=config.vision.fps_limit)
    app.state.frame_capture_service = frame_capture_service
    frame_capture_service.start()

    # 启动协调器
    coordinator = GGDCoordinator(vision_service)
    vision_service.coordinator = coordinator

    # 初始化音频服务
    audio_service = AudioCaptureService(coordinator)
    app.state.audio_service = audio_service # 将实例挂载到 app 状态中，方便后续调用
    
    # 在子线程中启动音频服务 (避免阻塞异步主线程)
    audio_thread = threading.Thread(
        target=audio_service.run, 
        name="AudioCaptureThread",
        daemon=True
    )
    audio_thread.start()
    log.info("AudioCaptureService started in a background thread.")
    
    yield  # --- 【运行阶段】 ---
    
    # --- 【关闭阶段】 ---
    await coordinator.stop() # 停止协调器
    vision_service.stop() # 停止视觉服务
    frame_capture_service.stop() # 停止视频帧捕获服务
    log.warning("=== GGD Agent Shutting Down ===")
    log.info("Stopping AudioCaptureService...")
    
    try:
        audio_service.stop()
        # 等待线程安全退出（可选）
        audio_thread.join(timeout=2)
        log.success("AudioCaptureService stopped cleanly.")
    except Exception as e:
        log.error(f"Error while stopping audio service: {e}")

# 2. 初始化应用
app = FastAPI(
    title="GooseGooseDuck Agent Multi-Modal API",
    lifespan=lifespan
)

# 3. 跨域中间件
app.add_middleware(
    CORSMiddleware,   
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. 注册视觉分析路由
app.include_router(my_router, prefix="/api/v1", tags=["vision"])

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "audio_thread_alive": threading.enumerate()
    }

if __name__ == "__main__":
    # 配置 Uvicorn 运行参数
    # 使用字符串 "main:app" 以支持更好的热重载和多进程模式
    log.info("Launching Integrated GGD Agent Server...")
    
    uvicorn.run(
        "main:app", 
        host=config.server.host, 
        port=config.server.port, 
        loop="auto", 
        http="httptools",
        workers=1 # 处理音频流这种有状态的服务时，建议先使用单 worker 避免资源抢占
    )