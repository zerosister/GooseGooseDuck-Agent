from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Body
from backend.utils.logger import log
from backend.app.core.ws_manager import ws_manager # 直接导入单例
import json

router = APIRouter()

@router.get("/status")
async def get_service_status():
    return {"status": "online", "agent": "GGD_Perception_Agent"}

@router.websocket("/ws/stream")
async def vision_websocket(websocket: WebSocket):
    """
    视觉流双向通信：
    1. 通过单例 ws_manager 统一管理连接
    2. 接收视频帧并实时返回 HSV 检测结果 (用于前端点亮网格)
    """
    await ws_manager.connect(websocket) # 使用全局单例
    vision_service = websocket.app.state.vision_service

    try:
        while True:
            # 1. 接收视频帧字节流
            data = await websocket.receive_bytes()
            
            # 2. 视觉处理 (同步执行，确保响应速度)
            result = vision_service.process_frame(data)
            
            # 3. 立即返回视觉反馈
            # 添加 type 字段区分“视觉状态”和“语音文本”
            await websocket.send_json({
                "type": "processed_frame",
                "active_seat": result.get("active_seat"),
                "speaker_name": result.get("speaker_name")
            })
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        log.error(f"WebSocket 异常: {e}")
        ws_manager.disconnect(websocket)

@router.post("/calibrate")
async def save_calibration(request: Request, config: dict = Body(...)):
    """保存前端传来的标定坐标并触发服务重载"""
    try:
        config_path = "roi_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        
        # 触发视觉服务重载并获取返回的推导结果
        updated_data = request.app.state.vision_service.update_config()
        
        log.success("标定数据已更新并保存。")
        return {
            "status": "success", 
            "config_preview": updated_data.get("config_preview") if updated_data else None
        }
    except Exception as e:
        log.error(f"保存标定失败: {e}")
        return {"status": "error", "message": str(e)}

@router.get("/calibration/current")
async def get_current_calibration(request: Request):
    """获取当前生效的 ROI 配置预览"""
    try:
        vision_service = request.app.state.vision_service
        current_data = vision_service.update_config()
        return {
            "status": "success",
            "config_preview": current_data.get("config_preview") if current_data else None
        }
    except Exception as e:
        log.error(f"获取当前配置失败: {e}")
        return {"status": "error", "message": str(e)}