from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Body
from backend.utils.logger import log
from backend.app.core.ws_manager import ws_manager # 直接导入单例
import json
import cv2
import base64

router = APIRouter()

@router.get("/status")
async def get_service_status():
    return {"status": "online", "agent": "GGD_Perception_Agent"}

@router.websocket("/ws/stream")
async def vision_websocket(websocket: WebSocket):
    """
    视觉推流 WebSocket 接口
    1. 触发发送视频帧
    2. 定时发送说话人信息
    """
    await ws_manager.connect(websocket) # 使用全局单例

    try:
        while True:
            # 1. 接收视频帧字节流
            data = await websocket.receive_json()

            if data.get("type") == "request_capture_frame":
                # 前端请求当前视频帧用于标定展示
                capture_service = websocket.app.state.frame_capture_service
                frame = capture_service.get_latest_frame()
                if frame is not None:
                    # 编码为 base64 发回给前端
                    _, buffer = cv2.imencode('.jpg', frame)
                    base64_str = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send_json({
                        "type": "calibration_frame",
                        "image": f"data:image/jpeg;base64,{base64_str}"
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