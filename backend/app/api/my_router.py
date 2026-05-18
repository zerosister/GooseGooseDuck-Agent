from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Body
from backend.utils.logger import log
from backend.app.core.ws_manager import ws_manager # 直接导入单例
from backend.utils.config_loader import get_public_config, save_public_config
import json
import cv2
import base64

router = APIRouter()

@router.get("/status")
async def get_service_status():
    return {"status": "online", "agent": "GGD_Perception_Agent"}

@router.get("/config")
async def get_config():
    return {
        "status": "success",
        "config": get_public_config()
    }

def _apply_runtime_config(request: Request, saved_config: dict):
    vision_config = saved_config.get("vision", {})
    frame_capture_service = getattr(request.app.state, "frame_capture_service", None)
    if frame_capture_service:
        frame_capture_service.update_config(
            vision_config.get("mode"),
            vision_config.get("target"),
            vision_config.get("fps_limit"),
        )

    game_setting = saved_config.get("game_setting", {})
    vision_service = getattr(request.app.state, "vision_service", None)
    if vision_service and "seat_num" in game_setting:
        vision_service.set_seat_num(game_setting["seat_num"])

@router.put("/config")
async def update_config(request: Request, config: dict = Body(...)):
    try:
        current_config = get_public_config()
        saved_config = save_public_config(config)
        _apply_runtime_config(request, saved_config)
        restart_required = (
            current_config.get("server", {}).get("host") != saved_config.get("server", {}).get("host")
            or current_config.get("server", {}).get("port") != saved_config.get("server", {}).get("port")
        )
        return {
            "status": "success",
            "config": saved_config,
            "restart_required": restart_required,
            "message": "配置已保存，运行时配置已更新。" if not restart_required else "配置已保存，采集和游戏设置已热更新；端口变更需要重启应用后生效。"
        }
    except Exception as e:
        log.error(f"保存应用配置失败: {e}")
        return {"status": "error", "message": str(e)}

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
