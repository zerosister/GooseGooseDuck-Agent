import yaml
from path_tool import get_abs_path

def load_audio_config():
    with open(get_abs_path("config/input_capture.yaml"), "r") as f:
        return yaml.load(f)["audio"]

audio_config = load_audio_config()