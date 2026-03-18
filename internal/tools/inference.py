import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from internal.services.video_fight_tracker import VideoFightTracker
from internal.utils.config_loader import load_config
from internal.core.fight_detector import FightDetector

video_path = "/home/jeremyjfn/app/fighting-detection/PaddleDetection/storage/private/videos/fight_0400.mpeg"
output_dir = "/home/jeremyjfn/app/fighting-detection/PaddleDetection/storage/private/results/"

# Load configuration
config = load_config("configs/config.yml")

fight_detector = FightDetector(
    cfg_path=config["paddle_detection"]["config_path"],
    device=config["paddle_detection"]["device"],
)

fight_tracker = VideoFightTracker()
predictor = fight_detector.predict_video(
    video_file=video_path,
    output_dir=output_dir,
    fight_tracker=fight_tracker,
)
predictor.run(video_path, thread_idx=0)

fight_tracker.save_all_predictions(json_output_path=os.path.join(output_dir, f"predictions_{predictor.file_name}.json"))
