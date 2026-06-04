import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from internal.utils.logging_utils import setup_logging
from internal.constants.infer_name import VIDEO_ACTION
from internal.services.fight.video_fight_tracker import VideoFightTracker
from internal.utils.config_loader import load_config
from internal.core.predictor_wrapper import PredictorWrapper

video_path = "/home/jeremyjfn/app/skyure-df/PaddleDetection/storage/private/videos/vehicle_plate_0400.mpeg"
output_dir = "/home/jeremyjfn/app/skyure-df/PaddleDetection/storage/private/eval/results/"

# Load configuration
config = load_config("configs/config.yml")
setup_logging()

predictor_wrapper = PredictorWrapper(
    device=config.system.device,
)

fight_tracker = VideoFightTracker()
predictor = predictor_wrapper.predict_video(
    cfg_path=config.system.config_path,
    video_file=video_path,
    output_dir=output_dir,
    trackers={VIDEO_ACTION: fight_tracker},
)
predictor.run(video_path, thread_idx=0)

fight_tracker.save_all_predictions(json_output_path=os.path.join(output_dir, f"predictions_{predictor.file_name}.json"))
