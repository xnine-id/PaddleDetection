import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PaddleDetection/deploy")))

from PaddleDetection.deploy.pipeline.pipe_utils import get_test_images
from internal.services.trackers.images.images_vehicle_plate_tracker import ImagesVehiclePlateTracker
from internal.constants.infer_name import VEHICLE_PLATE
from internal.utils.logging_utils import setup_logging
from internal.utils.config_loader import load_config
from internal.core.predictor_wrapper import PredictorWrapper

image_dir = "/home/jeremyjfn/app/skyure-df/PaddleDetection/storage/private/eval/frames/vehicle_plate/images"
output_dir = (
    "/home/jeremyjfn/app/skyure-df/PaddleDetection/storage/private/eval/results/vehicle_plate"
)

# Load configuration
config = load_config("configs/config.yml")
setup_logging()

predictor_wrapper = PredictorWrapper(
    cfg_path=config.system.config_path,
    device=config.system.device,
)

vehicleplate_tracker = ImagesVehiclePlateTracker()
predictor = predictor_wrapper.predict_images(
    image_dir=image_dir,
    output_dir=output_dir+"/imgs",
    trackers={VEHICLE_PLATE: vehicleplate_tracker},
)

input = get_test_images(image_dir, None)
predictor.run(input, thread_idx=0)

vehicleplate_tracker.save_all_predictions(
    json_output_path=os.path.join(output_dir, f"predictions_vehicle_plate.json")
)
