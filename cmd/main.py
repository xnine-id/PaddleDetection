import logging
import sys
import os
import argparse
import signal
import threading
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from internal.utils.logging_utils import setup_logging
from internal.utils.config_loader import load_config
from internal.manager.camera_manager import CameraManager

logger = logging.getLogger("Main")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Multi-Camera Manager for Fighting Detection")
    parser.add_argument("--config", default="configs/config.yml", help="Path to configuration file")
    args = parser.parse_args()
    
    config_path = args.config
    logger.info(f"Using config: {config_path}")

    config = load_config(config_path)

    manager = CameraManager(config)
    manager.start()

    stop_event = threading.Event()

    def handle_signal(signum, frame):
        if not stop_event.is_set():
            stop_event.set()
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            try:
                manager.stop()
            finally:
                # Ensure immediate exit to avoid repeated signals causing core dump
                os._exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        # Block main thread to keep process alive until a signal arrives
        stop_event.wait()
    except KeyboardInterrupt:
        handle_signal(signal.SIGINT, None)

if __name__ == "__main__":
    main()
