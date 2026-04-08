OLD_PATH="/home/jeremyjfn/app/fighting-detection/PaddleDetection"
NEW_PATH="/home/jeremyjfn/app/skyure-df/PaddleDetection"

find venv/bin -type f -exec sed -i "1s|$OLD_PATH|$NEW_PATH|" {} +