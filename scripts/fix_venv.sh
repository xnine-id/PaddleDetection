OLD_PATH="/home/ams/app/skyure/PaddleDetection"
NEW_PATH="/home/ams/app/x9/skyure/PaddleDetection"

find venv/bin -type f -exec sed -i "1s|$OLD_PATH|$NEW_PATH|" {} +