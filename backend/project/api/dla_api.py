import cv2
import numpy as np
from flask import Blueprint, jsonify, request

from project.predictor import make_predictions, prepare_predictor

dla_blueprint = Blueprint("", __name__)

PREDICTOR = None
if not PREDICTOR:
    PREDICTOR = prepare_predictor()

@dla_blueprint.route("/analyse-image-json", methods=["POST"])
def analyse_image_json():
    f = request.files["file"]

    img_str = f.read()

    nparr = np.frombuffer(img_str, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)  # BGR format
    print(f"Image shape: {image.shape}")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    global PREDICTOR

    json_data = make_predictions(image, True,PREDICTOR )
    del nparr
    del image
    del img_str
    return jsonify(json_data)
