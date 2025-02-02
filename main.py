from typing import Union
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()


@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    """
    Applies Gaussian blur and edge detection to an image and saves it to a file.

    Args:
        file: The image file to process.

    Returns:
        A FileResponse containing the processed image.
    """
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(grey_image, (7, 7), 0)
    edge = cv2.Canny(blur_image, 100, 200)

    # Save the processed image to a temporary file
    output_path = "processed_image.png"
    cv2.imwrite(output_path, edge)

    return FileResponse(output_path, media_type="image/png")
