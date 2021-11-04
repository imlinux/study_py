import fitz
import cv2
import numpy as np


def extract_img_from_pdf(file, start = 0):
    with fitz.open(file) as pdf_file:
        for page_num in range(start, len(pdf_file)):
            page = pdf_file[page_num]
            for img in page.getImageList():
                base_image = pdf_file.extractImage(img[0])
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
                yield img