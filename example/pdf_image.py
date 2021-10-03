import cv2
import fitz
import numpy as np

pdf_file = fitz.open("/home/dong/tmp/zuowen/JUYE_F_00007.pdf")

for page in pdf_file:

    for img in page.getImageList():
        base_image = pdf_file.extractImage(img[0])
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        print(decoded.shape)
        # cv2.imshow("", decoded)
        # cv2.waitKey(0)

