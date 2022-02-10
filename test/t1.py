from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(use_angle_cls=False, lang="ch", use_gpu=False,
                det_model_dir="/home/dong/test",
                det_limit_type="min",
                det_limit_side_len=736,
                rec_model_dir="/home/dong/model/rec_chinese_lite_sigmai_no_v2.0_infer",
                rec_char_dict_path="/home/dong/dev/PaddleOCR/ppocr/utils/no.txt")

img = cv2.imread("/home/dong/tmp/score2/0.jpg")
result = ocr.ocr(img)

for r in result:
    [p1, p2, p3, p4] = r[0]
    print(r[1:])