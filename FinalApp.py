from FinalImageManager import FinalImageManager
from StructuringElement import StructuringElement
import numpy as np
im = FinalImageManager()

# im.read("FinalDIP67.bmp")
im.read("test1.bmp")

# หาขอบกระดาษ
# im.cannyEdgeDetector(100, 180)
# im.houghTransform(0.5)
# im.dot(6,45,135) # top left
# im.dot(6,550,65) # top right
# im.dot(6,69,516) # bottom left
# im.dot(6,755,320) # bottom right
# im.write("test.bmp")

#แปลงเป็น gray scale
# im.convertToGray()



# # กำหนดจุดเป้าหมาย-ปลายทาง
# srcPoints = np.array([
# [45,135], # top-left
# [550,65], # top-right
# [755,320], # bottom-right
# [69,516] # bottom-left
# ])

# dstPoints = np.array([
# [0, 0], # top-left
# [800, 0], # top-right
# [800, 600], # bottom-right
# [0, 600] # bottom-left
# ])

# # # ปรับความเอียง
# im.applyHomography(im.calculateHomography(srcPoints, dstPoints))



templates = {
    '0': np.array([
        [0,1,1,1,1,1,0],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [0,1,1,1,1,1,0],
    ]),
    '1': np.array([
        
    ]),
    
    }

recognized_text = ""

for char in characters:
    recognized_char = im.match_character(char, templates)
    if recognized_char:
        recognized_text += recognized_char

print("MICR Data:", recognized_text)