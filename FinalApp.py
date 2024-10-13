from FinalImageManager import FinalImageManager
from StructuringElement import StructuringElement
import numpy as np
im = FinalImageManager()

# หาขอบกระดาษ
# im.read("FinalDIP67.bmp")
# im.cannyEdgeDetector(100, 180)
# im.houghTransform(0.5)
# im.dot(6,45,135) # top left
# im.dot(6,550,65) # top right
# im.dot(6,69,516) # bottom left
# im.dot(6,755,320) # bottom right
# im.write("test.bmp")

# แปลงเป็น gray scale
# im.read("FinalDIP67.bmp")
# im.convertToGray()

# กำหนดจุดเป้าหมาย-ปลายทาง
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

# ปรับความเอียง
# im.applyHomography(im.calculateHomography(srcPoints, dstPoints))
# im.write("test1.bmp")

# หาเส้นขอบและตีกริต
# x = "x"
# y = "y"
# im.read("test1.bmp")

# im.drawLine(y,271)
# im.drawLine(y,281)
# im.drawLine(y,281)
# im.drawLine(y,291)
# im.drawLine(y,301)
# im.drawLine(y,311)
# im.drawLine(y,321)
# im.drawLine(y,331)
# im.drawLine(y,341)
# im.drawLine(y,351)
# im.drawLine(y,361) # range = 90

# im.drawLine(x,141)
# im.drawLine(x,150)
# im.drawLine(x,159)
# im.drawLine(x,168)
# im.drawLine(x,177)
# im.drawLine(x,186)
# im.drawLine(x,195)
# im.drawLine(x,204) 
# # 28
# im.drawLine(x,232)
# im.drawLine(x,241)
# im.drawLine(x,250)
# im.drawLine(x,259)
# im.drawLine(x,268)
# im.drawLine(x,277)
# im.drawLine(x,286)
# im.drawLine(x,295) # range = 63
# # 28
# im.drawLine(x,323)
# im.drawLine(x,332)
# im.drawLine(x,341)
# im.drawLine(x,350)
# im.drawLine(x,359)
# im.drawLine(x,368)
# im.drawLine(x,377)
# im.drawLine(x,386) 
# # 28
# im.drawLine(x,414)
# im.drawLine(x,423)
# im.drawLine(x,432)
# im.drawLine(x,441)
# im.drawLine(x,450)
# im.drawLine(x,459)
# im.drawLine(x,468)
# im.drawLine(x,477) 
# # 28
# im.drawLine(x,505)
# im.drawLine(x,514)
# im.drawLine(x,523)
# im.drawLine(x,532)
# im.drawLine(x,541)
# im.drawLine(x,550)
# im.drawLine(x,559)
# im.drawLine(x,568) 
# # 28
# im.drawLine(x,596)
# im.drawLine(x,605)
# im.drawLine(x,614)
# im.drawLine(x,623)
# im.drawLine(x,632)
# im.drawLine(x,641)
# im.drawLine(x,650)
# im.drawLine(x,659) 

# im.write("test2.bmp")

# ได้ pattern คือ1ช่องมีขนาด กว้าง 9px สูง 10px spacebar 28px

# หาเกณค่าสีขาว-ดำ
# im.read("test1.bmp")
# data = im.getData()
# print("black= " , im.avgColor((650, 351),(659, 361)))   # 142.8
# print("white= " , im.avgColor((159, 271),(168, 281)))   # 180.8 avg=162

# บันทึกสี
im.read("test1.bmp")
characters = np.zeros((6, 9, 7), dtype=int)

x = 141
for k in range(6):  # มีเลข6ตัว
    for j in range(7):  # width
        y = 271
        for i in range(9):  # height
            
            avgColor = im.avgColor((x, y),(x + 9, y + 10))
            if avgColor[0] > 162:
                # ช่องนี้เป็นสีชาว ใส่เลข 0 ใน array
                characters[k, i, j] = 0
            else:
                # ช่องนี้เป็นสีดำ ใส่เลข 1 ใน array
                characters[k, i, j] = 1
            y += 10 # ขยับไปช่องต่อไปในแกน y

        x += 9  # ขยับไปช่องต่อไปในแกน x

    x += 28     # ขยับไปตัวเลขตัวต่อไป

# print(characters)   # เช็คผลลัพท์

# เปรียบเทียบสิ่งที่อ่านได้กับ templates
templates = im.getTamplate()
text = ""
for char in characters:
    char = im.match_character(char, templates)
    if char:
        text += char

print("Decoded E-13B:", text)