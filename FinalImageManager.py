import math
from PIL import Image
import numpy as np
from collections import deque

class FinalImageManager:
    width = None
    height = None
    bitDepth = None

    img = None
    data = None
    original = None

    def read(self, fileName):
        global img 
        global data 
        global original 
        global width 
        global height 
        global bitDepth 
        img = Image.open(fileName)
        data = np.array(img)
        original = np.copy(data)
        width = data.shape[1]
        height = data.shape[0]
        mode_to_bpp = {"1":1,"L":8,"P":8,"RGB":24,"RGBA":32,"CMYK":32,"YCbCr":24,"LAB":24,"HSV":24,"I":32,"F":32}
        bitDepth = mode_to_bpp[img.mode]

        print("Image %s with %s x %s pixels (%s bits per pixels) has been read!" % (img.filename, width, height, bitDepth))
    
    def getData(self):
        global data
        return data
    
    def write(self, fileName):
        global img 
        img = Image.fromarray(data)
        try:
            img.save(fileName)
        except:
            print("Write file error")
        else:
            print("Image %s has been written!" %(fileName))

    def convertToGray(self):
        global data
        gray = np.dot(data[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        data = np.stack((gray, gray, gray), axis=-1)

    def restoreToOriginal(self):
        global data
        width = original.shape[0]
        height = original.shape[1]
        data = np.zeros([width, height, 3])
        data = np.copy(original)

    def resizeNearestNeighbour(self, scaleX, scaleY):
        global data
        global width
        global height
        newWidth = (int)(round(width * scaleX))
        newHeight = (int)(round(height * scaleY))

        data_temp = np.zeros([width, height, 3])
        data_temp = data.copy()

        data = np.resize(data, [newWidth, newHeight, 3])

        for y in range(newHeight):
            for x in range(newWidth):
                xNearest = (int)(round(x / scaleX))
                yNearest = (int)(round(y / scaleY))
                xNearest = width - 1 if xNearest >= width else xNearest
                xNearest = 0 if xNearest < 0 else xNearest
                yNearest = height - 1 if yNearest >= height else yNearest
                yNearest = 0 if yNearest < 0 else yNearest
                data[x, y, :] = data_temp[xNearest, yNearest, :]

    def resizeBilinear(self, scaleX, scaleY):
        global width
        global height
        global data
        
        # คำนวณขนาดใหม่
        newWidth = int(round(width * scaleX))
        newHeight = int(round(height * scaleY))

        # สร้างสำเนาของข้อมูลต้นฉบับ
        data_temp = data.copy()

        # สร้างอาร์เรย์ใหม่สำหรับภาพที่ถูกปรับขนาด
        resized_data = np.zeros((newHeight, newWidth, 3), dtype=data.dtype)

        for y in range(newHeight):
            for x in range(newWidth):
                # คำนวณตำแหน่งในภาพต้นฉบับ
                oldX = x / scaleX
                oldY = y / scaleY

                # หาพิกัด 4 จุดรอบๆ จุดเก่า
                x1 = int(math.floor(oldX))
                y1 = int(math.floor(oldY))
                x2 = min(x1 + 1, width - 1)
                y2 = min(y1 + 1, height - 1)

                # ดึงค่าสีจากพิกัด 4 จุด
                color11 = data_temp[y1, x1, :]
                color12 = data_temp[y2, x1, :]
                color21 = data_temp[y1, x2, :]
                color22 = data_temp[y2, x2, :]

                # คำนวณค่าการอินเตอร์โพเลชัน
                dx = oldX - x1
                dy = oldY - y1

                # อินเตอร์โพเลชันในแนวนอน
                P1 = (1 - dx) * color11 + dx * color21
                P2 = (1 - dx) * color12 + dx * color22

                # อินเตอร์โพเลชันในแนวตั้ง
                P = (1 - dy) * P1 + dy * P2

                # ปัดเศษและแปลงเป็นชนิดข้อมูลเดิม
                P = np.round(P).astype(data.dtype)

                # กำหนดค่าให้กับ resized_data
                resized_data[y, x, :] = P

        # อัปเดตข้อมูลภาพและขนาด
        data = resized_data
        height, width = data.shape[:2]

    def erosion(self, se):
        self.convertToGray()
        
        # Zero-padding [height + 2*se.height, width + 2*se.width, 3]
        pad_y, pad_x = se.origin
        padded_data = np.zeros((height + 2 * pad_y, width + 2 * pad_x, 3), dtype=data.dtype)
        padded_data[pad_y:pad_y + height, pad_x:pad_x + width, :] = data

        eroded_data = np.zeros_like(data)

        for y in range(pad_y, pad_y + height):
            for x in range(pad_x, pad_x + width):
                # Extract the region of interest
                subData = padded_data[y - pad_y:y + se.height - pad_y, x - pad_x:x + se.width - pad_x, 0]

                # Apply ignore elements if any
                for point in se.ignoreElements:
                    subData[int(point[0]), int(point[1])] = se.elements[int(point[0]), int(point[1])]

                # Check if the structuring element fits
                if np.array_equal(subData, se.elements):
                    # Set to minimum value (assuming binary image: 0)
                    eroded_data[y - pad_y, x - pad_x] = 0
                else:
                    # Otherwise, keep the pixel as is (or set to background)
                    eroded_data[y - pad_y, x - pad_x] = 255

        # Update the image data
        data[:, :, 0] = eroded_data
        data[:, :, 1] = eroded_data
        data[:, :, 2] = eroded_data

    def dilation(self, se):
        self.convertToGray()
        
        # Zero-padding [height + 2*se.height, width + 2*se.width, 3]
        pad_y, pad_x = se.origin
        padded_data = np.zeros((height + 2 * pad_y, width + 2 * pad_x, 3), dtype=data.dtype)
        padded_data[pad_y:pad_y + height, pad_x:pad_x + width, :] = data

        dilated_data = np.zeros_like(data)

        for y in range(pad_y, pad_y + height):
            for x in range(pad_x, pad_x + width):
                # Extract the region of interest
                subData = padded_data[y - pad_y:y + se.height - pad_y, x - pad_x:x + se.width - pad_x, 0]

                # Apply ignore elements if any
                for point in se.ignoreElements:
                    subData[int(point[0]), int(point[1])] = se.elements[int(point[0]), int(point[1])]

                # Check if any element in the structuring element matches
                if np.any(subData & se.elements):
                    # Set to maximum value (assuming binary image: 255)
                    dilated_data[y - pad_y, x - pad_x] = 255
                else:
                    # Otherwise, keep the pixel as is (or set to background)
                    dilated_data[y - pad_y, x - pad_x] = 0

        # Update the image data
        data[:, :, 0] = dilated_data
        data[:, :, 1] = dilated_data
        data[:, :, 2] = dilated_data

    def removeSaltPepperNoise(self, se):
        self.erosion(se)
        
        self.dilation(se)

        self.dilation(se)

        self.erosion(se)

    def linearSpatialFilter(self, kernel, size):
        global data

        if (size % 2 ==0):
            print("Size Invalid: must be odd number!")
            return
        
        data_zeropaded = np.zeros([height + int(size/2) * 2, width + int(size/2) * 2,  3])
        data_zeropaded[int(size/2):height + int(size/2), int(size/2):width + int(size/2), :] = data
        for y in range(int(size/2), int(size/2) + height):
            for x in range(int(size/2), int(size/2) + width):
                subData = data_zeropaded[y - int(size/2):y + int(size/2) + 1, x - int(size/2):x + int(size/2) + 1, :]

                sumRed = np.sum(np.multiply(subData[:,:,0:1].flatten(), kernel))
                sumGreen = np.sum(np.multiply(subData[:,:,1:2].flatten(), kernel))
                sumBlue = np.sum(np.multiply(subData[:,:,2:3].flatten(), kernel))
                
                sumRed = 255 if sumRed > 255 else sumRed
                sumRed = 0 if sumRed < 0 else sumRed
                
                sumGreen = 255 if sumGreen > 255 else sumGreen
                sumGreen = 0 if sumGreen < 0 else sumGreen
                
                sumBlue = 255 if sumBlue > 255 else sumBlue
                sumBlue = 0 if sumBlue < 0 else sumBlue
                
                data[y - int(size/2), x - int(size/2), 0] = sumRed
                data[y - int(size/2), x - int(size/2), 1] = sumGreen
                data[y - int(size/2), x - int(size/2), 2] = sumBlue   

    # def cannyEdgeDetector(self, lower, upper):
    #     global data
    #     #Step 1 - Apply 5 x 5 Gaussian filter
    #     gaussian = [2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0,
    #     4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
    #     5.0 / 159.0, 12.0 / 159.0, 15.0 / 159.0, 12.0 / 159.0, 5.0 / 159.0,
    #     4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
    #     2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0]
        
    #     self.linearSpatialFilter(gaussian, 5)
    #     self.convertToGray()

    #     #Step 2 - Find intensity gradient
    #     sobelX = [ 1, 0, -1,
    #                 2, 0, -2,
    #                 1, 0, -1]
    #     sobelY = [ 1, 2, 1,
    #                 0, 0, 0,
    #                 -1, -2, -1]

    #     magnitude = np.zeros([width, height])
    #     direction = np.zeros([width, height])
        
    #     data_zeropaded = np.zeros([height + 2, width + 2, 3])
    #     data_zeropaded[1:height + 1, 1:width + 1, :] = data
        
    #     for y in range(1, height + 1):
    #         for x in range(1, width + 1):
    #             gx = 0
    #             gy = 0
                
    #             subData = data_zeropaded[x - 1:x + 2, y - 1:y + 2, :]
                
    #             gx = np.sum(np.multiply(subData[:,:,0:1].flatten(), sobelX))
    #             gy = np.sum(np.multiply(subData[:,:,0:1].flatten(), sobelY))
                
    #             magnitude[x - 1, y - 1] = math.sqrt(gx * gx + gy * gy)
    #             direction[x - 1, y - 1] = math.atan2(gy, gx) * 180 / math.pi

    #     #Step 3 - Nonmaxima Suppression
    #     gn = np.zeros([width, height])

    #     for y in range(1, height + 1):
    #         for x in range(1, width + 1):
    #             targetX = 0
    #             targetY = 0

    #             #find closest direction
    #             if (direction[x - 1, y - 1] <= -157.5):
    #                 targetX = 1
    #                 targetY = 0
    #             elif (direction[x - 1, y - 1] <= -112.5):
    #                 targetX = 1
    #                 targetY = -1
    #             elif (direction[x - 1, y - 1] <= -67.5):
    #                 targetX = 0
    #                 targetY = 1
    #             elif (direction[x - 1, y - 1] <= -22.5):
    #                 targetX = 1
    #                 targetY = 1
    #             elif (direction[x - 1, y - 1] <= 22.5):
    #                 targetX = 1
    #                 targetY = 0
    #             elif (direction[x - 1, y - 1] <= 67.5):
    #                 targetX = 1
    #                 targetY = -1
    #             elif (direction[x - 1, y - 1] <= 112.5):
    #                 targetX = 0
    #                 targetY = 1
    #             elif (direction[x - 1, y - 1] <= 157.5):
    #                 targetX = 1
    #                 targetY = 1
    #             else:
    #                 targetX = 1
    #                 targetY = 0

    #             if (y + targetY >= 0 and y + targetY < height and x + targetX >= 0 and x + targetX < width and magnitude[x - 1, y - 1] < magnitude[x + targetY - 1, y + targetX - 1]):
    #                 gn[x - 1, y - 1] = 0
    #             elif (y - targetY >= 0 and y - targetY < height and x - targetX >= 0 and x - targetX < width and magnitude[x - 1, y - 1] < magnitude[x - targetY - 1, y - targetX - 1]):
    #                 gn[x - 1, y - 1] = 0
    #             else:
    #                 gn[x - 1, y - 1] = magnitude[x - 1, y - 1]
                
    #             #set back first
    #             gn[x - 1, y - 1] = 255 if gn[x - 1, y - 1] > 255 else gn[x - 1, y - 1]
    #             gn[x - 1, y - 1] = 0 if gn[x - 1, y - 1] < 0 else gn[x - 1, y - 1]
                
    #             data[x - 1, y - 1, 0] = gn[x - 1, y - 1]
    #             data[x - 1, y - 1, 1] = gn[x - 1, y - 1]
    #             data[x - 1, y - 1, 2] = gn[x - 1, y - 1]


    #     #upper threshold checking with recursive
    #     for y in range(height):
    #         for x in range(width):
    #             if (data[x, y, 0] >= upper):
    #                 data[x, y, 0] = 255
    #                 data[x, y, 1] = 255
    #                 data[x, y, 2] = 255

    #                 self.hystConnect(x, y, lower)

    #     #clear unwanted values
    #     for y in range(height):
    #         for x in range(width):
    #             if (data[x, y, 0] != 255):
    #                 data[x, y, 0] = 0
    #                 data[x, y, 1] = 0
    #                 data[x, y, 2] = 0

    def cannyEdgeDetector(self, lower, upper):
        # Step 1 - Apply 5 x 5 Gaussian filter
        gaussian = [
            2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0,
            4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
            5.0 / 159.0, 12.0 / 159.0, 15.0 / 159.0, 12.0 / 159.0, 5.0 / 159.0,
            4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
            2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0
        ]
        
        self.linearSpatialFilter(gaussian, 5)
        self.convertToGray()

        # Step 2 - Find intensity gradient using Sobel operators
        sobelX = [
            1, 0, -1,
            2, 0, -2,
            1, 0, -1
        ]
        sobelY = [
            1, 2, 1,
            0, 0, 0,
            -1, -2, -1
        ]

        # Initialize magnitude and direction arrays correctly [height, width]
        magnitude = np.zeros((height, width))
        direction = np.zeros((height, width))
        
        # Zero-padding: [height + 2, width + 2, 3]
        data_padded = np.zeros((height + 2, width + 2, 3), dtype=data.dtype)
        data_padded[1:height + 1, 1:width + 1, :] = data

        for y in range(1, height + 1):
            for x in range(1, width + 1):
                gx = 0
                gy = 0
                
                # Correct indexing: [y-1:y+2, x-1:x+2, :]
                subData = data_padded[y - 1:y + 2, x - 1:x + 2, 0]  # Grayscale channel
                
                # Apply Sobel kernels
                gx = np.sum(np.multiply(subData.flatten(), sobelX))
                gy = np.sum(np.multiply(subData.flatten(), sobelY))
                
                # Calculate gradient magnitude and direction
                magnitude[y - 1, x - 1] = math.sqrt(gx * gx + gy * gy)
                direction[y - 1, x - 1] = math.atan2(gy, gx) * 180 / math.pi

        # Step 3 - Nonmaxima Suppression
        gn = np.zeros((height, width))

        for y in range(1, height + 1):
            for x in range(1, width + 1):
                targetX = 0
                targetY = 0

                angle = direction[y - 1, x - 1]

                # Normalize angle to [0, 180)
                if angle < 0:
                    angle += 180

                # Determine the direction
                if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                    targetX, targetY = 1, 0
                elif (22.5 <= angle < 67.5):
                    targetX, targetY = 1, -1
                elif (67.5 <= angle < 112.5):
                    targetX, targetY = 0, -1
                elif (112.5 <= angle < 157.5):
                    targetX, targetY = 1, 1
                else:
                    targetX, targetY = 1, 0  # Default case

                current_mag = magnitude[y - 1, x - 1]

                # Neighboring pixels based on direction
                pos1_x = x - 1 + targetX
                pos1_y = y - 1 + targetY
                pos2_x = x - 1 - targetX
                pos2_y = y - 1 - targetY

                # Check bounds before accessing magnitude
                if (0 <= pos1_x < width and 0 <= pos1_y < height and
                    0 <= pos2_x < width and 0 <= pos2_y < height):
                    if current_mag < magnitude[pos1_y, pos1_x] or current_mag < magnitude[pos2_y, pos2_x]:
                        gn[y - 1, x - 1] = 0
                    else:
                        gn[y - 1, x - 1] = current_mag
                else:
                    gn[y - 1, x - 1] = current_mag

                # Clamp to [0, 255]
                gn[y - 1, x - 1] = min(max(gn[y - 1, x - 1], 0), 255)
                
                # Set the grayscale value back to the image
                val = gn[y - 1, x - 1]
                data[y - 1, x - 1, 0] = val
                data[y - 1, x - 1, 1] = val
                data[y - 1, x - 1, 2] = val

        # Step 4 - Double Threshold and Hysteresis
        for y in range(height):
            for x in range(width):
                if data[y, x, 0] >= upper:
                    data[y, x] = [255, 255, 255]
                    self.hystConnect(x, y, lower)

        # Step 5 - Clear unwanted values
        for y in range(height):
            for x in range(width):
                if data[y, x, 0] != 255:
                    data[y, x] = [0, 0, 0]


    def hystConnect(self, x, y, threshold):
        global data

        for i in range(y - 1, y + 2):
            for j in range(x - 1, x + 2):
                if ((j < width) and (i < height) and
                    (j >= 0) and (i >= 0) and
                    (j != x) and (i != y)):

                    value = data[j, i, 0]
                    if (value != 255):
                        if (value >= threshold):
                            data[j, i, 0] = 255
                            data[j, i, 1] = 255
                            data[j, i, 2] = 255

                            self.hystConnect(j, i, threshold)
                        else:
                            data[j, i, 0] = 0
                            data[j, i, 1] = 0
                            data[j, i, 2] = 0

    def houghTransform(self, percent):
        global data

        #The image should be converted to edge map first
        
        #Work out how the hough space is quantized
        numOfTheta = 720
        thetaStep = math.pi / numOfTheta
        
        highestR = int(round(max(width, height) * math.sqrt(2)))
        
        centreX = int(width / 2)
        centreY = int(height / 2)
        
        print("Hough array w: %s height: %s" % (numOfTheta, (2*highestR)))
        
        #Create the hough array and initialize to zero
        houghArray = np.zeros([numOfTheta, 2*highestR])
        
        #Step 1 - find each edge pixel
        #Find edge points and vote in array
        for y in range(3, height - 3):
            for x in range(3, width - 3):
                pointColor = data[x, y, 0]
                if (pointColor != 0):
                    #Edge pixel found
                    for i in range(numOfTheta):
                        #Step 2 - Apply the line equation and update hough array
                        #Work out the r values for each theta step
                        
                        r = int((x - centreX) * math.cos(i * thetaStep) + (y - centreY) * math.sin(i * thetaStep))

                    #Move all values into positive range for display purposes
                        r = r + highestR
                        if (r < 0 or r >= 2 * highestR):
                            continue
                        
                        #Increment hough array
                        houghArray[i, r] = houghArray[i, r] + 1
        #Step 3 - Apply threshold to hough array to find line
        #Find the max hough value for the thresholding operation
        maxHough = np.amax(houghArray)

        #Set the threshold limit
        threshold = percent * maxHough

        #Step 4 - Draw lines
        # Search for local peaks above threshold to draw
        for i in range(numOfTheta):
            for j in range(2 * highestR):
                #only consider points above threshold
                if (houghArray[i, j] >= threshold):
                    # see if local maxima
                    draw = True
                    peak = houghArray[i, j]
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            #not seeing itself
                            if (k == 0 and l == 0):
                                continue
                            testTheta = i + k
                            testOffset = j + l
                            
                            if (testOffset < 0 or testOffset >= 2*highestR):
                                continue
                            if (testTheta < 0):
                                testTheta = testTheta + numOfTheta
                            if (testTheta >= numOfTheta):
                                testTheta = testTheta - numOfTheta
                            if (houghArray[testTheta][testOffset] > peak):
                                #found bigger point
                                draw = False
                                break

                    #point found is not local maxima
                    if (not(draw)):
                        continue
                    #if local maxima, draw red back
                    tsin = math.sin(i*thetaStep)
                    tcos = math.cos(i*thetaStep)
                    
                    if (i <= numOfTheta / 4 or i >= (3 * numOfTheta) / 4):
                        for y in range(height):
                            #vertical line
                            x = int((((j - highestR) - ((y - centreY) * tsin)) / tcos) + centreX)

                            if(x < width and x >= 0):
                                data[x, y, 0] = 255
                                data[x, y, 1] = 0
                                data[x, y, 2] = 0

                    else:
                        for x in range(width):
                            #horizontal line
                            y = int((((j - highestR) - ((x - centreX) * tcos)) / tsin) + centreY)

                            if(y < height and y >= 0):
                                data[x, y, 0] = 255
                                data[x, y, 1] = 0
                                data[x, y, 2] = 0

    def calculateHomography(self, srcPoints, dstPoints):
        A = np.zeros((8, 8))
        b = np.zeros(8)

        for i in range(4):
            xSrc, ySrc = srcPoints[i]
            xDst, yDst = dstPoints[i]

            A[2 * i] = [xSrc, ySrc, 1, 0, 0, 0, -xSrc * xDst, -ySrc * xDst]
            A[2 * i + 1] = [0, 0, 0, xSrc, ySrc, 1, -xSrc * yDst, -ySrc * yDst]
            b[2 * i] = xDst
            b[2 * i + 1] = yDst

        # Solve using Gaussian elimination
        homography = self.gaussianElimination(A, b)

        # ตรวจสอบ Homography ด้วยการแมปจุดต้นทางไปยังจุดปลายทาง
        for i in range(4):
            src = srcPoints[i]
            dst = dstPoints[i]
            mapped = self.applyHomographyToPoint(homography, src[0], src[1])
            print(f"Point {src} mapped to {mapped}, Expected: {dst}")

        return homography

    def gaussianElimination(self, A, b):
        n = len(b)
        for i in range(n):
            # Pivoting
            maxIndex = np.argmax(np.abs(A[i:, i])) + i
            A[[i, maxIndex]] = A[[maxIndex, i]]
            b[i], b[maxIndex] = b[maxIndex], b[i]
            
            # Normalize the row
            for k in range(i + 1, n):
                factor = A[k, i] / A[i, i]
                b[k] -= factor * b[i]
                A[k, i:] -= factor * A[i, i:]
            
        # Back substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
            
        # The last element of the homography matrix (h33) is 1
        homography = np.zeros(9)
        homography[:8] = x
        homography[8] = 1

        return homography
    
    def invertHomography(self, H):
        # คำนวณดีเทอร์มิแนนท์ของเมทริกซ์ 3x3
        det = (H[0] * (H[4] * H[8] - H[5] * H[7])
            - H[1] * (H[3] * H[8] - H[5] * H[6])
            + H[2] * (H[3] * H[7] - H[4] * H[6]))

        if det == 0:
            raise ValueError("Matrix is not invertible")

        invDet = 1.0 / det

        # คำนวณอินเวอร์สของ Homography
        invH = np.zeros(9)
        invH[0] = invDet * (H[4] * H[8] - H[5] * H[7])
        invH[1] = invDet * (H[2] * H[7] - H[1] * H[8])
        invH[2] = invDet * (H[1] * H[5] - H[2] * H[4])

        invH[3] = invDet * (H[5] * H[6] - H[3] * H[8])
        invH[4] = invDet * (H[0] * H[8] - H[2] * H[6])
        invH[5] = invDet * (H[2] * H[3] - H[0] * H[5])

        invH[6] = invDet * (H[3] * H[7] - H[4] * H[6])
        invH[7] = invDet * (H[1] * H[6] - H[0] * H[7])
        invH[8] = invDet * (H[0] * H[4] - H[1] * H[3])

        # ตรวจสอบว่า H * invH = identity matrix
        H_matrix = H.reshape((3, 3))
        invH_matrix = invH.reshape((3, 3))
        identity = np.dot(H_matrix, invH_matrix)
        print("H * invH =\n", identity)

        return invH

    # def applyHomography(self, H):
    #     global data, height, width

    #     data_temp = np.copy(data)

    #     invH = self.invertHomography(H)

    #     for y in range(height):
    #         for x in range(width):
    #             # Apply the inverse of the homography to find the corresponding source pixel
    #             sourcePoint = self.applyHomographyToPoint(invH, x, y)
                
    #             srcX = int(round(sourcePoint[0]))
    #             srcY = int(round(sourcePoint[1]))
                
    #             # Debug: แสดงพิกัดต้นทางและปลายทาง
    #             if (y == 0 and x == 0) or (y == height-1 and x == width-1):
    #                 print(f"Destination Point: ({x}, {y}) -> Source Point: ({srcX}, {srcY})")
                
    #             # Check if the calculated source coordinates are within the source image bounds
    #             if 0 <= srcX < width and 0 <= srcY < height:
    #                 # Copy the pixel from the source image to the destination image
    #                 data_temp[y, x] = data[srcY, srcX]
    #             else:
    #                 # If out of bounds, set the destination pixel to a default color
    #                 data_temp[y, x] = [0, 0, 0]

    #     # Copy the processed image back to the original image
    #     data = np.copy(data_temp)

    def applyHomographyToPoint(self, H, x, y):
        # Homogeneous coordinates calculation after transformation
        xh = H[0] * x + H[1] * y + H[2]
        yh = H[3] * x + H[4] * y + H[5]
        w = H[6] * x + H[7] * y + H[8]

        # Normalize by w to get the Cartesian coordinates in the destination image
        xPrime = xh / w
        yPrime = yh / w

        return np.array([xPrime, yPrime])

    def bilinear_interpolate(self, image, x, y):
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, image.shape[1] - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, image.shape[0] - 1)

        Ia = image[y0, x0]
        Ib = image[y1, x0]
        Ic = image[y0, x1]
        Id = image[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    def applyHomography(self, H):
        global data, height, width

        data_temp = np.zeros([height, width, 3])
        data_temp = np.copy(data)

        invH = self.invertHomography(H)

        for y in range(height):
            for x in range(width):
                # Apply the inverse of the homography to find the corresponding source pixel
                sourcePoint = self.applyHomographyToPoint(invH, x, y)

                srcX = sourcePoint[0]
                srcY = sourcePoint[1]

                # Check if the calculated source coordinates are within the source image bounds
                if 0 <= srcX < width and 0 <= srcY < height:
                    # Use bilinear interpolation instead of rounding
                    data_temp[y, x] = self.bilinear_interpolate(data, srcX, srcY)
                else:
                    # If out of bounds, set the destination pixel to a default color
                    data_temp[y, x] = [0, 0, 0]

        # Copy the processed image back to the original image
        data = np.copy(data_temp)

    def medianFilter(self, size):
        global data

        if size % 2 == 0:
            print("Size Invalid: must be odd number!")
            return

        k = int(size // 2)
        data_zeropaded = np.zeros([height + k * 2, width + k * 2, 3])
        data_zeropaded[k:height + k, k:width + k, :] = data

        for y in range(k, k + height):
            for x in range(k, k + width):
                subData = data_zeropaded[y - k:y + k + 1, x - k:x + k + 1, :]

                medianRed = np.median(subData[:, :, 0])
                medianGreen = np.median(subData[:, :, 1])
                medianBlue = np.median(subData[:, :, 2])

                data[y - k, x - k, 0] = medianRed
                data[y - k, x - k, 1] = medianGreen
                data[y - k, x - k, 2] = medianBlue

    def find_connected_components(self):
        global width
        global height
        global data
        visited = np.zeros_like(data, dtype=bool)
        components = []

        for y in range(height):
            for x in range(width):
                if data[y, x] == 1 and not visited[y, x]:
                    # เริ่มต้น BFS
                    queue = [(y, x)]
                    visited[y, x] = True
                    component = []

                    while queue:
                        cy, cx = queue.pop(0)
                        component.append((cy, cx))

                        # ตรวจสอบ 8-connected neighbors
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    if binary_image[ny, nx] == 1 and not visited[ny, nx]:
                                        queue.append((ny, nx))
                                        visited[ny, nx] = True
                    components.append(component)
        return components

    def get_bounding_box(component):
        ys = [p[0] for p in component]
        xs = [p[1] for p in component]
        return (min(ys), min(xs), max(ys), max(xs))

    def segment_characters(self):
        global data
        components = self.find_connected_components(data)
        bounding_boxes = [self.get_bounding_box(comp) for comp in components]
        # อาจจะต้องเรียงลำดับ bounding boxes จากซ้ายไปขวา
        bounding_boxes.sort(key=lambda box: box[1])
        return bounding_boxes
