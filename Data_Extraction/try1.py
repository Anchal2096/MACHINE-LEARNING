"""import cv2
import numpy as np
from PIL import Image
from pytesseract import image_to_string

# Path of working folder on Disk
src_path = "Dataset/"


def get_string(img_path):
    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite(src_path + "removed_noise.png", img)

    #  Apply threshold to get image with only black and white
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Write the image after apply opencv to do some ...
    cv2.imwrite(src_path + "thres.png", img)

    # Recognize text with tesseract for python
    result = image_to_string(Image.open("/home/atrivedi/Machine-Learning/Data_Extraction/" + src_path +
                                        "thres.png"))

    # Remove template file
    # os.remove(temp)

    return result


print('--- Start recognize text from image ---')
print(get_string("/home/atrivedi/Machine-Learning/Data_Extraction/" + src_path + "img (1).jpeg"))

print("------ Done -------")

"""









import cv2
from PIL import Image
import PIL.Image
import pytesseract
import glob

path = "/home/atrivedi/MachineLearning/DataSets For ML/Receipts/*.*"
for file in glob.glob(path):
    warped = cv2.imread(file)
    imS = cv2.resize(warped, (1350, 1150))
    # cv2.imshow("output", imS)
    cv2.imwrite('Output Image.PNG', imS)
    output = pytesseract.image_to_string(PIL.Image.open('Output Image.PNG').convert("RGB"), lang='eng')
    print(output)

# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
# TESSDATA_PREFIX = 'C:/Program Files (x86)/Tesseract-OCR'



