from date_extractor import extract_dates
import datetime
import pytesseract
from PIL import Image
import datefinder
import numpy as np
import datetime
from datetime import date
import re

dtype = np.int64
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(Image.open("F:\Pycharm\Machine-Learning\Data_Extraction\Dataset\img1.jpeg"))
# print(text)
# text = text.replace(" ", "")
dates = extract_dates(text)
# x = datetime.datetime(dates)
print(dates)
for i in dates:
    print('the extracted date is', i.strftime("%d %b %Y"))
    break

# print(text)
# st = text.split("\n") #  list containg the text of the image
