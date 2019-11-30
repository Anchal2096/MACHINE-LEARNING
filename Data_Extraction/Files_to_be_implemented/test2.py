import re
import datetime
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
text = pytesseract.image_to_string(Image.open("F:\Pycharm\Machine-Learning\Data_Extraction\Dataset\img (2).jpeg"))


filepath = 'new 1.txt'
datepattern = '%d-%m-%Y'

x = re.findall('\d\d-\d\d-\d\d\d\d', text)
y = []
for item in x:
    try:
        date = datetime.datetime.strptime(item, datepattern)
        y.append(date)
    except:
        pass
res = []
for item in y:
    a, b = str(item).split(' ')
    res.append(a)

for item in res: print(item)