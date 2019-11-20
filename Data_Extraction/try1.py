import pytesseract
from PIL import Image

text = pytesseract.image_to_string(Image.open("F:\Pycharm\Machine-Learning\Data_Extraction\Dataset\img1.jpeg"))

print(text)
