from recognizer import recognizer
import cv2
import numpy as np
from fpdf import FPDF
from PIL import Image

pdf = FPDF(orientation='P',unit='mm', format='A4')
data = recognizer("014.jpg")

pdf.add_page(format=(data['height'],data['width']))
img = cv2.imread(data['file'])
print(img.shape)
print(img[1].shape)
print(img[1][1].shape)
pdf.set_draw_color(r=255, g=0, b=0)
pdf.set_fill_color(10)
#pdf.rect(0,0,10,10,'FD')

for rect in data['rects']:
    x1=int(rect.x)
    x2=int(rect.x+rect.width)
    y1=int(rect.y)
    y2=int(rect.y+rect.height)
    #print(x1,x2,y1,y2)
    imgk=img[y1:y2,x1:x2]
    #print(imgk)
    imgr = Image.fromarray(imgk)
    pdf.rect(rect.x,rect.y,rect.width,rect.height,'FD')
    #print(rect)
    pdf.image(imgr,x=rect.x,y=rect.y,w=rect.width,h=rect.height)
    

pdf.output("pdftest14.pdf")


