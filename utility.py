import cv2
import numpy as np
import math
from Rect import Rect

REJECTWIDTHFRAC =0.8
REJECTHEIGHTFRAC =0.8
DETECTREGION =190
AREAFACTOR =0.1
TABLEUNIQUECOLOR=5
COLORREDUCER=96
IMGMAXSIZE=1024

L1Distance = lambda p1p2: abs(p1p2[1]-p1p2[0])
L1DistanceFrac = lambda p1p2: L1Distance(p1p2)/p1p2[0]

def showImage(name,img):
    imgr=cv2.resize(img,(600,800))
    cv2.imshow(name,imgr)

def cascading(funcs, img,repeat=1):
    imgp=np.array(img)
    while repeat>0:
        for func in funcs:
            imgp=func(imgp)
        repeat-=1
    return imgp

def colorSpread(img8, pointSet:np.ndarray):
    #TODO use dynamicNPy instead of list
    points =pointSet.flatten()
    colors = []
    n=len(points)
    for i in range(0,n,2):
        clr=img8[points[i+1]][points[i]][0]
        colors.append(clr)
    return np.std(np.array(colors))

def boundingObject(contour):

    """AI is creating summary for boundingObject
    Change Me if wordArt or non-rectangular text.
    You may need to create a new class as well

    Args:
        contour ([type]): [description]

    Returns:
        [type]: [description]
    """
    x,y,w,h = cv2.boundingRect(contour)
    return Rect(x,y,width=w,height=h)



unicodeatoz=["ब","द","अ","म","भ","ा","न","ज","ष्","व","प","ि","फ","ल","य","उ","त्र","च","क","त","ग","ख","ध","ह","थ","श"]
unicodeAtoZ=["ब्","ध","ऋ","म्","भ्","ँ","न्","ज्","क्ष्","व्","प्","ी","ः","ल्","इ","ए","त्त","च्","क्","त्","ग्","ख्","ध्","ह्","थ्","श्"]
unicode0to9=["ण्","ज्ञ","द्द","घ","द्ध","छ","ट","ठ","ड","ढ"]
symbolsDict={
    "~":"ञ्",
    "`":"ञ",
    "!":"१",
    "@":"२",
    "#":"३",
    "$":"४",
    "%":"५",
    "^":"६",
    "&":"७",
    "*":"८",
    "(":"९",
    ")":"०",
    "-":"(",
    "_":")",
    "+":"ं",
    "[":"ृ",
    "{":"र्",
    "]":"े",
    "}":"ै",
    "\\":"्",
    "|":"्र",
    ";":"स",
    ":":"स्",
    "'":"ु",
    "\"":"ू",
    ",":",",
    "<":"?",
    ".":"।",
    ">":"श्र",
    "/":"र",
    "?":"रु",
    "=":".",
    "ˆ":"फ्",
    "Î":"ङ्ख",
    "å":"द्व",
    "÷":"/"}

def devunicode(c):
    if c in symbolsDict:
        return c
    a = ord(c)
    if 48<=a<=57:
        return unicode0to9[a-48]
    elif 65<=a<=90:
        return unicodeAtoZ[a-65]
    elif 97<=a<=122:
        return unicodeatoz[a-97]
    else:
        return chr(a)


def preeti2Uni(text):
    result=""
    for c in text:
        result=result+devunicode(c)
    return result

keyBoard ={
    'a':'अ',
    'b':'ब',
    'c':'',
    'd':'',
    'e':'',
    'A':"ा",
}