import cv2
import numpy as np
import math
from customDataStruct.Rect import Rect

#Rectangle Literals
lowerLeft = 0
upperLeft = 1
upperRight = 2
lowerRight = 3

#detection parameters
REJECTWIDTHFRAC =0.8
REJECTHEIGHTFRAC =0.8
DETECTREGION =190
AREAFACTOR =0.1
TABLEUNIQUECOLOR=5
COLORREDUCER=96
IMGMAXSIZE=1024

#Distance functions
L1Distance = lambda p1p2: abs(p1p2[1]-p1p2[0])
L1DistanceFrac = lambda p1p2: L1Distance(p1p2)/p1p2[0]

def showImage(name,img):
    imgr=cv2.resize(img,(750,1000))#600,800
    cv2.imshow(name,imgr)

def colorSpread(img8, pointSet:np.ndarray):
    #TODO use dynamicNPy instead of list
    points =pointSet.flatten()
    colors = []
    n=len(points)
    for i in range(0,n,2):
        clr=img8[points[i+1]][points[i]][0]
        colors.append(clr)
    return np.std(np.array(colors))

def colorCount(rect:Rect, imgc:np.ndarray)->int:
    origin = rect['upperLeft']
    end = rect['lowerRight']
    imgd = imgc[end[1]:origin[1],origin[0]:end[0]]
    imgd=imgd.flatten()
    u= np.unique(imgd,axis=0)
    return len(u),np.std(imgd)

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



nepaliKeyBoardPreeti={
    'a':"ब", 'b':"द", 'c':"अ", 'd':"म", 'e':"भ", 'f':"ा", 'g':"न", 'h':"ज", 'i':"ष्",
    'j':"व",'k':"प", 'l':"ि", 'm':"फ", 'n':"ल", 'o':"य", 'p':"उ", 'q':"त्र", 'r':"च",
    's':"क", 't':"त", 'u':"ग", 'v':"ख", 'w':"ध", 'x':"ह", 'y':"थ", 'z':"श", 'A':"ब्",
    'B':"ध", 'C':"ऋ", 'D':"म्", 'E':"भ्", 'F':"ँ", 'G':"न्", 'H':"ज्", 'I':"क्ष्", 'J':"व्",
    'K':"प्", 'L':"ी", 'M':"ः", 'N':"ल्", 'O':"इ", 'P':"ए", 'Q':"त्त", 'R':"च्", 'S':"क्",
    'T':"त्", 'U':"ग्", 'V':"ख्", 'W': "ध्", 'X':"ह्", 'Y':"थ्", 'Z':"श्", '0':"ण्", '1':"ज्ञ",
    '2':"द्द", '3':"घ", '4':"द्ध", '5':"छ", '6':"ट", '7':"ठ", '8':"ड", '9':"ढ",
    "~":"ञ्", "`":"ञ", "!":"१", "@":"२", "#":"३", "$":"४", "%":"५", "^":"६", "&":"७",
    "*":"८", "(":"९", ")":"०", "-":"(", "_":")", "+":"ं", "[":"ृ", "{":"र्", "]":"े",
    "}":"ै", "\\":"्", "|":"्र", ";":"स", ":":"स्", "'":"ु", "\"":"ू", ",":",", "<":"?",
    ".":"।", ">":"श्र", "/":"र", "?":"रु", "=":".", "ˆ":"फ्", "Î":"ङ्ख", "å":"द्व", "÷":"/"}



def preeti2Uni(text):
    result=""
    for c in text:
        if c in nepaliKeyBoardPreeti:
            c = nepaliKeyBoardPreeti[c]
        result=result + c
    return result

recognizableCharacters = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'ा', 'ि', 'ी', 'े', 'ै', 'ु', 'ू', 'ृ', 'ं', 'ँ', 'ः', '्', '।',
    'अ', 'आ', 'उ', 'ए', 'इ', 'ऋ', 'ॐ', 
    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 
    'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', '१', '२', '३', '४', '५', '६', '७', '८', '९', '॰', ]

nepaliKeyBoard={
    'a': 'ा', 'i': 'ि', 'I': 'ी', 'e': 'े', 'E': 'ै', 'u': 'ु', 'U': 'ू', 'R': 'ृ', 
    'A': 'अ', 'F': 'आ', 'W': 'इ', 'o': 'उ', 'O': 'ए', 'L': 'ऋ', 'P': 'ॐ', 
    'k': 'क', 'K': 'ख', 'g': 'ग', 'G': 'घ', 'M': 'ङ', 
    'c': 'च', 'C': 'छ', 'j': 'ज', 'J': 'झ', 'Y': 'ञ', 
    'T': 'ट', 'Q': 'ठ', 'D': 'ड', 'V': 'ढ', 'N': 'ण', 
    't': 'त', 'q': 'थ', 'd': 'द', 'v': 'ध', 'n': 'न', 
    'p': 'प', 'f': 'फ', 'b': 'ब', 'B': 'भ', 'm': 'म', 
    'y': 'य', 'r': 'र', 'l': 'ल', 'w': 'व', 'S': 'श', 'H': 'ष', 's': 'स', 'h': 'ह', 
    '1': '१', '2': '२', '3': '३', '4': '४', '5': '५', '6': '६', '7': '७', '8': '८', '9': '९', 
    ',': '्', '.': '।',  'z': 'ः', 'Z': "्र", 'x': 'ं', 'X': 'ँ'}
