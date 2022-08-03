import numpy as np
from detector import detectBylines,doubleContourDetect
from utility import showImage, preeti2Uni, keyBoard
import cv2
from pdfClass import PDFReader
from tkinter import Checkbutton,Tk,Text,Button,END,filedialog,BooleanVar,Radiobutton,IntVar,INSERT

globalData ={}

def showAllDetectedRegion(imgo:np.ndarray, imageTitle:str, showLabel=False, source=None):
    """AI is creating summary for showAllDetectedRegion

    Args:
        imgo (np.ndarray): [description]
        imageTitle (str): [description]
        showLabel (bool, optional): [description]. Defaults to False.
        source ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    img = np.array(imgo)
    label = 1
    result = {}
    detectedRects = doubleContourDetect(img)[0]
    for rect in detectedRects:
        cv2.rectangle(img,rect["lowerLeft"], rect["upperRight"], (0,0,255), 2)
        if source:
            if saveAsImg.get():
                pt1 = rect['lowerLeft']
                pt2 = rect['upperRight']
                imgs = imgo[pt1[0]:pt2[0], pt1[1]:pt2[1]]
                cv2.imwrite(source[:-4]+f'label{label}.jpg',imgs)
            else:
                result[label] = rect
        if showLabel:
            cv2.putText(img,f'{label}', rect['lowerLeft'],color=(255,0,0),thickness=3,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2)
        label += 1
    showImage(imageTitle, img)
    return result

def main():
    """AI is creating summary for main
    """
    pageIndex = pageNum.get()
    pageRange = globalData['pageRange']
    if pageIndex<pageRange[0]:
        pageIndex=pageRange[0]
        pageNum.set(pageRange[0])
    if pageIndex>pageRange[1]:
        pageIndex=pageRange[1]
        pageNum.set(pageRange[1])

    imgPath = globalData['fileName'][:-4]+f'Img/img{pageIndex}.jpg'
    imgo = cv2.imread(imgPath)

    result = showAllDetectedRegion(imgo,"Without Labels", source=imgPath)
    _ = showAllDetectedRegion(imgo,"With Labels", True)

    with open (imgPath[:-4]+'.txt','r') as file:
        text = file.read()
    if convertText.get():
        text = preeti2Uni(text)
    textbox.delete("1.0", "end-1c")
    textbox.insert(END, text)

    if not saveAsImg.get():
        with open (imgPath[:-4]+'.rct','w') as rectFile:
            rectFile.write(result.__str__())

def fileSelector(ext = ("PDF files","*.pdf")):
    """AI is creating summary for fileSelector

    Args:
        ext (tuple, optional): [description]. Defaults to ("PDF files","*.pdf").

    Returns:
        [type]: [description]
    """
    filename = filedialog.askopenfilename(
            initialdir = "/",  
            title = "Select a File",
            filetypes = (ext,("all files","*.*")))
    #filename='/Users/kartikeshmishra/Kartikesh/NepaliOCR/NepaliOCR/trainImages/train.prg'
    if filename:
        globalData['fileName'] = filename
        return filename

def get_new_pdf():
    """AI is creating summary for get_new_pdf
    """
    filename = fileSelector()
    pdf = PDFReader(filename)
    imgNames = pdf.toImage()
    globalData['imgNames'] = imgNames
    allTexts = pdf.extractText()
    if convertText.get():
        allTexts =[preeti2Uni(text) for text in allTexts]
    with open (filename[:-4]+'.prg', 'w') as file:
        file.write(f'1,{pdf.totalPages()}')
    for pageNum, text in enumerate(allTexts):
        with open (filename[:-4]+f'Img/img{pageNum+1}.txt','w') as file:
            file.write(text)
    globalData['pageRange'] =(1,pdf.totalPages())
    main()

def loadProgress():
    """AI is creating summary for loadProgress
    """
    filename = fileSelector(("Progress files","*.prg"))
    with open (filename, 'r') as file:
        fileContent = file.read()
    imageRange = fileContent.split(',')
    imageRange = [int(val) for val in imageRange]
    globalData['pageRange'] =imageRange
    main()


def submit_text():
    """AI is creating summary for submit_text
    """
    txtPath = globalData['fileName'][:-4]+f'Img/img{pageNum.get()}.txt'
    with open (txtPath, 'w') as file:
        file.write(textbox.get("1.0","end-1c"))
    pass

def prevClick():
    """AI is creating summary for prevClick
    """
    pageNum.set(pageNum.get()-1)
    main()

def nextClick():
    """AI is creating summary for nextClick
    """
    pageNum.set(pageNum.get()+1)
    main()

def keyPressed(event):
    """AI is creating summary for keyPressed

    Args:
        event ([type]): [description]

    Returns:
        [type]: [description]
    """
    keyP = event.char
    if nepaliTyping.get() and keyP in keyBoard:
        keyP = keyBoard[keyP]
    event.widget.insert("insert", keyP)
    return "break"

def pasteText(event):
    cliptext = ws.clipboard_get()
    if cliptext:
        textbox.insert(INSERT,cliptext)

def handleLabel():
    text = textbox.get('1.0','end-1c')
    textL = text.split('\n')
    result =''
    i=0
    for txt in textL:
        txtL = txt.split(' ')
        for t in txtL:
            if f'{i}:' in t and t.index(f'{i}:') ==0:
                result = result + t+'\n'
            else:
                result = result + f'{i}:{t}\n'
            i=i+1
    textbox.delete('1.0','end-1c')
    textbox.insert(END,result)

if __name__ == '__main__':
    winH, winW = 700,900
    ws = Tk()
    ws.title('DataSet Collector')
    ws.geometry(f'{winW}x{winH}')
    ws.config(bg='#000000')
    Button(
        ws,
        text='New Pdf',
        padx=20,
        pady=10,
        bg='#000088',
        fg='blue',
        command=get_new_pdf
    ).place(x=0, y=0)

    Button(
        ws,
        text='Load Progress',
        padx=20,
        pady=10,
        bg='cyan',
        fg='blue',
        command=loadProgress
    ).place(x=100, y=0)

    Button(
        ws,
        text='Save Page',
        padx=20,
        pady=10,
        bg='#ff00ff',
        fg='blue',
        command=submit_text
    ).place(x=winW-200, y=winH-50)
    pageNum = IntVar()

    Button(
        ws,
        text='Prev',
        padx=20,
        pady=10,
        bg='#ff00ff',
        fg='blue',
        command=prevClick
    ).place(x=0, y=winH//2-30)

    Button(
        ws,
        text='Next',
        padx=20,
        pady=10,
        bg='#ff00ff',
        fg='blue',
        command=nextClick
    ).place(x=winW-100, y=winH//2-30)

    Button(
        ws,
        text='Label',
        padx=20,
        pady=10,
        bg='#ff00ff',
        fg='blue',
        command=handleLabel
    ).place(x=0, y=winH-100)

    textbox = Text(
        ws,
        height=2*winH//35,
        width=winW//9,
        wrap='word',
        bg='#ffffff'
    )
    textbox.place(x=100, y=100)
    textbox.bind("<Key>",keyPressed)
    textbox.bind("<Control-v>",pasteText)
    textbox.bind("<BackSpace>",lambda _ :textbox.delete(INSERT-1))
    textbox.bind("<Return>",lambda _ :textbox.insert(INSERT-1,'\n'))

    convertText = BooleanVar()
    checkbox = Checkbutton (ws, text = "Need Unicode Conversion", variable = convertText, onvalue = True, offvalue = False)
    checkbox.place(x=winW-300, y=20)

    nepaliTyping = BooleanVar()
    checkbox = Checkbutton (ws, text = "Nepali Typing", variable = nepaliTyping, onvalue = True, offvalue = False)
    checkbox.place(x=winW-300, y=45)

    saveAsImg = BooleanVar()
    R1 = Radiobutton(ws, text="Save as rects", variable=saveAsImg, value=False)
    R1.place(x=winW//2, y=winH-70)
    R2 = Radiobutton(ws, text="Save as Images", variable=saveAsImg, value=True)
    R2.place(x=winW//2+100,y=winH-70)

    ws.mainloop()

    
    


