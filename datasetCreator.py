import numpy as np
from detector import detectBylines,doubleContourDetect,detect
from utility import showImage, preeti2Uni, nepaliKeyBoard
import cv2
from customDataStruct.pdfClass import PDFReader
from tkinter import Checkbutton,Tk,Text,Button,END,filedialog,BooleanVar,Radiobutton,IntVar,INSERT
from docxParse import parseDocx

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
    detectedRects = doubleContourDetect(img)
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
    if docText.get():
        allTexts = parseDocx(filename[:-3]+'docx')
    else:
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
    if nepaliTyping.get() and keyP in nepaliKeyBoard:
        keyP = nepaliKeyBoard[keyP]
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
    #TODO tweak the parameters for different pc may require little experimentation 
    winH, winW = 800,1000
    ws = Tk()
    ws.title('DataSet Collector')
    ws.geometry(f'{winW}x{winH}')
    ws.config(bg='#000000')
    loadPdf = Button(
        ws,
        text='New Pdf',
        padx=20,
        pady=10,
        bg='#000088',
        fg='blue',
        command=get_new_pdf
    )
    loadPrg = Button(
        ws,
        text='Load Progress',
        padx=20,
        pady=10,
        bg='cyan',
        fg='blue',
        command=loadProgress
    )
    savePage = Button(
        ws,
        text='Save Page',
        padx=20,
        pady=10,
        bg='#ff00ff',
        fg='blue',
        command=submit_text
    )
    prev = Button(
        ws,
        text='Prev',
        padx=20,
        pady=10,
        bg='#ff00ff',
        fg='blue',
        command=prevClick
    )
    nextButton = Button(
        ws,
        text='Next',
        padx=20,
        pady=10,
        bg='#ff00ff',
        fg='blue',
        command=nextClick
    )
    labeler = Button(
        ws,
        text='Label',
        padx=20,
        pady=10,
        bg='#ff00ff',
        fg='blue',
        command=handleLabel
    )

    loadPdf.place(x=0, y=0)             #Top Left placing
    loadPrg.place(x=winW//10, y=0)      #right next to previous button
    savePage.place(x=4*winW//5, y=19*winH//20) #bottom Right
    prev.place(x=0, y=9*winH//20)
    nextButton.place(x=9*winW//10, y=9*winH//20)
    labeler.place(x=0, y=7*winH//8)

    docText = BooleanVar()
    checkbox = Checkbutton (ws, text = "Separate Doc for text", variable = docText, onvalue = True, offvalue = False)
    checkbox.place(x=0, y=winH//12)

    pageNum = IntVar()

    textbox = Text(
        ws,
        height=2*winH//35,
        width=winW//9,
        wrap='word',
        bg='#ffffff'
    )
    textbox.place(x=winW//10, y=winH//8)
    textbox.bind("<Key>",keyPressed)
    textbox.bind("<Control-v>",pasteText)
    textbox.bind("<BackSpace>",lambda _ :textbox.delete(INSERT-1))
    textbox.bind("<Return>",lambda _ :textbox.insert(INSERT-1,'\n'))

    convertText = BooleanVar()
    checkbox = Checkbutton (ws, text = "Need Unicode Conversion", variable = convertText, onvalue = True, offvalue = False)
    checkbox.place(x=7*winW//10, y=winH//40)

    nepaliTyping = BooleanVar()
    checkbox = Checkbutton (ws, text = "Nepali Typing", variable = nepaliTyping, onvalue = True, offvalue = False)
    checkbox.place(x=7*winW//10, y=winH//20)

    saveAsImg = BooleanVar()
    R1 = Radiobutton(ws, text="Save as rects", variable=saveAsImg, value=False)
    R1.place(x=winW//2, y=9*winH//10)
    R2 = Radiobutton(ws, text="Save as Images", variable=saveAsImg, value=True)
    R2.place(x=6*winW//10,y=9*winH//10)

    ws.mainloop()

    
    


