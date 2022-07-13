from PyPDF2 import PdfFileReader
from tkinter import Tk,Text,Button,END,LEFT
from tkinter import filedialog
from preeti2uni import unicode0to9,unicodeAtoZ,unicodeatoz,symbolsDict

ws = Tk()
ws.title('TextExtractor')
ws.geometry('800x600')
ws.config(bg='#D9653B')

def choose_pdf():
      filename = filedialog.askopenfilename(
            initialdir = "/",  
            title = "Select a File",
            filetypes = (("PDF files","*.pdf*"),("all files","*.*")))
      if filename:
          return filename

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

def convertToDevanagari(text):
    result=""
    for c in text:
        result=result+devunicode(c)
    return result


def read_pdf():
    filename = choose_pdf()
    reader = PdfFileReader(filename)
    pageObj = reader.getNumPages()
    for page_count in range(pageObj):
        page = reader.getPage(page_count)
        page_data = page.extractText()
        page_data = convertToDevanagari(page_data)
        textbox.insert(END, f'{page_data}'.encode('utf-16'))

def copy_pdf_text():
    content = textbox.get(1.0, "end-1c")
    ws.withdraw()
    ws.clipboard_clear()
    ws.clipboard_append(content)
    ws.update()
    ws.destroy()


textbox = Text(
    ws,
    height=30,
    width=80,
    wrap='word',
    bg='#D9BDAD'
)
textbox.pack(expand=True)

Button(
    ws,
    text='Choose Pdf File',
    padx=20,
    pady=10,
    bg='#262626',
    fg='white',
    command=read_pdf
).pack(expand=True, side=LEFT, pady=10)

Button(
    ws,
    text="Copy Text",
    padx=20,
    pady=10,
    bg='#262626',
    fg='white',
    command=copy_pdf_text
).pack(expand=True, side=LEFT, pady=10)


ws.mainloop()