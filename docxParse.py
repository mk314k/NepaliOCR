from docx import Document

def parseDocx(docFile:str):
    doc = Document(docFile)
    pageNum = 1
    pageText = ''
    allText =[]
    for para in doc.paragraphs:
        paraText = para.text
        if paraText != '':
            pageText = pageText + paraText + '\n'
        if paraText == f'{pageNum}':
            allText.append(pageText)
            pageText =''
            pageNum += 1
    return allText