from docx import Document

def parseDocx(docFile:str)->list[str]:
    """Reads given docx file and parse each paragraphs.
    Assumes each page ends with a page num
    #TODO breaking the assumption, to find better ways to detect page break

    Args:
        docFile (str): docx filename including full file path

    Returns:
        list[str]: each string in the list represent separate page
    """
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