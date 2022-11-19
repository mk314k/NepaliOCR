from docx import Document


def parse_docx(doc_file: str) -> list[str]:
    """Read paragraphs from a .docx file and split pages at numeric page markers."""
    doc = Document(doc_file)
    page_num = 1
    page_text = ""
    all_text = []
    for para in doc.paragraphs:
        para_text = para.text
        if para_text != "":
            page_text = page_text + para_text + "\n"
        if para_text == f"{page_num}":
            all_text.append(page_text)
            page_text = ""
            page_num += 1
    return all_text


parseDocx = parse_docx
