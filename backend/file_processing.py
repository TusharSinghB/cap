from PyPDF2 import PdfReader
import csv

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_text_from_txt(txt_file):
    return txt_file.read().decode('utf-8')

def is_csv_content(text):
    try:
        sample = text.strip().split('\n')[:5]
        dialect = csv.Sniffer().sniff('\n'.join(sample))
        reader = csv.reader(sample, dialect)
        rows = list(reader)
        col_counts = [len(row) for row in rows]
        return len(set(col_counts)) == 1 and len(rows) >= 2
    except Exception:
        return False