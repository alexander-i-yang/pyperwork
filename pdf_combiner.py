from PyPDF2 import PdfFileWriter, PdfFileReader, PdfMerger
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def get_canvas():
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    form = can.acroForm
    return form, packet, can

def save_pdf(packet, pagenum, original, dest):
    #move to the beginning of the StringIO buffer
    packet.seek(0)

    # create a new PDF with Reportlab
    new_pdf = PdfFileReader(packet, strict=False)
    # read your existing PDF
    existing_pdf = PdfFileReader(open(original, "rb"))
    output = PdfFileWriter()
    # add the "watermark" (which is the new pdf) on the existing page
    page = existing_pdf.getPage(pagenum)
    if new_pdf.numPages > 0:
        page.mergePage(new_pdf.getPage(0))
    output.addPage(page)
    # finally, write "output" to a real file
    outputStream = open(dest, "wb")
    output.write(outputStream)
    outputStream.close()

def combine_pdfs(pdf_paths, dest):
    merger = PdfMerger()

    for pdf in pdf_paths:
        merger.append(pdf)

    merger.write(dest)
    merger.close()
