import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
import pdf_combiner
import secrets
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def to_pdf(filename, extension):
    pdf = pytesseract.image_to_pdf_or_hocr(f'{filename}.{extension}', extension='pdf')
    with open(f'{filename}.pdf', 'w+b') as f:
        f.write(pdf)  # pdf type is bytes by default


def detect_blanks(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    i = 0
    ret = np.empty(3)
    for c in cnts:
        maxs, mins = np.amax(c, axis=0)[0], np.amin(c, axis=0)[0]
        x1, x2, y1, y2 = mins[0], maxs[0], mins[1], maxs[1]
        ret = np.vstack([ret, [x1, x2, y1]])
        # reduced_c = np.array([[x1, y1], [x2, y2]])
        # cv2.drawContours(image, [reduced_c], -1, (36, 255, 12), 3)
        image[y1 - 3:y2 + 3, x1 - 2:x2 + 2] = [255, 255, 255]
        i += 1

    ret = np.delete(ret, 0, 0)
    # cv2.imshow('image', image)
    return image, ret.astype(int)


def convert_to_jpg(pdf_path):
    # Store Pdf with convert_from_path function
    images = convert_from_path(pdf_path, poppler_path=r'C:\Program Files\poppler-22.04.0\Library\bin')
    ret = []
    for i in range(len(images)):
        # Save pages as images in the pdf
        img_name = 'page' + str(i) + '.jpg'
        images[i].save(img_name, 'JPEG')
        ret.append(img_name)
    return ret

def check_match(text, line_coords, form, x, y, w, h):
    ender = text[:-1].lower()
    for f in secrets.fill_keywords.keys():
        fill_keys = secrets.fill_keywords[f][0]
        for fill_key in fill_keys:
            if ender.endswith(fill_key) or text.endswith(fill_key):
                line, dist = find_nearest_line(line_coords, y + h, x + w)
                if dist < 40*40:
                    # im3 = im2.copy()
                    # im3[line[2] - 3:line[2] + 3, line[0]:line[1]] = [0, 0, 255]
                    print(f'MATCH [{text}] [{text[-1] if len(text) > 0 else ""}] [{fill_key}] [{line}]')
                    x = line[0] * 72 / 200
                    y = 11 * 72 - line[2] * 72 / 200
                    h = int(h * 72 / 200)
                    w = int((line[1] - line[0]) * 72 / 200)
                    options = secrets.fill_keywords[f][1]
                    form.choice(name=fill_key+str(time.time()), tooltip=fill_key,
                                x=x, y=y, borderStyle='solid', options=options,
                                borderWidth=0, height=h, width=w, fontSize=h-2, value=options[0],
                                fieldFlags=1 << 18, forceBorder=True)
                    # form.textfield(name=fill_key, tooltip=fill_key,
                    #                x=x, y=y, borderStyle='solid',
                    #                fontSize=h - 2, value=str(f), borderWidth=0,
                    #                width=w, height=h, forceBorder=True)
                    return


def find_text(img, line_coords, form):
    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    # Creating a copy of image
    im2 = img.copy()

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    i = 0
    num_cnts = len(contours)

    for cnt in contours:
        if int(i % (num_cnts / 10)) == 1:
            print(f'{int(i / num_cnts * 10) * 10}%')
        x, y, w, h = cv2.boundingRect(cnt)
        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]
        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)
        text = text[:-1]
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2)
        check_match(text, line_coords, form, x, y, w, h)
        i += 1
    # cv2.imshow('Text', im2)
    # cv2.waitKey()
    return packet


def find_nearest_line(array, yval, xval):
    array = np.asarray(array)
    a = np.abs(array[:, 2] - yval)
    b = np.abs(array[:, 0] - xval)
    distances = a * a + b * b
    idx = distances.argmin()
    return array[idx], distances[idx]


if __name__ == '__main__':
    orig = 'immblank.pdf'
    dest = 'immblank_filled.pdf'
    img_paths = convert_to_jpg(orig)
    i = 0
    paths = []
    for img_path in img_paths:
        print(f"--PAGE {i}/{len(img_paths)-1}--")
        # if i < 4:
        #     i += 1
        #     continue
        image, lines = detect_blanks(img_path)
        lines = lines[lines[:, 2].argsort()]  # Sort coords by 3rd column
        form, packet, can = pdf_combiner.get_canvas()
        find_text(image, lines, form)
        can.save()
        p = f'immblank{i}.pdf'
        paths.append(p)
        pdf_combiner.save_pdf(packet, i, orig, p)
        i += 1
    pdf_combiner.combine_pdfs(paths, dest)