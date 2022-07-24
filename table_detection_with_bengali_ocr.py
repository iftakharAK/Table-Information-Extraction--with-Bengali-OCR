# @title Import Libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from pdf2image import convert_from_path

# @title Sort All The Detected Contours


def sort_contours(cnts, method="left-to-right"):

    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return (cnts, boundingBoxes)

# @title Create Kernel


def get_kernels(img, img_bin, thresh):

    kernel_len = np.array(img).shape[1]//100
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    return img_bin, kernel,  ver_kernel, hor_kernel

# @title Apply Vertival Kernels


def get_vertical_lines(img_bin, ver_kernel):

    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

    return vertical_lines

# @title Apply Horizontal Kernels


def get_horizontal_lines(img_bin, hor_kernel):

    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

    return horizontal_lines

# @title Get the List of Boxes


def get_list_of_box(img, contours):

    box = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if (20 < h < 2000):
            image = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            crop_img = img[y:y+h, x:x+w]
            cv2_imshow(crop_img)

            box.append([x, y, w, h])

    return box

# @title Count Row & Column


def get_row_and_columns(box, mean):

    row = []
    column = []
    j = 0

    for i in range(len(box)):
        if(i == 0):
            column.append(box[i])
            previous = box[i]
        else:
            if(box[i][1] <= previous[1]+mean/2):
                column.append(box[i])
                previous = box[i]
                if(i == len(box)-1):
                    row.append(column)
            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    return row, column

# @title Count Total Cells


def count_cells(row):

    countcol = 0

    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    center = [int(row[i][j][0]+row[i][j][2]/2)
              for j in range(len(row[i])) if row[0]]
    center = np.array(center)
    center.sort()

    return countcol, center

# @title Arrange all the Bounding boxes in Order


def arrange_boxes_in_order(row, countcol, center):

    finalboxes = []

    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    return finalboxes

# @title Apply OCR on all of the detected cells


def get_cell_ocr_string_list(finalboxes, bitnot):

    outer = []

    for i in range(len(finalboxes)):

        for j in range(len(finalboxes[i])):

            inner = ''
            if(len(finalboxes[i][j]) == 0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):

                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x+h, y:y+w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(
                        finalimg, 2, 2, 2, 2,   cv2.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv2.resize(
                        border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel, iterations=1)
                    erosion = cv2.erode(dilation, kernel, iterations=1)
                    cv2_imshow(erosion)

                    out = pytesseract.image_to_string(erosion, lang="ben")
                    if(len(out) == 0):
                        out = pytesseract.image_to_string(erosion, lang="ben")
                    inner = inner + " " + out
                outer.append(inner)

    return outer

# @title Extract Table Contents From Image


def extract_table_from_image(file_path):

    file = r''+file_path
    img = cv2.imread(file, 0)
    thresh, img_bin = cv2.threshold(
        img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255-img_bin

    img_bin, kernel,  ver_kernel, hor_kernel = get_kernels(
        img, img_bin, thresh)

    vertical_lines = get_vertical_lines(img_bin, ver_kernel)

    horizontal_lines = get_horizontal_lines(img_bin, hor_kernel)

    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(
        img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)

    contours, hierarchy = cv2.findContours(
        img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)

    box = get_list_of_box(img, contours)

    row, column = get_row_and_columns(box, mean)

    countcol = 0

    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    center = [int(row[i][j][0]+row[i][j][2]/2)
              for j in range(len(row[i])) if row[0]]
    center = np.array(center)
    center.sort()
    finalboxes = arrange_boxes_in_order(row, countcol, center)

    outer = get_cell_ocr_string_list(finalboxes, bitnot)

    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    data = dataframe.style.set_properties(align="left")

    return dataframe

# @title Get All The Pdf File Paths


def read_all_pdf_file_names(pdf_file_path):

    all_file_list = os.listdir(pdf_file_path)

    pdf_file_list = list(
        filter(lambda file_name: '.pdf' in file_name, all_file_list))

    return pdf_file_list

# @title Convert Pdf to Images


def convert_pdf_to_images(pdf_file_path, image_file_save_directory):

    pdf_file_list = read_all_pdf_file_names(pdf_file_path)

    for pdf_file in pdf_file_list:

        if not os.path.isdir(image_file_save_directory):

            os.mkdir(image_file_save_directory)

    pages = convert_from_path(pdf_file_path+'/'+pdf_file, 500)

    for i in range(0, len(pages)):

        pages[i].save(image_file_save_directory+'/page_'+str(i)+'.jpg', 'JPEG')

# @title Get All The Converted Image Paths


def get_all_converted_images_file_names(converted_image_save_directory):

    all_file_list = os.listdir(converted_image_save_directory)

    image_file_list = list(
        filter(lambda file_name: '.jpg' in file_name, all_file_list))

    return image_file_list


def main():

    pdf_directory = 'InputPdf'

    converted_image_save_directory = '/content/ConvertedImagesFromPdf'

    table_df_save_directory = 'OutputExcelFIles'

    convert_pdf_to_images(pdf_directory, converted_image_save_directory)

    image_file_list = get_all_converted_images_file_names(
        converted_image_save_directory)

    if not os.path.isdir(table_df_save_directory):

        os.mkdir(table_df_save_directory)

    for img_file in image_file_list:

        try:
            print(converted_image_save_directory+'/'+img_file)
            dataframe = extract_table_from_image(
                converted_image_save_directory+'/'+img_file)
            dataframe.to_excel(table_df_save_directory+'/' +
                               img_file.split('.')[0]+'.xlsx', engine='xlsxwriter')

        except:

            print("Didn't Find Any Table Image")


if __name__ == "__main__":
    main()
