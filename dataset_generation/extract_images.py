import fitz
import io
from PIL import Image
import argparse
import cv2
import os
import json

name_image_pairing = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 2
}

image_type_pairing = {
    0 : "RGB",
    1 : "IR",
    2 : "RGB",
    3 : "IR",
    4 : "RGB",
    5 : "IR"
}

defect_type_pairing = {
    0 : 0,
    1 : 0,
    2 : 1,
    3 : 1,
    4 : 2,
    5 : 2
}

def options():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('pdf_path', type=str, help='Path to the pdf file')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset to generate')
    parser.add_argument('--page_init', type=int, help="Page to begin extracting images from")
    parser.add_argument('--page_end', type=int, help="Page to end extracting images from")
    return parser.parse_args()

def valid_size(h,w):

    ret = True if (h == 177 or h == 191) and w == 142 else False
    return ret

def get_useful_images_in_page(page):

    images_in_page = []

    dict_page = page.get_text("dict")["blocks"]
    images = list(filter(lambda x : x["type"] == 1 , dict_page))
    for idx, block in enumerate(images):
        if valid_size( block["width"],block["height"]):
            images_in_page.append(block["image"])

    return images_in_page


def clear_non_words(text):
    cleared_text = []
    for tuple in text:
        for element in tuple:
            if isinstance(element, str):
                cleared_text.append(element)

    return cleared_text

def get_filenames_and_deffects_in_page(page):
    text = page.get_text("words")
    text = clear_non_words(text)
    file_names = []
    deffects   = []
    for word in text:
        if word.startswith("DJI_"):
            file_names.append(word)

    d = ""
    in_deffect = False
    for word in text:
        
        if word == "Anom:":
            d = ""
            in_deffect = True
        
        elif word == "Delta":
            deffects.append(d[:-1])
            in_deffect = False
        
        elif in_deffect:
            d += word + " "


    return file_names, deffects

def assign_filename_and_save(images, filenames, ds_name, defects_in_page):

    for i in range(0,len(name_image_pairing.items()),2):
        print(i)
        if i // 2 < len(filenames):
            photo_name = filenames[i // 2][:-4]
            ap         = appearances[photo_name]
            print(f"Current appearance value {ap}")
            os.makedirs(os.path.join(ds_name, photo_name), exist_ok= True)

            print(image_type_pairing[i])
            print(image_type_pairing[i+1])

            if ap > 0 :
                rgb_photo_name = os.path.join("./", ds_name, photo_name, photo_name + '_' + image_type_pairing[i]   + "("+ str(ap) +")" +'.JPG')
                ir_photo_name  = os.path.join("./", ds_name, photo_name, photo_name + '_' + image_type_pairing[i+1] + "("+ str(ap) +")" +'.JPG')
            else:
                rgb_photo_name = os.path.join("./", ds_name, photo_name, photo_name + '_' + image_type_pairing[i]   + '.JPG')
                ir_photo_name  = os.path.join("./", ds_name, photo_name, photo_name + '_' + image_type_pairing[i+1] +'.JPG')

            if photo_name not in defects_per_image.keys():
                defects_per_image[photo_name] = []

            defects_per_image[photo_name].append({
                 "image_number" : ap,
                 "defect"       : defects_in_page[i // 2]})

            appearances[photo_name] += 1


            fout    = open(rgb_photo_name, "xb")
            fout.write(images[i])
            fout.close()

            fout    = open(ir_photo_name, "xb")
            fout.write(images[i+1])
            fout.close()
                
            print(f"Saved RGB image with name {rgb_photo_name} and ir image with name {ir_photo_name}")
            print(f"Increased appearance from {ap} to {appearances[photo_name]}")


def check_correct_number(path):

    for f in os.listdir(path):
        counter = 0
         
        for sf in os.listdir(os.path.join(path,f)):
            counter += 1

        assert counter // 2 == appearances[f] , f" In directory {f} there are {counter} images while its appearances where {appearances[f]}"


if __name__ == "__main__":

    args = options()
    os.makedirs(args.dataset_name, exist_ok=True)

    pdf_file = fitz.open(args.pdf_path)
    
    page = pdf_file.pages(start = args.page_init, stop = args.page_end, step = 1)
    
    appearances = {}

    pcount = args.page_init + 1

    defects_per_image = {}

    for p in page:
    
        images_in_page = get_useful_images_in_page(p)

        if len(images_in_page):
            print(
                f"[+] Found a total of {len(images_in_page)} images in page {pcount}")
        else:
            print("[!] No images found on page", pcount)


        print(f"[+] Found a total of {len(images_in_page)} useful images")

        file_names_deffects = get_filenames_and_deffects_in_page(p)

        file_names = file_names_deffects[0]
        defects_per_image_in_page   = file_names_deffects[1]

        print("[info] Found the following file names: \n", "\n ".join(file_names))

        for f in file_names:
            if f[:-4] not in list(appearances.keys()):
                appearances[f[:-4]] = 0

        assign_filename_and_save(images_in_page, file_names, args.dataset_name, defects_per_image_in_page)

        pcount += 1

    with open("./" + args.dataset_name + "/defects_tags.json", "x") as outfile:
        json.dump(defects_per_image, outfile, indent = 1)

    check_correct_number(args.dataset_name)



    

