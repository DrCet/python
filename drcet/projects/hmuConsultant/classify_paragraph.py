import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


from drcet.data.imagePreprocess import paragraph_segment, line_segment
import json
import cv2
import pytesseract
import matplotlib.pyplot as plt

def create_json_dataset(data, json_path, img_path, label, index):    
    entry = {
        "image": img_path,
        "label": label
    }
    if index % 5 != 0:
        data["train"].append(entry)
    else:
        data["test"].append(entry)
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def save_text_to_gt_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


def page_to_paragraph(page_img, bboxes, file):
    n_par = 0
    paragraphs = []
    n_par_list = []
    for bbox in bboxes:
        x,y,w,h = bbox
        if h>32 and w>32:
            try:
                cropped_paragraph = img[y-10:y+h+6, x-10:x+w+6]
                cropped_paragraph = cv2.copyMakeBorder(
                cropped_paragraph,
                8,8,8,8,
                cv2.BORDER_CONSTANT,
                value = (255,255,255)
                )   
                n_par += 1 #number of stored paragraph
                n_par_list.append(n_par)
                par_path = f'projects/hmuConsultant/paragraph_classification_dataset/images/{file[:-3]}__{n_par}.jpg'

                plt.imshow(cropped_paragraph)
                plt.axis('off')
                plt.title('Paragraph')
                plt.show()

                cv2.imwrite(par_path, cropped_paragraph)
                label = input(f'Paragraph label {file[:-3]}__{n_par}___:') # {0: ignore, 1: true, 2: table}
                create_json_dataset(par_data, 'projects/hmuConsultant/paragraph_classification_dataset/annotations.json', par_path, label, n_par)
                paragraphs.append(cropped_paragraph)
            except:
                pass
            
    return paragraphs, n_par_list


def paragraph_to_lines(paragraphs, n_par_list, file):
    total_lines = 0
    for i, par in enumerate(paragraphs):
        paragraph, boxes = line_segment(image= par, scale = (1,1))
        n_par = n_par_list[i]
        n_line = 0
        for box in boxes:
            x,y,w,h = box
            if w > h and w > 32 and h > 16:
                try:
                    cropped_line = paragraph[y-10:y+h+6, x-10:x+w+6]
                    n_line += 1
                    total_lines += 1
                    text = pytesseract.image_to_string(cropped_line, config='--psm 6', lang ='umh')
                    # plt.figure(figsize= (1500, 500))
                    # plt.imshow(cropped_line)
                    # plt.axis('off')
                    # plt.title(f'Line: {text}')
                    # plt.show()
                    line_path = f'projects/hmuConsultant/line_dataset/images/{file[:-3]}__{n_par}__{n_line}.jpg'
                    cv2.imwrite(line_path, cropped_line)
                    text = pytesseract.image_to_string(cropped_line, config='--psm 6', lang ='umh')
                    label = 0 # input('Label for this line:') # {0: need to correct, 1: true}
                    create_json_dataset(line_data, 'projects/hmuConsultant/line_dataset/annotations.json', line_path, label, total_lines)
                    save_text_to_gt_file(text, f'projects/hmuConsultant/line_dataset/images/{file[:-3]}__{n_par}__{n_line}.gt.txt')
                    print(f'Done {total_lines} lines')
                except:
                    pass
                

org_dir = "book_1"
n_par = 0
with open('projects/hmuConsultant/paragraph_classification_dataset/annotations.json', 'r') as json_file:
    par_data = json.load(json_file)
with open('projects/hmuConsultant/line_dataset/annotations.json', 'r') as json_file:
    line_data = json.load(json_file)

train_data = par_data['train']
true_train = [example for example in train_data if example['label'] in ["0", "1"]]
print(len(true_train)/ len(train_data))

test_data = par_data['test']
true_test = [example for example in test_data if example['label'] in ['0', '1']]
print(len(true_test)/len(test_data))




# n_file_walked = 0
# for file in os.listdir(org_dir)[437:]: 
#     path = os.path.join(org_dir, file)
#     img, bboxes = paragraph_segment(path)
#     img = cv2.copyMakeBorder(
#             img,
#             8,8,8,8,
#             cv2.BORDER_CONSTANT,
#             value = (255,255,255)
#         )
#     copy = img.copy()
#     for bbox in bboxes:

#         x,y,w,h = bbox
#         if h>32 and w>32:
#             try:
#                 cv2.rectangle(copy, (x-10,y-10), (x+w+6, y+h+6), (0,255,255), 2)
#             except:
#                 pass
#     plt.imshow(copy)
#     plt.axis('off')
#     plt.title('Page')
#     plt.show()



#     paragraphs, n_par_list = page_to_paragraph(img, bboxes, file)
#     # paragraph_to_lines(paragraphs, n_par_list, file)

#     n_file_walked = +1
#     print(f'Segmented {n_file_walked} pages')
