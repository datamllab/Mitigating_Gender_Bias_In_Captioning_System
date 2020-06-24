import os
import json
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

COCO_val_caption_path = "captions_val_gender_2014.json"
COCO_train_caption_path = "captions_train_gender_2014.json"
COCO_K_split_path = "dataset_coco.json"
COCO_val_instance_path = "/Users/tangruixiang/Desktop/Fairness_Dataset/code/COCO/annotations2014/instances_val2014.json"
COCO_IMAGE_PATH = "/Volumes/rxtang/COCO"

Downlad_name_List = ['att2in-sc_beam5_test.json',
                  'att2in_beam5_test.json',
                  'fc-sc_beam5_test.json',
                  'fc_beam5_test.json',
                  'lrcn-sc_beam5_test.json',
                  'lrcn_beam5_test.json',
                  'nbt_beam5_test.json',
                  'td-sc_beam5_test.json',
                  'td_beam5_test.json']

GAIC_name_list = ['gaic_beam5_test.json',
                    'att_beam5_result.json',
                  'adapatt_beam5_result.json',
                  'gaices_beam5_test.json',
                  'epoch21_beam5_test.json']

class Figure:
    def __init__(self):
        self.result_dict = {}
        self.k_split = None
        self.id_to_object = {}
        self.woman_list = ['woman', 'women', 'girl', 'girls']
        self.man_list = ['man', 'men', 'boy', 'boys']
        self.neutral = ['person', 'people', 'human', 'baby']

    def _gender_for_caption(self, caption):
        is_woman = False
        is_man = False
        is_neutral = False
        _caption = caption.lower()

        for item in self.woman_list:
            if re.findall('\W' + item + '\W', _caption):
                is_woman = True
                break
        for item in self.man_list:
            if re.findall('\W' + item + '\W', _caption):
                is_man = True
                break
        for item in self.neutral:
            if re.findall('\W' + item + '\W', _caption):
                is_neutral = True
                break

        if is_woman is True and is_man is not True:
            return 0
        elif is_woman is False and is_man is True:
            return 1
        elif is_woman is True and is_man is True:
            return 2
        elif is_woman is False and is_man is False and is_neutral is True:
            return 2
        else:
            return 3

    # read model from github
    def read_download_json_data(self, path):
        with open(path, 'rb') as f:
            json_file = json.load(f)
            return json_file

    # read model produce by ourself
    def read_gaic_json_data(self, path):
        with open(path, 'rb') as f:
            json_file = json.load(f)
            image_dict = {}
            for item in json_file:
                image_dict[str(item['image_id'])] = {'caption': item['caption'], 'image_id': item['image_id']}
            final_dict = {}
            final_dict['imgToEval'] = image_dict
        return final_dict

    def read_result(self, root_path, Downlad_name_list, GAIC_name_list):
        for name in Downlad_name_list:
            model_name = name.split('_')[0]
            file_path = os.path.join(root_path, name)
            json_file = self.read_download_json_data(file_path)
            self.result_dict[model_name] = json_file

        for name in GAIC_name_list:
            model_name = name.split('_')[0]
            file_path = os.path.join(root_path, name)
            json_file = self.read_gaic_json_data(file_path)
            self.result_dict[model_name] = json_file

    def calulate_error_rate(self, test_image_list):
        for name, result in self.result_dict.items():
            temp_result = copy.deepcopy(result['imgToEval'])

            set1 = set()
            for item in temp_result:
                set1.add(int(item))

            set2 = set()
            for item in test_image_list:
                set2.add(int(item['coco_id']))

            set3 = set1 & set2


            result_list = []
            for item in test_image_list:
                image_id = item['coco_id']
                gender_label = item['gender']
                caption = temp_result[str(image_id)]['caption']
                gender_infer = self._gender_for_caption(caption)
                #temp_result[str(image_id)]['gender_inference'] = gender_infer
                #temp_result[str(image_id)]['gender'] = gender_label
                #changed_set.add(int(image_id))
                result_list.append({'coco_id': item['coco_id'],
                               'gender': gender_label,
                               'gender_inference': gender_infer})

            total_error, man_ration, man_correct, man_error, man_other,\
            woman_correct, woman_error, woman_other = self._get_classification_result_list(result_list)
            print('model name:', name)
            print('total_error:', total_error)
            print('man_ratio:', man_ration)
            print('woman_correct:', woman_correct, 'woman_error:', woman_error, "woman_other", woman_other)
            print('man_correct:', man_correct, 'man_error:', man_error, "man_other", man_other)

    def _get_classification_result_list(self, result_data):
        num_woman_correct = 0
        num_woman_wrong = 0
        num_woman_other = 0
        num_man_correct = 0
        num_man_wrong = 0
        num_man_other = 0
        total_man = 0
        total_woman = 0
        discard_woman = 0
        discard_man = 0

        n = 0
        for image in result_data:
            if 'gender_inference' in image:
                n+=1
            else:
                print(image['coco_id'])



        for image in result_data:
            if image['gender'] == 0:
                total_woman += 1
                if image['gender_inference'] == 0:
                    num_woman_correct += 1
                elif image['gender_inference'] == 1 :
                    num_woman_wrong += 1
                elif image['gender_inference'] == 3 :
                    #num_woman_wrong += 1
                    discard_woman += 1
                else:
                    num_woman_other += 1

            elif image['gender'] == 1:
                total_man += 1
                if image['gender_inference'] == 1:
                    num_man_correct += 1
                elif image['gender_inference'] == 0 or image['gender_inference'] == 1:
                    num_man_wrong += 1
                elif image['gender_inference'] == 3 :
                    #num_man_wrong += 1
                    discard_man += 1
                else:
                    num_man_other += 1

        total_error = np.array(num_woman_wrong + num_man_wrong)/(total_man + total_woman)
        man_ration = (num_man_correct + num_man_wrong) / (num_man_correct + num_man_wrong + num_woman_correct + num_woman_wrong)
        man_correct = num_man_correct/(num_man_wrong + num_man_correct + num_man_other)
        man_error = num_man_wrong/(num_man_wrong + num_man_correct + num_man_other)
        man_other = num_man_other/(num_man_wrong + num_man_correct + num_man_other)
        woman_correct = num_woman_correct / (num_woman_wrong + num_woman_correct + num_woman_other)
        woman_error = num_woman_wrong / (num_woman_wrong + num_woman_correct + num_woman_other)
        woman_other = num_woman_other / (num_woman_wrong + num_woman_correct + num_woman_other)
        return total_error, man_ration, man_correct, man_error, man_other, woman_correct, woman_error, woman_other

    # this function will add (1)gender (2)object list into k split
    def combine_k_split_COCO(self, coco_path_val, coco_path_train, coco_path_instance, k_split_path):
        coco_dict = {}
        json_val = self.read_download_json_data(path=coco_path_val)
        json_train = self.read_download_json_data(path=coco_path_train)
        json_k_split = self.read_download_json_data(path=k_split_path)
        json_instance = self.read_download_json_data(path=coco_path_instance)
        for item in json_train['images']:
            coco_dict[item['id']] = {'gender': item['gender'], 'category_id': item['category_id'], 'id': item['id']}
        for item in json_val['images']:
            coco_dict[item['id']] = {'gender': item['gender'], 'category_id': item['category_id'], 'id': item['id']}

        for item in json_k_split['images']:
            id = item['cocoid']
            gender = coco_dict[id]['gender']
            category_id = coco_dict[id]['category_id']
            item['gender'] = gender
            item['category_id'] = category_id

        json_k_split['categories'] = json_instance['categories']
        jsonData = json.dumps(json_k_split)
        file = open('Ksplit_gender_category.json', 'w')
        file.write(jsonData)
        file.close()
        print('write json')

    def show_example(self, image_path, data):

        for object in self.k_split['categories']:
            self.id_to_object[object['id']] = object['name']

        for item in data['images']:
            img_path = os.path.join(image_path, item['filepath'], item['filename'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            catergories = ""
            for object in item['category_id'].split():
                catergories += " " + self.id_to_object[int(object)]
            title = ''
            if item['gender'] == 0:
                title = 'woman'
            elif item['gender'] == 1:
                title = 'man'
            elif item['gender'] == 2:
                title = 'neutral'
            elif item['gender'] == 3:
                title = 'discard'

            title += ':' + catergories
            plt.title(title)

            # close the axis value
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            plt.show()


if __name__ == "__main__":
    figure = Figure()
    #figure.combine_k_split_COCO(coco_path_val=COCO_val_caption_path,
    #                            coco_path_train=COCO_train_caption_path,
    #                            coco_path_instance=COCO_val_instance_path,
    #                            k_split_path=COCO_K_split_path)
    figure.k_split = figure.read_download_json_data(path='Ksplit_gender_category.json')
    #figure.show_example(image_path=COCO_IMAGE_PATH, data=figure.k_split)
    figure.read_result(root_path='json_results', Downlad_name_list=Downlad_name_List, GAIC_name_list=GAIC_name_list)
    figure.calulate_error_rate(test_image_list=figure.k_split['secret_test'])
