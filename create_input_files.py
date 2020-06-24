from utils import create_input_files, create_gender_fine_tuning_input_files, create_COCOv2_input_files

def Create_Input_Files():
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/Users/tangruixiang/Desktop/Fairness_Dataset/code/COCO/Karpathy_split/coco/dataset.json',
                       image_folder='/Volumes/Yuening\ Passport/rxtang/coco/images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='processed_data',
                       max_len=50)

def Create_Gender_Fine_Tuning_Input_Files():
    # Create input files (without word map)
    create_gender_fine_tuning_input_files(dataset='coco',
                                          karpathy_json_path='COCOv2/COCOv2_fine_tune.json',
                                          image_folder='/Volumes/rxtang/COCO',
                                          mask_folder=None,
                                          captions_per_image=5,
                                          min_word_freq=5,
                                          output_folder='/Volumes/rxtang/COCOv2_fine_tune',
                                          word_map_file='COCOv2/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json',
                                          max_len=50)
    '''
    create_gender_fine_tuning_input_files(dataset='coco',
                                          karpathy_json_path='COCOv2/COCOv2_fine_tune.json',
                                          image_folder='/Volumes/rxtang/COCO',
                                          mask_folder='/Volumes/rxtang/COCO/Fine_tune_2014',
                                          captions_per_image=5,
                                          min_word_freq=5,
                                          output_folder='/Volumes/rxtang/COCOv2_fine_tune',
                                          word_map_file='COCOv2/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json',
                                          max_len=50)
    '''

def Create_COCOv2_Input_Files():
    # Create input files (without word map)
    create_COCOv2_input_files(dataset='coco',
                       karpathy_json_path='COCOv2/COCOv2_test.json',
                       image_folder='/Volumes/rxtang/COCO',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/Volumes/rxtang/COCOv2_test',
                       word_map_file='COCOv2/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json',
                       max_len=50)

if __name__ == '__main__':
    Create_COCOv2_Input_Files()