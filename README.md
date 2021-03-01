# Mitigating Gender Bias In Captioning System

This is the pytorch implemention for **WWW（The Web Conference）2021** paper “Mitigating Gender Bias In Captioning system” [**[Link]**](https://arxiv.org/abs/2006.08315)[**[cite]**](https://scholar.googleusercontent.com/scholar.bib?q=info:QGqoJ2Lx9X0J:scholar.google.com/&output=citation&scisdr=CgUkJWA5EJDGksOPh3o:AAGBfm0AAAAAYDyKn3o8-XFn66hwVXKBwyfl2hQV7I7d&scisig=AAGBfm0AAAAAYDyKn0DAJ2hfctjZm-AESmKz0m9nl_jX&scisf=4&ct=citation&cd=-1&hl=en). Recent studies have shown that captioning datasets, such as the COCO dataset, may contain severe social bias which could potentially lead to unintentional discrimination in learning models. 
In this work, we specifically focus on the gender bias problem. 

<p align="center">
<img src="https://github.com/CaptionGenderBias2020/Mitigating_Gender_Bias_In_Captioning_System_NIPS2020/blob/master/Figures/Examples_Figure-1.jpg" img width="750" height="420" />
</p>

## Environment
- `pytorch==0.41`
- `torchvision==0.2.1`
- `python==3.6`
- `scikit-image==0.16.2`
- `numpy==1.18.1`
- `h5py==2.10.0`

## COCO-GB: Dataset Creation and Analysis
The COCO-GB dataset are created for quantifying gender bias in models. We construct COCO-GB
v1 based on a widely used split and create a gender-balanced secret test dataset. COCO-GB v2 is
created by reorganizing the train/test split so that the gender-object joint distribution in training set is
very different from testing set. 

- Gender-object joint distribution of COCO training dataset
<p align="center">
<img src="https://github.com/CaptionGenderBias2020/Mitigating_Gender_Bias_In_Captioning_System_NIPS2020/blob/master/Figures/training_distribution.png" img width="490" height="164" />
</p>

- Gender-object joint distribution of original COCO test dataset
<p align="center">
<img src="https://github.com/CaptionGenderBias2020/Mitigating_Gender_Bias_In_Captioning_System_NIPS2020/blob/master/Figures/ori_test_distribution.png" img width="490" height="164" />
</p>

- Gender-object joint distribution of COCO-GB v1 secret test dataset
<p align="center">
<img src="https://github.com/CaptionGenderBias2020/Mitigating_Gender_Bias_In_Captioning_System_NIPS2020/blob/master/Figures/secret_test_distribution.png" img width="490" height="164" />
</p>

- Gender-object joint distribution of COCO-GB v2 test dataset
<p align="center">
<img src="https://github.com/CaptionGenderBias2020/Mitigating_Gender_Bias_In_Captioning_System_NIPS2020/blob/master/Figures/COCOv2_test_distribution.png" img width="490" height="164" />
</p>


## Benchmarking Captioning Models on COCO-GB v1
  
To reveal the gender bias in existing models, we utilize the gender prediction performance to quantify 
bias learned by models. Models are trained on Karpathy split, obtain caption quality from
 original test split, and evaluate gender prediciton performance on the COCO-GB v1 secret test dataset.

### Evaluation of Gender Prediction Accuracy
```
python Benchmarking_existing_models/benchmarking.py 
```
We saved the caption results of baseline models in folder Benchmarking_existing_models/json_results

### Evaluation of Caption Quality
```
Please download the COCO official Evaluation Tool from https://github.com/tylin/coco-caption
```
## Image Captioning Model with Guided Attention
We propose a novel Guided Attention Image Captioning model (GAIC) to mitigate gender bias 
by self-supervising on model’s visual attention. GAIC has two complementary streams to
encourage the model to explore correct gender features. The training pipeline can seamlessly add
extra supervision to accelerate the self-exploration process. Besides, GAIC is model-agnostic and
can be easily applied to various captioning models.

<p align="center">
<img src="https://github.com/CaptionGenderBias2020/Mitigating_Gender_Bias_In_Captioning_System_NIPS2020/blob/master/Figures/model_cropped-1.jpg" img width="820" height="250" />
</p>

### Data pipeline
See `create_input_files()` in [`utils.py`](https://github.com/CaptionGenderBias2020/Mitigating_Gender_Bias_In_Captioning_System_NIPS2020/blob/master/utils.py).

This reads the data downloaded and saves the following files –

An **HDF5 file containing images for each split in an `I, 3, 256, 256` tensor**, where `I` is the number of images in the split. Pixel values are still in the range [0, 255], and are stored as unsigned 8-bit `Int`s.
A **JSON file for each split with a list of `N_c` * `I` encoded captions**, where `N_c` is the number of captions sampled per image. These captions are in the same order as the images in the HDF5 file. Therefore, the `i`th caption will correspond to the `i // N_c`th image.
A **JSON file for each split with a list of `N_c` * `I` caption lengths**. The `i`th value is the length of the `i`th caption, which corresponds to the `i // N_c`th image.
A **JSON file which contains the `word_map`**, the word-to-index dictionary.

### Training Baseline Model
```
python train.py 
```
We choose Att model as the baseline model. We train baseline for 5 epochs on COCO dataset.
### Training GAIC Model
```
python fine_tune.py 
```
We Construct GAIC model based on the baseline model. Fine-tune the GAIC model on the fine-tune set for 1 epoch. For training GAICes model, please set the `supervised_training = True`.
### Evaluation of Caption Quality and Gender Accuracy
```
python eval.py
```
This code will evaluate the model caption quality on the original test dataset, and evaluate model gender prediction accuracy on COCO-GB v1 or COCO-GB v2.
### Show Qualitative Results
```
python caption.py
```
This code will show the attention maps for inputting images.
