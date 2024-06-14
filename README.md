# APDDv2: Aesthetics of Paintings and Drawings Dataset with Artist Labeled Scores and Comments

This is an open-source aesthetics of paintings and drawings dataset called APDD, which was completed by the victory-lab.

## Introduction

The Aesthetics Paintings and Drawings Dataset (APDDv2) stands as the pioneering comprehensive painting dataset, encompassing 24 distinct artistic categories and 10 aesthetic attributes. Comprising 10,023 painting images, 85,191 Scoring labels, and 6,249 linguistic comments, APDDv2 addresses the scarcity of datasets in the realm of aesthetic evaluation of artistic images. Its primary aim is to furnish researchers and practitioners with high-quality painting data, thereby fostering the development and application of automatic aesthetic evaluation methodologies.

ArtCLIP, tailored specifically for particular painting styles, is an artistic assessment network trained on the APDDv2 dataset. The ArtCLIP models excel in providing a comprehensive assessment of the total aesthetic score of artistic images, along with scoring various aesthetic attributes.

## APDDv2

- `APDDv2-10023.csv`: This file contains annotations for the APDDv2 dataset, including artistic categories, total aesthetic scores, aesthetic attribute scores and language comments for 10,023 images.
- `filesource.csv`: This file records the source URLs for 1892 images in the APDD dataset. The remaining images are sourced from Wikiart(https://www.wikiart.org/) and student painting assignments.

We provide alternatives to obtain the dataset:

Baidu Netdisk: [Click here to download](https://drive.google.com/file/d/1ap5dhuEgpPC5PrJozAu2V), Access code: 9y91

Google Drive: [Click here to download](https://drive.google.com/file/d/1ap5dhuEgpPC5PrJozAu2VFmUNIRZrar2/view?usp=drive_link)

## ArtCLIP

Clone the repository:

```sh
git clone https://github.com/BestiVictory/APDDv2.git
cd APDDv2/
```

Install dependencies (works with python3.9):

```
pip3 install -r requirements.txt
```

The model checkpoints and pretrained data can be downloaded from 
[gcloud directory link](https://drive.google.com/drive/folders/1AOVKmSqZCW09J_Ypr7KzSYfRxQre-w_m?usp=drive_link).
The folder contains the following items:

-   **./modle_weights/0.ArtCLIP_weight--e11-train2.4314-test4.0253_best.pth**: ArtCLIP model weight, which fine-tuned vanilla CLIP on categorized DPC2022 dataset.
-   **./modle_weights/1.Score_reg_weight--e4-train0.4393-test0.6835_best.pth**: ArtCLIP finetuned on the APDDv2 database for predicting total aesthetic scores.
-   **./modle_weights/2.Theme and logic_reg_weight--e5-train0.3792-test0.5953_best.pth**: ArtCLIP for predicting theme and logic scores.
-   **./modle_weights/3.Creativity_reg_weight--e5-train0.4212-test0.7122_best.pth**: ArtCLIP for predicting creativity scores.
-   **./modle_weights/4.Layout and composition_reg_weight--e6-train0.2783-test0.6342_best.pth**: ArtCLIP for predicting layout and composition scores.
-   **./modle_weights/5.Space and perspective_reg_weight--e7-train0.2168-test0.5998_best.pth**: ArtCLIP for predicting space and perspective scores.
-   **./modle_weights/6.The sense of order_reg_weight--e5-train0.3708-test0.6206_best.pth**: ArtCLIP for predicting sense of order scores.
-   **./modle_weights/7.Light and shadow_reg_weight--e7-train0.1937-test0.6518_best.pth**: ArtCLIP for predicting light and shadow scores.
-   **./modle_weights/8.Color_reg_weight--e5-train0.2905-test0.5871_best.pth**: ArtCLIP for predicting color scores.
-   **./modle_weights/9.Details and texture_reg_weight--e4-train0.4385-test0.7034_best.pth**: ArtCLIP for predicting details and texture scores.
-   **./modle_weights/10.The overall_reg_weight--e3-train0.5131-test0.6343_best.pth**: ArtCLIP for predicting the overall scores.
-   **./modle_weights/11.Mood_reg_weight--e7-train0.3108-test0.7097_best.pth**: ArtCLIP for predicting mood scores.


