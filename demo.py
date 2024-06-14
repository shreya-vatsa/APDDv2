import os
import sys
sys.path.append('/home/ubuntu/YourPath')
import torch
import numpy as np
import warnings
import models.clip as clip
warnings.filterwarnings("ignore")
from models.aesclip import AesCLIP_reg
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

def init():
    parser = argparse.ArgumentParser(description="PyTorch")
    args = parser.parse_args()
    return args

opt = init()

def get_score(opt, y_pred):
    score_np = y_pred.data.cpu().numpy()
    return y_pred, score_np

def load_model(weight_path, device):
    model = AesCLIP_reg(clip_name='ViT-B/16', weight="./modle_weights/0.AesCLIP_weight--e11-train2.4314-test4.0253_best.pth")
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    return model

def predict(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, preprocess = clip.load('ViT-B/16', device)

    # 加载11个模型
    score_model = load_model("./modle_weights/1.Score/AesCLIP_reg_weight--e4-train0.4393-test0.6835_best.pth", device)
    theme_model = load_model("./modle_weights/2.Theme and logic/AesCLIP_reg_weight--e5-train0.3792-test0.5953_best.pth", device)
    creativity_model = load_model("./modle_weights/3.Creativity/AesCLIP_reg_weight--e5-train0.4212-test0.7122_best.pth", device)
    layout_model = load_model("./modle_weights/4.Layout and composition/AesCLIP_reg_weight--e6-train0.2783-test0.6342_best.pth", device)
    space_model = load_model("./modle_weights/5.Space and perspective/AesCLIP_reg_weight--e7-train0.2168-test0.5998_best.pth", device)
    sense_model = load_model("./modle_weights/6.The sense of order/AesCLIP_reg_weight--e5-train0.3708-test0.6206_best.pth", device)
    light_model = load_model("./modle_weights/7.Light and shadow/AesCLIP_reg_weight--e7-train0.1937-test0.6518_best.pth", device)
    color_model = load_model("./modle_weights/8.Color/AesCLIP_reg_weight--e5-train0.2905-test0.5871_best.pth", device)
    details_model = load_model("./modle_weights/9.Details and texture/AesCLIP_reg_weight--e4-train0.4385-test0.7034_best.pth", device)
    overall_model = load_model("./modle_weights/10.The overall/AesCLIP_reg_weight--e3-train0.5131-test0.6343_best.pth", device)
    mood_model = load_model("./modle_weights/11.Mood/AesCLIP_reg_weight--e7-train0.3108-test0.7097_best.pth", device)

    image = Image.open('./Artimage.jpg').convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # 预测11个分数
    score_pred = score_model(image_input)
    theme_pred = theme_model(image_input)
    creativity_pred = creativity_model(image_input)
    layout_pred = layout_model(image_input)
    space_pred = space_model(image_input)
    sense_pred = sense_model(image_input)
    light_pred = light_model(image_input)
    color_pred = color_model(image_input)
    details_pred = details_model(image_input)
    overall_pred = overall_model(image_input)
    mood_pred = mood_model(image_input)

    _, score_pred = get_score(opt, score_pred)
    _, theme_pred = get_score(opt, theme_pred)
    _, creativity_pred = get_score(opt, creativity_pred)
    _, layout_pred = get_score(opt, layout_pred)
    _, space_pred = get_score(opt, space_pred)
    _, sense_pred = get_score(opt, sense_pred)
    _, light_pred = get_score(opt, light_pred)
    _, color_pred = get_score(opt, color_pred)
    _, details_pred = get_score(opt, details_pred)
    _, overall_pred = get_score(opt, overall_pred)
    _, mood_pred = get_score(opt, mood_pred)

    print('Total Aesthetic Score: ', score_pred * 10)
    print('Theme and Logic Score: ', theme_pred)
    print('Creativity Score: ', creativity_pred)
    print('Layout and Composition Score: ', layout_pred)
    print('Space and Perspective Score: ', space_pred)
    print('Sense of Order Score: ', sense_pred)
    print('Light and Shadow Score: ', light_pred)
    print('Color Score: ', color_pred)
    print('Details and Texture Score: ', details_pred)
    print('The Overall Score: ', overall_pred)
    print('Mood Score: ', mood_pred)

if __name__ == "__main__":
    predict(opt)
