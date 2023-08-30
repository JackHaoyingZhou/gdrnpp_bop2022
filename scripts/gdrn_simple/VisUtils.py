import sys
from pathlib import Path
from typing import List, Tuple
from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import enum
import mmcv

font_path = Path(__file__).parent.parent / "fonts" / "droid_sans_mono.ttf"

class Color(enum.Enum):
    RED = (255,0,0)
    GREEN =  (0,255,0)
    BLUE = (0,0,255)
    YELLOW = (255,255,0)
    CYAN = (0,255,255)
    MAGENTA = (255,0,255)
    WHITE = (255,255,255)
    BLACK = (0,0,0)

class ColorCV(enum.Enum):
    RED = mmcv.color_val("red")
    GREEN =  mmcv.color_val("green")
    BLUE = mmcv.color_val("blue")
    YELLOW = mmcv.color_val("yellow")
    CYAN = mmcv.color_val("cyan")
    MAGENTA = mmcv.color_val("magenta")
    WHITE = mmcv.color_val("white")
    BLACK = mmcv.color_val("black")

def cal_projection_mask(rgb_rendered:np.ndarray)->np.ndarray:
    mask_gray = cv2.cvtColor(rgb_rendered, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask_gray, 20, 255,cv2.THRESH_BINARY)

    return mask

def combine_img_and_masks(rgb_im:np.ndarray, mask_list:List[Tuple[np.ndarray,Color]])->np.ndarray:
    vis = np.copy(rgb_im)
    for mask, color in mask_list:
        vis[mask != 0] = np.array(color.value)
    return vis

def add_ren_mask_to_img(rgb_im:np.ndarray, rendered_rgb:np.ndarray, color:Color)->np.ndarray:
    mask = cal_projection_mask(rendered_rgb)
    mask_list = [(mask, color)]
    vis = combine_img_and_masks(rgb_im, mask_list)
    return vis

def vis_gt_and_pred(rgb_im:np.ndarray, gt_rgb:np.ndarray, pred_rgb:np.ndarray)->np.ndarray:
    vis = add_ren_mask_to_img(rgb_im, gt_rgb, ColorCV.RED)
    vis = add_ren_mask_to_img(vis, pred_rgb, ColorCV.GREEN)

    # gt_mask = cal_projection_mask(gt_rgb)
    # pred_mask = cal_projection_mask(pred_rgb)

    # mask_list = [(gt_mask, ColorCV.RED), (pred_mask, ColorCV.GREEN)]
    # vis = combine_img_and_masks(rgb_im, mask_list)

    # # Annotate using PIL
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    vis = annotate_img(vis)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    return vis

def annotate_img(img:np.ndarray)->np.ndarray:
    assert font_path.exists(), f"Font {font_path} does not exist."

    font = ImageFont.truetype(font=str(font_path), size=20)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    draw.text((10,10), "gt projection", font=font, fill=Color.RED.value)
    draw.text((10,40), "pred projection", font=font, fill=Color.GREEN.value)

    return np.asarray(img_pil)



def vis_gt_and_pred_option2(rgb_im:np.ndarray, gt_rgb:np.ndarray, pred_rgb:np.ndarray)->np.ndarray:
    ret, gt_mask = cv2.threshold(gt_rgb, 20, 255,cv2.THRESH_BINARY)
    gt_mask_single_ch = gt_mask[:,:,0]
    gt_mask[gt_mask_single_ch != 0] = np.array(mmcv.color_val("red"))

    ret, pred_mask = cv2.threshold(pred_rgb, 20, 255,cv2.THRESH_BINARY)
    pred_mask_single_ch = pred_mask[:,:,0]
    pred_mask[pred_mask_single_ch != 0] = np.array(mmcv.color_val("green"))

    vis_mask = cv2.addWeighted(gt_mask, 0.5, pred_mask, 0.5, 0) 
    vis = cv2.addWeighted(rgb_im, 0.3, vis_mask, 0.7, 0)

    return vis



if __name__ == "__main__":
    import gdrn_simple #Load paths to core libs.

    assert font_path.exists(), f"Font {font_path} does not exist."

    font = ImageFont.truetype(font=str(font_path), size=20)
    print(font.getsize("Hello world!"))

    img = np.zeros((480,640,3), dtype=np.uint8)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    draw.text((10,10), "gt projection", font=font, fill=Color.RED.value)
    draw.text((10,40), "pred projection", font=font, fill=Color.GREEN.value)

    img_pil.show()