import torch, os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch.nn.functional as F
from PIL import Image
# Load BiRefNet with weights
from transformers import AutoModelForImageSegmentation

from torchvision.transforms.functional import normalize
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
# folder_paths.folder_names_and_paths["BiRefNet"] = ([os.path.join(folder_paths.models_dir, "BiRefNet")], folder_paths.supported_pt_extensions)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


class BiRefNet_ModelLoader_Innary:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device_type": (["cuda","cpu"], {}),
            },
        }

    RETURN_TYPES = ("BRNMODEL",)
    RETURN_NAMES = ("birefnetmodel",)
    FUNCTION = "load_model"
    CATEGORY = "🧩BiRefNet"
  
    def load_model(self, device_type):
        device = "cuda" if torch.cuda.is_available() and device_type == "cuda" else "cpu"
        print(f"Using device: {device}")
        birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
        birefnet.to(device)
        return [birefnet]



class BiRefNet_Innary:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "birefnetmodel": ("BRNMODEL",),
                "image": ("IMAGE",),
                "device_type": (["cuda","cpu"], {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "remove_background"
    CATEGORY = "🧩BiRefNet"
  
    def remove_background(self, birefnetmodel, image, device_type):
        processed_images = []
        processed_masks = []

        for image in image:
            orig_image = tensor2pil(image)
            w,h = orig_image.size
            image = resize_image(orig_image)
            im_np = np.array(image)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
            im_tensor = torch.unsqueeze(im_tensor,0)
            im_tensor = torch.divide(im_tensor,255.0)
            im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
            if torch.cuda.is_available() and device_type == "cuda":
                im_tensor=im_tensor.cuda()

            result = birefnetmodel(im_tensor)[-1].sigmoid()
            #print(result.shape)
            
            result = torch.squeeze(F.interpolate(result, size=(h,w), mode='bilinear') ,0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)    
            im_array = (result*255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            new_im = Image.new("RGBA", pil_im.size, (0,0,0,0))
            new_im.paste(orig_image, mask=pil_im)

            new_im_tensor = pil2tensor(new_im)  # 将PIL图像转换为Tensor
            pil_im_tensor = pil2tensor(pil_im)  # 同上

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        return new_ims, new_masks



NODE_CLASS_MAPPINGS = {
    "BiRefNet_ModelLoader_Innary": BiRefNet_ModelLoader_Innary,
    "BiRefNet_Innary": BiRefNet_Innary,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNet_ModelLoader_Innary": "BiRefNet Model Loader(Innary Ver.)",
    "BiRefNet_Innary": "BiRefNet(Innary Ver.)",
}
