from attentionHook import AttentionHook
from config import mapsSize, attnHooksResolutions, maskRefinement, DEVICE, LDMPath
from diffusers import StableDiffusionImg2ImgPipeline
import segmentation_refinement as refine
from transformers import CLIPProcessor, CLIPModel
import torch

# prepare CLIP
CLIP_path = "/home/corradini/.models/clip-vit-base-patch32"
CLIP_model = CLIPModel.from_pretrained(CLIP_path).to(DEVICE)
CLIP_processor = CLIPProcessor.from_pretrained(CLIP_path)
# prepare SDM
torch.set_grad_enabled(False)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(LDMPath, revision="fp16", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to(DEVICE)
attn_hook = AttentionHook(pipe.unet, mapsSize=mapsSize, resolutions=attnHooksResolutions)

refiner = None
if maskRefinement == "CRF":
    from crf import CRF
elif maskRefinement == "PSP":
    refiner = refine.Refiner(device=DEVICE)