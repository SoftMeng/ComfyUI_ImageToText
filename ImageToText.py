from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

import numpy as np

model_id = "vikhyatk/moondream2"
revision = "2024-04-02"

class ComfyUI_ImageToText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "log_prompt": (["No", "Yes"], {"default":"Yes"}),
            },
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('text_positive',)
    FUNCTION = "image2text"
    OUTPUT_NODE = True
    CATEGORY = "ComfyUI_Mexx"

    def image2text(self, images, log_prompt):
        pil_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)
        image = pil_images[0]
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        enc_image = model.encode_image(image)
        en = model.answer_question(enc_image, "Describe this image.", tokenizer)
        if log_prompt == "Yes":
            print(f"ImageToText: {en}")
        return [en]

NODE_CLASS_MAPPINGS = {
    "ComfyUI_ImageToText": ComfyUI_ImageToText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_ImageToText": "ComfyUI_ImageToText"
}
