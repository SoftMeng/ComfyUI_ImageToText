import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# 这个地方放图片文件夹的路径
base = './image/'

# 这里是模型地址，如果切换模型，可以改动这里
model_id = "vikhyatk/moondream2"
revision = "2024-04-02"


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if not f.startswith('.') and not f.endswith('.txt'):
                fullname = os.path.join(root, f)
                yield fullname


def main():
    print(f"加载模型: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    for imagefile in findAllFile(base):
        print(f"处理图片: {imagefile}")
        image = Image.open(imagefile)
        enc_image = model.encode_image(image)
        en = model.answer_question(enc_image, "Describe this image.", tokenizer)
        file_name, file_extension = os.path.splitext(imagefile)
        print(f"{file_name} 自然语言Tag: {en}")
        with open(file_name + ".txt", 'w', encoding='utf-8') as file:
            # 向文件中写入内容
            file.write(en)
            file.write('\n')

if __name__ == '__main__':
    main()
