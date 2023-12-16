"""
!pip install -q gradio pillow accelerate python-Levenshtein
"""

from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import gradio as gr

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base", low_cpu_mem_usage=True, device_map="cuda")

def display_image(image):
    try:
      image = image.convert("RGB")

      pixel_values = processor(image, return_tensors="pt").pixel_values

      outputs = model.generate(
          pixel_values.to('cuda'),
          min_length=1,
          max_new_tokens=850,
          bad_words_ids=[[processor.tokenizer.unk_token_id]],
      )

      sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
      sequence = processor.post_process_generation(sequence, fix_markdown=False)

      content = sequence.replace(r'\(', '$').replace(r'\)', '$').replace(r'\[', '$$').replace(r'\]', '$$')
      return content

    except Exception as e:
      return str(e)

demo = gr.Interface(
    display_image,
    inputs=gr.Image(type="pil", sources="upload"),
    outputs=gr.Textbox (show_copy_button=True),
    title="Academic OCR",
    description="If the image was not recognized, please re-upload the image without any zoom.",
    allow_flagging='never'
)

if __name__ == "__main__":
    demo.launch()