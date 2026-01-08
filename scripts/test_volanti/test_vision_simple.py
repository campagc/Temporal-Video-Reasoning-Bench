import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# --- CONFIGURAZIONE ---
# Metti qui il percorso di UN frame che hai controllato esistere
IMAGE_PATH = "data/processed_frames/step_1642/frame_0.jpg" 
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

def main():
    print(f"Test visione su: {IMAGE_PATH}")
    
    # Carica modello
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # Prompt semplicissimo
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PATH},
                {"type": "text", "text": "Describe precisely what is happening in this image. What objects do you see?"}
            ]
        }
    ]

    # Inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    
    print(f"\n--- DESCRIZIONE ---")
    print(output_text)

if __name__ == "__main__":
    main()