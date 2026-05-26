from transformers import Qwen2_5_VLForConditionalGeneration,Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path
 

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/home/bsquitieri-iit.local/models/Qwen3-VL-32B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("/home/bsquitieri-iit.local/models/Qwen3-VL-32B-Instruct")
 
png = ['/home/bsquitieri-iit.local/panda_live_viewer/image.png']
 
for idx, image_path in enumerate(png):
    print(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },/home/bsquitieri-iit.local/panda_live_viewer/prova_queen.py
                {"type": "text", "text": "Do you see one orange glass , red glass and a blue one? If yes, answer only SUCCESS otherwise FAILED"},            
            ],
        }
    ]
 
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
 
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, do_sample= False, max_new_tokens=128) #, temperature=0.01)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
 