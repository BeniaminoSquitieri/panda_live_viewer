"""Model loading and VLM inference helpers."""

from typing import Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .camera import cleanup, save_image
from .prompt import extract_reason, parse_status


def load_model(model_path: str):
    """Load the processor and causal vision-language model from disk."""
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return processor, model, next(model.parameters()).device


def run_inference(scene, prompt: str, processor, model, device, tokens: int, reasoning: bool, log_out: bool, logger) -> Tuple[str, str]:
    """Run the multimodal model on the current scene and return a verdict."""
    scene_path = save_image(scene)
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": scene_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.inference_mode():
            generation_limit = tokens
            if not reasoning:
                generation_limit = min(generation_limit, 4)
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=generation_limit,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        if log_out:
            logger.info(f"VLM output: {output_text}")

        status = parse_status(output_text)
        reason = extract_reason(output_text)
        if not output_text:
            reason = "Empty model output."
        return status, reason
    finally:
        cleanup(scene_path)
