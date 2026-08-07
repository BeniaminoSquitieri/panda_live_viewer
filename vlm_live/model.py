"""Model loading and VLM inference helpers."""

import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .camera import cleanup, save_image
from .prompt import log_response, parse_verdict


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


def _is_cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def run_text_inference(scene, prompt: str, processor, model, device, tokens: int, log_out: bool, logger) -> str:
    """Run the multimodal model on the current scene and return raw text."""
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

        generated_ids = None
        retry_tokens = min(tokens, 256)
        token_attempts = [tokens] if retry_tokens == tokens else [tokens, retry_tokens]
        for attempt_index, attempt_tokens in enumerate(token_attempts, start=1):
            try:
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=attempt_tokens,
                    )
                break
            except RuntimeError as exc:
                if not _is_cuda_oom(exc):
                    raise
                if attempt_index >= len(token_attempts):
                    raise
                logger.warning(
                    f"CUDA OOM during VLM generation with max_new_tokens={attempt_tokens}; "
                    f"clearing cache and retrying with max_new_tokens={token_attempts[attempt_index]}."
                )
                del inputs
                _cleanup_cuda_memory()
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)
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
            log_response(output_text, logger)

        return output_text
    finally:
        for name in ("generated_ids_trimmed", "generated_ids", "inputs", "image_inputs", "video_inputs"):
            if name in locals():
                del locals()[name]
        _cleanup_cuda_memory()
        cleanup(scene_path)


def run_inference(scene, prompt: str, processor, model, device, tokens: int, reasoning: bool, log_out: bool, logger) -> tuple[str, str]:
    """Run the multimodal model on the current scene and return a verdict."""
    generation_limit = tokens if reasoning else min(tokens, 4)
    output_text = run_text_inference(
        scene=scene,
        prompt=prompt,
        processor=processor,
        model=model,
        device=device,
        tokens=generation_limit,
        log_out=log_out,
        logger=logger,
    )

    return parse_verdict(output_text)
