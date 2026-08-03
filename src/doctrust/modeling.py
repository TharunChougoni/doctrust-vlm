from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)


class DocumentVLM:
    """Small adapter around Hugging Face image-to-text chat models."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_id = str(config["model_id"])
        dtype_name = str(config.get("dtype", "float16"))
        if dtype_name != "float16":
            raise ValueError("The local RTX 3060/T4 path is configured only for float16")
        compute_dtype = torch.float16

        quantization_config = None
        if bool(config.get("load_in_4bit", True)):
            # Granite's SigLIP tower contains attention output projections that
            # access raw weights directly. Quantizing those weights packs them
            # as uint8 and causes `Half and Byte` matrix multiplication errors.
            # Keep the vision path in FP16 and quantize only the language model.
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
            )

        common_kwargs: dict[str, Any] = {
            "dtype": compute_dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if quantization_config is not None:
            common_kwargs["quantization_config"] = quantization_config

        architecture = str(config["architecture"])
        model_class = {
            "vision2seq": AutoModelForVision2Seq,
            "image_text_to_text": AutoModelForImageTextToText,
        }.get(architecture)
        if model_class is None:
            raise ValueError(f"Unsupported model architecture: {architecture}")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = model_class.from_pretrained(self.model_id, **common_kwargs).eval()

    def answer(self, image_path: str | Path, question: str) -> str:
        """Generate a short answer for one image/question pair."""
        prompt = f"{self.config['prompt']}\n\nQuestion: {question}"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(Path(image_path).resolve())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(self.config.get("max_new_tokens", 64)),
            )
        prompt_length = inputs["input_ids"].shape[-1]
        answer_tokens = generated[:, prompt_length:]
        text = self.processor.batch_decode(answer_tokens, skip_special_tokens=True)[0]
        return text.strip()
