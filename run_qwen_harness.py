#!/usr/bin/env python3
"""Two-stage skeptical agent harness for Qwen2.5-VL semantic stop judging."""

import json
import re
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = SCRIPT_DIR / "models/Qwen2.5-VL-3B-Instruct-AWQ"
TASK_PROMPT = "move to the purple boxes"

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
MAX_NEW_TOKENS = 128
RUN_IMAGES_ROOT = SCRIPT_DIR / "run_images"
OFFLINE_RUN_DIR = RUN_IMAGES_ROOT / "run6"


def load_prompt_file(filename: str) -> str:
    return (SCRIPT_DIR / filename).read_text(encoding="utf-8").strip()


def load_offline_image_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("cur_image_*.png"), key=lambda p: int(p.stem.split("_")[-1]))


def build_messages(image: Image.Image, system_text: str, user_text: str) -> list:
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def load_model_and_processor(model_path: Path):
    print(f"Loading model from {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "5GiB", "cpu": "48GiB"},
        low_cpu_mem_usage=True,
    )
    model.tie_weights()
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return model, processor


def generate(model, processor, messages: list) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def parse_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    for candidate in (cleaned, text.strip()):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
    print(f"Warning: could not parse response, defaulting to CONTINUE:\n{text}")
    return {
        "decision": "CONTINUE",
        "confidence": 0.0,
        "target_match": "unclear",
        "proximity": "unclear",
        "reason": "parse failure",
    }


def main() -> None:
    system_text = load_prompt_file("system_prompt.md")
    user_text = load_prompt_file("task_specific_prompt.md").format(task_prompt=TASK_PROMPT)
    skeptic_system = load_prompt_file("harness_skeptic_system.md")
    skeptic_user_template = load_prompt_file("harness_skeptic_user.md")

    model, processor = load_model_and_processor(DEFAULT_MODEL)
    should_stop = False

    for image_path in load_offline_image_paths(OFFLINE_RUN_DIR):
        if should_stop:
            break
        print(f"Processing {image_path.name}")
        with Image.open(image_path) as image:
            image = image.copy()

            draft = generate(model, processor, build_messages(image, system_text, user_text))
            print("--- Stage 1 (draft) ---")
            print(draft)

            skeptic_user = skeptic_user_template.format(
                task_prompt=TASK_PROMPT, draft_response=draft.strip()
            )
            final = generate(model, processor, build_messages(image, skeptic_system, skeptic_user))
            print("--- Stage 2 (skeptic) ---")
            print(final)

            structured = parse_response(final)
            should_stop = (
                structured.get("decision") == "STOP" and structured.get("confidence", 0) >= 0.9
            )


if __name__ == "__main__":
    main() 
