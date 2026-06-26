#!/usr/bin/env python3
"""Run semantic stop-judge inference with MiniCPM-V-4.6-AWQ."""

import json
import re
from pathlib import Path

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils.quantization_config import AwqConfig, AwqBackend

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = SCRIPT_DIR / "models/MiniCPM-V-4.6-AWQ"
TASK_PROMPT = "move to the door"

SYSTEM_PROMPT_FILE = "system_prompt.md"
TASK_PROMPT_FILE = "task_specific_prompt.md"

DOWNSAMPLE_MODE = "16x"
MAX_SLICE_NUMS = 36
MAX_NEW_TOKENS = 128
IMMEDIATE_STOP_CONFIDENCE = 0.95
CONSECUTIVE_STOP_CONFIDENCE = 0.9
CONSECUTIVE_STOPS_REQUIRED = 4
RUN_IMAGES_ROOT = SCRIPT_DIR / "run_images"
OFFLINE_RUN_DIR = RUN_IMAGES_ROOT / "run3_no-stop"


def history_indices(n: int) -> list[int]:
    return [i for i in (n - 4, n - 2, n) if i >= 0]


def frame_label(frame_index: int, frame_indices: list[int]) -> str:
    if len(frame_indices) == 1:
        return f"Current camera frame (step {frame_index}):"
    if frame_index == frame_indices[-1]:
        return f"Newest camera frame (step {frame_index}):"
    if frame_index == frame_indices[0]:
        return f"Oldest camera frame in history (step {frame_index}):"
    return f"Earlier camera frame (step {frame_index}):"


def build_messages(
    images: list[Image.Image], frame_indices: list[int], system_text: str, user_text: str
) -> list:
    content = []
    for frame_index, img in zip(frame_indices, images):
        content.append({"type": "text", "text": frame_label(frame_index, frame_indices)})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content},
    ]


def load_model_and_processor(model_path: Path):
    print(f"Loading model from {model_path}")
    awq_config = AwqConfig(
        bits=4,
        group_size=128,
        zero_point=True,
        backend=AwqBackend.TORCH_AWQ,
        modules_to_not_convert=[
            "in_proj_b",
            "in_proj_a",
            "model.vision_tower",
            "model.merger",
            "lm_head",
        ],
    )
    processor = AutoProcessor.from_pretrained(str(model_path))
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        quantization_config=awq_config,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model, processor


def prepare_inputs(processor, messages: list, device, downsample_mode: str):
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        downsample_mode=downsample_mode,
        max_slice_nums=MAX_SLICE_NUMS,
    ).to(device)


def generate_and_decode(
    model, processor, inputs, downsample_mode: str, max_new_tokens: int
) -> str:
    with __import__("torch").inference_mode():
        output_ids = model.generate(
            **inputs,
            downsample_mode=downsample_mode,
            max_new_tokens=max_new_tokens,
        )

    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def parse_response(response: str) -> dict:
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    fixed = re.sub(r'"\s*\n\s*"decision"', '",\n"decision"', cleaned)
    for candidate in (cleaned, fixed, response.strip()):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if match:
                snippet = re.sub(r'"\s*\n\s*"decision"', '",\n"decision"', match.group(0))
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
    print(f"Warning: could not parse response, defaulting to CONTINUE:\n{response}")
    return {
        "reason": "parse failure",
        "decision": "CONTINUE",
        "confidence": 0.0,
        "target_match": "unclear",
        "proximity": "unclear",
    }


def main() -> None:
    system_text = (SCRIPT_DIR / SYSTEM_PROMPT_FILE).read_text(encoding="utf-8").strip()
    print(system_text)
    task_template = (SCRIPT_DIR / TASK_PROMPT_FILE).read_text(encoding="utf-8").strip()
    user_text = task_template.format(task_prompt=TASK_PROMPT)
    print(user_text)
    model, processor = load_model_and_processor(DEFAULT_MODEL)
    image_paths = sorted(
        (p for p in OFFLINE_RUN_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda p: int(re.search(r"\d+", p.name).group()),
    )
    should_stop = False
    consecutive_stop_votes = 0

    for n, image_path in enumerate(image_paths):
        if should_stop:
            break
        indices = history_indices(n)
        print(f"Processing {image_path.name} (history indices {indices})")
        images = []
        for i in indices:
            with Image.open(image_paths[i]) as img:
                images.append(img.convert("RGB").copy())
        messages = build_messages(images, indices, system_text, user_text)
        inputs = prepare_inputs(processor, messages, model.device, DOWNSAMPLE_MODE)
        response = generate_and_decode(
            model, processor, inputs, DOWNSAMPLE_MODE, MAX_NEW_TOKENS
        )
        print(response)
        parsed = parse_response(response)
        decision = parsed.get("decision")
        confidence = parsed.get("confidence", 0)

        if decision == "STOP" and 1 >= confidence >= IMMEDIATE_STOP_CONFIDENCE:
            should_stop = True
            print(f"Final: STOP (confidence {confidence:.2f} >= {IMMEDIATE_STOP_CONFIDENCE})")
        elif decision == "STOP" and confidence >= CONSECUTIVE_STOP_CONFIDENCE:
            consecutive_stop_votes += 1
            if consecutive_stop_votes >= CONSECUTIVE_STOPS_REQUIRED:
                should_stop = True
                print(f"Final: STOP ({CONSECUTIVE_STOPS_REQUIRED} consecutive votes)")
            else:
                print(f"STOP vote {consecutive_stop_votes}/{CONSECUTIVE_STOPS_REQUIRED}")
        else:
            consecutive_stop_votes = 0
            print("Final: CONTINUE")


if __name__ == "__main__":
    main()
