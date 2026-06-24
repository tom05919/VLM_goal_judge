#!/usr/bin/env python3
"""Run semantic stop-judge inference with Qwen2.5-VL-3B-Instruct-AWQ."""

import json
import sys
import time
from pathlib import Path

# # Prefer this env's site-packages over PYTHONPATH entries from other conda envs.
# _QWEN_SITE = Path(sys.executable).resolve().parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
# sys.path[:] = [p for p in sys.path if "/envs/sim/" not in p]
# if _QWEN_SITE.is_dir():
#     _qwen_site_str = str(_QWEN_SITE)
#     sys.path = [p for p in sys.path if p != _qwen_site_str]
#     sys.path.insert(0, _qwen_site_str)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "omni-VLA/OmniVLA/inference"))

import rclpy
import torch
from isaacsim_controller import IsaacSimPublisher
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

DEFAULT_MODEL = SCRIPT_DIR / "models/Qwen2.5-VL-3B-Instruct-AWQ"
TASK_PROMPT = "move to the purple boxes"

SYSTEM_PROMPT_FILE = "system_prompt.md"
TASK_PROMPT_FILE = "task_specific_prompt.md"

# Keep vision tokens modest so inference fits on a ~12GB GPU alongside other jobs.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
MAX_NEW_TOKENS = 128
RUN_IMAGES_ROOT = SCRIPT_DIR / "run_images"


def load_prompt_file(filename: str) -> str:
    return (SCRIPT_DIR / filename).read_text(encoding="utf-8").strip()


def build_user_prompt(task_prompt: str, task_template: str) -> str:
    return task_template.format(task_prompt=task_prompt)


def create_run_image_dir() -> Path:
    """Create run_images/run{N} for this session (run0, run1, ...)."""
    RUN_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
    run_num = 0
    while (RUN_IMAGES_ROOT / f"run{run_num}").exists():
        run_num += 1
    run_dir = RUN_IMAGES_ROOT / f"run{run_num}"
    run_dir.mkdir()
    print(f"Saving VLM input images to {run_dir}")
    return run_dir


def save_current_image(image, run_dir: Path, index: int) -> Path | None:
    """Save the PIL image passed to the VLM as cur_image_[number].png."""
    if image is None:
        return None
    path = run_dir / f"cur_image_{index}.png"
    image.save(path)
    print(f"Saved VLM input image to {path}")
    return path


def build_messages(
    node: IsaacSimPublisher, system_text: str, user_text: str, run_dir: Path, image_index: int
) -> list:
    current_image_PIL = node.get_latest_image_pil()
    save_current_image(current_image_PIL, run_dir, image_index)
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": current_image_PIL},
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


def prepare_inputs(processor, messages: list, device):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)


def generate_and_decode(model, processor, inputs, max_new_tokens: int) -> str:
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

def parse_response(response: str) -> bool:
    #decode the response to a json object and return True if the decision is STOP and False otherwise
    json_response = json.loads(response)
    return json_response["decision"] == "STOP" and json_response["confidence"] >= 0.95

def main() -> None:
    rclpy.init()
    node = IsaacSimPublisher()

    system_text = load_prompt_file(SYSTEM_PROMPT_FILE)
    print(system_text)
    task_template = load_prompt_file(TASK_PROMPT_FILE)
    user_text = build_user_prompt(TASK_PROMPT, task_template)
    print(user_text)
    model, processor = load_model_and_processor(DEFAULT_MODEL)
    run_dir = create_run_image_dir()
    should_stop = False
    image_index = 0

    while not should_stop:
        messages = build_messages(node, system_text, user_text, run_dir, image_index)
        image_index += 1
        inputs = prepare_inputs(processor, messages, model.device)
        response = generate_and_decode(model, processor, inputs, MAX_NEW_TOKENS)
        print(response)
        should_stop = parse_response(response)
        time.sleep(1)

    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
