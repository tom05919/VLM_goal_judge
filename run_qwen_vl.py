#!/usr/bin/env python3
"""Run semantic stop-judge inference with Qwen2.5-VL-3B-Instruct-AWQ."""

import json
import re
from pathlib import Path

from PIL import Image

# # Prefer this env's site-packages over PYTHONPATH entries from other conda envs.
# _QWEN_SITE = Path(sys.executable).resolve().parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
# sys.path[:] = [p for p in sys.path if "/envs/sim/" not in p]
# if _QWEN_SITE.is_dir():
#     _qwen_site_str = str(_QWEN_SITE)
#     sys.path = [p for p in sys.path if p != _qwen_site_str]
#     sys.path.insert(0, _qwen_site_str)

SCRIPT_DIR = Path(__file__).resolve().parent
# sys.path.insert(0, str(SCRIPT_DIR.parent / "omni-VLA/OmniVLA/inference"))

# import rclpy
import torch
# from isaacsim_controller import IsaacSimPublisher
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
OFFLINE_RUN_DIR = RUN_IMAGES_ROOT / "run6"


def load_prompt_file(filename: str) -> str:
    return (SCRIPT_DIR / filename).read_text(encoding="utf-8").strip()


def build_user_prompt(task_prompt: str, task_template: str) -> str:
    return task_template.format(task_prompt=task_prompt)


def load_offline_image_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("cur_image_*.png"), key=lambda p: int(p.stem.split("_")[-1]))


def history_indices(n: int) -> list[int]:
    return [i for i in (n - 4, n - 2, n) if i >= 0]


# def create_run_image_dir() -> Path:
#     """Create run_images/run{N} for this session (run0, run1, ...)."""
#     RUN_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
#     run_num = 0
#     while (RUN_IMAGES_ROOT / f"run{run_num}").exists():
#         run_num += 1
#     run_dir = RUN_IMAGES_ROOT / f"run{run_num}"
#     run_dir.mkdir()
#     print(f"Saving VLM input images to {run_dir}")
#     return run_dir
#
#
# def save_current_image(image, run_dir: Path, index: int) -> Path | None:
#     """Save the PIL image passed to the VLM as cur_image_[number].png."""
#     if image is None:
#         return None
#     path = run_dir / f"cur_image_{index}.png"
#     image.save(path)
#     print(f"Saved VLM input image to {path}")
#     return path
#
#
# def build_messages(
#     node: IsaacSimPublisher, system_text: str, user_text: str, run_dir: Path, image_index: int
# ) -> list:
#     frame_buffer[image_index] = node.get_latest_image_pil().copy()
#     save_current_image(frame_buffer[image_index], run_dir, image_index)
#     indices = history_indices(image_index)
#     images = [frame_buffer[i] for i in indices]
#     return build_messages(images, indices, system_text, user_text)


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
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    for candidate in (cleaned, response.strip()):
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    break
                except json.JSONDecodeError:
                    pass
    else:
        print(f"Warning: could not parse response, defaulting to CONTINUE:\n{response}")
        return False
    return parsed.get("decision") == "STOP" and parsed.get("confidence", 0) >= 0.95

def main() -> None:
    # rclpy.init()
    # node = IsaacSimPublisher()

    system_text = load_prompt_file(SYSTEM_PROMPT_FILE)
    print(system_text)
    task_template = load_prompt_file(TASK_PROMPT_FILE)
    user_text = build_user_prompt(TASK_PROMPT, task_template)
    print(user_text)
    model, processor = load_model_and_processor(DEFAULT_MODEL)
    # run_dir = create_run_image_dir()
    image_paths = load_offline_image_paths(OFFLINE_RUN_DIR)
    should_stop = False

    for n, image_path in enumerate(image_paths):
        if should_stop:
            break
        indices = history_indices(n)
        print(f"Processing {image_path.name} (history indices {indices})")
        images = []
        for i in indices:
            with Image.open(image_paths[i]) as img:
                images.append(img.copy())
        messages = build_messages(images, indices, system_text, user_text)
        inputs = prepare_inputs(processor, messages, model.device)
        response = generate_and_decode(model, processor, inputs, MAX_NEW_TOKENS)
        print(response)
        should_stop = parse_response(response)

    # frame_buffer: dict[int, Image.Image] = {}
    # image_index = 0
    # while not should_stop:
    #     frame_buffer[image_index] = node.get_latest_image_pil().copy()
    #     indices = history_indices(image_index)
    #     images = [frame_buffer[i] for i in indices]
    #     messages = build_messages(images, indices, system_text, user_text)
    #     inputs = prepare_inputs(processor, messages, model.device)
    #     response = generate_and_decode(model, processor, inputs, MAX_NEW_TOKENS)
    #     print(f"history indices {indices}")
    #     print(response)
    #     should_stop = parse_response(response)
    #     image_index += 1
    #     time.sleep(1)
    #
    # node.stop()
    # node.destroy_node()
    # rclpy.shutdown()


if __name__ == "__main__":
    main()
