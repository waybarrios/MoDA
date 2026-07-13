"""
Convert LLaVA v1.5 mix665k dataset format to LlamaFactory ShareGPT format.

LLaVA format:
{
  "id": "000000033471",
  "image": "coco/train2017/000000033471.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat are the colors..."},
    {"from": "gpt", "value": "The bus is white and red."}
  ]
}

LlamaFactory ShareGPT format:
{
  "messages": [
    {"role": "user", "content": "<image>What are the colors..."},
    {"role": "assistant", "content": "The bus is white and red."}
  ],
  "images": ["coco/train2017/000000033471.jpg"]
}
"""

import json
import argparse
from pathlib import Path


ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
}


def convert_entry(entry: dict, image_base_dir: str) -> dict:
    """Convert a single LLaVA entry to LlamaFactory format."""
    messages = []
    for turn in entry["conversations"]:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        content = turn["value"]
        messages.append({"role": role, "content": content})

    result = {"messages": messages}

    # Handle image field
    if "image" in entry and entry["image"]:
        image_path = entry["image"]
        if image_base_dir:
            image_path = str(Path(image_base_dir) / image_path)
        result["images"] = [image_path]

    return result


def convert_dataset(input_path: str, output_path: str, image_base_dir: str = ""):
    """Convert entire dataset from LLaVA to LlamaFactory format."""
    print(f"Loading {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)

    print(f"Converting {len(data)} entries...")
    converted = []
    skipped = 0
    for entry in data:
        try:
            converted.append(convert_entry(entry, image_base_dir))
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Skipped entry {entry.get('id', '?')}: {e}")

    print(f"Converted: {len(converted)}, Skipped: {skipped}")

    with open(output_path, "w") as f:
        json.dump(converted, f, ensure_ascii=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LLaVA dataset to LlamaFactory format")
    parser.add_argument("--input", required=True, help="Input LLaVA JSON file")
    parser.add_argument("--output", required=True, help="Output LlamaFactory JSON file")
    parser.add_argument("--image_base_dir", default="", help="Base directory for image paths")
    args = parser.parse_args()

    convert_dataset(args.input, args.output, args.image_base_dir)
