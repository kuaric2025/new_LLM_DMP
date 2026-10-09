from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from openai import OpenAI


ACTION_ALIASES = {
    "reach": "reach to",
    "move_to": "move to",
}


def load_llm_config(config_path: str | Path | None = None) -> tuple[dict, dict[str, str], str]:
    """Load the LLM configuration + build the system prompt.

    Returns (config, action_id_mapping, system_prompt).
    """
    path = config_path or os.environ.get("LLM_CONFIG_PATH", "config.json")
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    width = config["width"]
    height = config["height"]
    actions: Sequence[str] = config["actions"]
    action_id_mapping = config.get("action_id_mapping")
    if action_id_mapping is None:
        actions_list = config.get("actions_list", [str(i + 1) for i in range(len(actions))])
        action_id_mapping = dict(zip(actions, actions_list))

    system_prompt = build_system_prompt(width, height, actions, action_id_mapping)
    return config, action_id_mapping, system_prompt


def build_system_prompt(
    _width: int | None = None,
    _height: int | None = None,
    _actions: Sequence[str] | None = None,
    _action_id_mapping: dict[str, str] | None = None,
) -> str:
    """Symbolic task prompt. Optional config args match `load_llm_config` / may be used later for templating."""
    return (
        # "You are the high-level task planner for a mobile manipulator robot operating in a structured indoor workspace."
        # "Your job is to generate safe, clear, and executable task plans for transferring luggages item between zones.\n"
        # "Workspace definition:\n"
        # "- Zone A: Home position of the robot.\n"
        # "- Zone B: Luggage transfer zone, also the drop-off zone.\n"
        # "- Zone C: Luggage transfer zone, also the pick-up zone.\n"
        # "- The robot starts in Zone A.\n"
        # "- A luggage item may be located in Zone B or Zone C.\n"
        # "- The task is to move one luggage item from Zone C to Zone B, or from Zone B to Zone C.\n"
        # "- if there are multiple luggage items in the same zone, you should transfer them one by one."
        # "- if there are no luggage items in the zone, report that no transfer is needed."
        # "- if it's task to move multiple luggage items, after the 1st transfer, instead of returning to Zone A, stay in the same zone and start the next transfer."

        # "Operational rules:\n"
        # "- Always begin from Zone A.\n"
        # "- First navigate to the source zone containing the luggage.\n"
        # "- Pick the luggage safely using the manipulator.\n"
        # "- Navigate to the target zone.\n"
        # "- Place the luggage fully inside the target zone.\n"
        # "- After completing the transfer, return the robot to Zone A unless explicitly instructed otherwise.\n"
        # "- Only one luggage item is handled at a time, if multiple luggages are required to be transfered, should transfer them one by one.\n"
        # "- Never confuse source and target zones.\n"
        # "- Do not invent zones, objects, or actions not defined in the instruction.\n"
        # "- If the source zone and target zone are the same, report that no transfer is needed.\n"
        # "- If the instruction does not clearly specify source and target, ask for clarification.\n"

        # "- the attributes of the luggage item should be the color of the luggage item."
        # "Output format: Return ONLY valid JSON array with the following structure:\n"
        # "[\n"
        # "  {\n"
        # "    \"step_id\": 1,\n"
        # "    \"action\": \"navigate to\",\n"
        # "    \"target_object\": {\n"
        # "      \"name\": \"luggage\",\n"
        # "      \"part\": \"handle\",\n"
        # "      \"attributes\": \"red\",\n"
        # "      \"ref_id\": obj_luggage_1\n"
        # "    }\n"
        # "    \"source zone\": \"A\",\n"
        # "    \"target zone\": \"B\",\n"
        # "  },\n"
        # "  {\n"
        # "    \"step_id\": 2,\n"
        # "    \"action\": \"pick\",\n"
        # "    \"target_object\": {\n"
        # "      \"name\": \"luggage\",\n"
        # "      \"part\": \"handle\",\n"
        # "      \"attributes\": \"blue\",\n"
        # "      \"ref_id\": \"obj_luggage_2\"\n"
        # "    }\n"
        # "    \"source zone\": \"A\",\n"
        # "    \"target zone\": \"B\",\n"
        # "  },\n"
        # "  {\n"
        # "    \"step_id\": 3,\n"
        # "    \"action\": \"pour\",\n"
        # "    \"action_id\": \"5\",\n"
        # "    \"source_object\": {\"ref_id\": \"obj_kettle_1\"},\n"
        # "    \"target_object\": {\n"
        # "      \"name\": \"mug\",\n"
        # "      \"part\": \"handle\",\n"
        # "      \"attributes\": \"blue\"\n"
        # "    }\n"
        # "    \"source zone\": \"A\",\n"
        # "    \"target zone\": \"B\",\n"
        # "  }\n"
        # "]\n"
        # "Important:\n"
        # "- Return ONLY the JSON array, no additional text or explanations.\n"
        # "- Ensure all JSON is valid and properly formatted.\n"
        "You are a symbolic task interpreter for a robotic system."
        "Your job is to convert a user instruction into a SMALL list of high-level symbolic actions that represent physical robot intent.\n"

        "You may use the provided image and the user instruction together to resolve object references.\n"
        "When the user does not name an object exactly, identify it by visible attributes such as color, size, tag, relative position, or other clearly distinguishable appearance cues.\n"

        "RULES:\n"
        "- Each action MUST contain a concrete object name or location.\n"
        "- Actions MUST be physical, intention-level robot actions.\n"
        "- Actions must be atomic (one clear intent per action).\n"
        "- Do NOT describe how the action is performed.\n"
        "- Do NOT mention tools, sensors, perception, vision, detection, segmentation, reasoning, or algorithms.\n"
        "- Do NOT include explanations.\n"  
        "- Maximum 6 actions.\n"    
        "- Do NOT drop or merge actions just to stay short. If the user lists N intents, the plan MUST contain all N.\n"

        "GROUNDING RULES:\n"
        "- If multiple objects of the same category are present, distinguish them using visible attributes.\n"
        "- Prefer concise grounded names such as:\n"
        "- blue suitcase\n"
        "- black suitcase\n"
        "- large blue suitcase with heavy tag \n"
        "- If the instruction refers to a subset, infer membership from visible evidence in the image.\n"
        '- Terms like "overweighted", "heavy", "tagged", "large", "small", "left", "right", "front", "back", and colors should be grounded to the visible objects when possible.\n'
        '- If one object is clearly marked with a heavy/overweight tag, treat that object as the excluded item for instructions like "except the overweighted one".\n'
        "- Only ask for clarification if the referenced objects cannot be distinguished from the image and instruction.\n"

        "GUIDANCE:\n"
        "- Read the user instruction and organize it into an ordered sequence of actions.\n"
        "- Split the instruction only by intent and execution order.\n"

        "IMPORTANT:\n"
        "- Each symbolic action MUST involve only ONE active object.\n"
        "- If an instruction involves a source and a target, split it into separate actions so that only the moved object appears.\n"
        "- If the task involves navigation (go to, move to, navigate to), express it as 'Navigate to <zone/location>'.\n"
        "- If the task is only about identifying or localizing an object, do NOT add navigation unless the user explicitly asks for it.\n"
        "- Never add navigation before object identification unless the user explicitly asks to go somewhere first.\n"

        "OUTPUT FORMAT:\n"
        "- Return only the ordered symbolic actions, one per line.\n"
        "- Do not return any extra commentary.\n"
    )


def apply_action_ids(plan: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Add action_id to each LLM step based on the mapping (handles aliases)."""
    for step in plan:
        action_name = step.get("action", "").strip().lower()
        if not action_name:
            step["action_id"] = None
            continue
        canonical = ACTION_ALIASES.get(action_name, action_name)
        action_id = mapping.get(canonical)
        if action_id is None:
            for key, value in mapping.items():
                if key.lower() == canonical or canonical in key.lower():
                    action_id = value
                    break
        step["action_id"] = action_id
    return plan


def encode_image_file(path: Path) -> str:
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported image type for {path}")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("utf-8")


def encode_image_array(rgb_array: np.ndarray, *, format: str = "png") -> str:
    if rgb_array.dtype != np.uint8 or rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected uint8 RGB image, got shape={rgb_array.shape}, dtype={rgb_array.dtype}")
    if format.lower() == "png":
        success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
        mime = "image/png"
    elif format.lower() in {"jpg", "jpeg"}:
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
        mime = "image/jpeg"
    else:
        raise ValueError(f"Unsupported format: {format}")
    if not success:
        raise ValueError("Failed to encode image")
    return f"data:{mime};base64," + base64.b64encode(encoded.tobytes()).decode("utf-8")


def ask_gpt4o(prompt: str, system_prompt: str, image_paths: Optional[list[Path]] = None, image_arrays: Optional[list[np.ndarray]] = None) -> list[dict]:
    """Send text + images to GPT-4o and return the parsed JSON response as a list of dictionaries.
    
    Args:
        prompt: text prompt
        image_paths: list of Path objects pointing to image files (optional)
        image_arrays: list of numpy RGB arrays (H, W, 3) uint8 (optional)
    
    Returns:
        List of dictionaries representing the action plan steps
    
    At least one of image_paths or image_arrays must be provided if images are needed.
    """
    client = OpenAI()
    content = []
    if prompt.strip():
        content.append({"type": "input_text", "text": prompt.strip()})
    
    # Process file paths
    if image_paths:
        for image_path in image_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": encode_image_file(image_path),
                }
            )
    
    # Process numpy arrays
    if image_arrays:
        for rgb_array in image_arrays:
            content.append(
                {
                    "type": "input_image",
                    "image_url": encode_image_array(rgb_array),
                }
            )

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {"role": "user", "content": content},
        ],
        temperature=0,      
        )

    # Prefer output_text (available on Responses API); otherwise assemble text chunks.
    response_text = None
    if getattr(response, "output_text", None):
        response_text = response.output_text
    else:
        text_parts: list[str] = []
        for item in getattr(response, "output", []):
            for block in getattr(item, "content", []):
                if getattr(block, "type", "") in {"output_text", "text"}:
                    if getattr(block, "text", None):
                        text_parts.append(block.text)
        response_text = "\n".join(text_parts)
    
    # Parse JSON from response
    if response_text is None:
        raise ValueError("No response text received from GPT-4o")
    
    # Try to extract JSON from markdown code blocks if present
    import re
    # First try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1)
    else:
        # Try to find JSON array by looking for balanced brackets
        # Find the first '[' and then find the matching ']'
        start_idx = response_text.find('[')
        if start_idx != -1:
            bracket_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(response_text)):
                if response_text[i] == '[':
                    bracket_count += 1
                elif response_text[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
            if bracket_count == 0:
                response_text = response_text[start_idx:end_idx]
    
    # Parse JSON
    try:
        # import pdb; pdb.set_trace()
        parsed_json = json.loads(response_text.strip())
        if not isinstance(parsed_json, list):
            raise ValueError(f"Expected JSON array, got {type(parsed_json)}")
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response: {e}\nResponse text: {response_text[:500]}")


def main():
    # image_path = Path("/home/mhumais/Downloads/layout.png")
    # image_path = Path("/home/mhumais/Downloads/heavy.JPG")
    image_path = Path("/home/mhumais/Downloads/heavy.png")

    # prompt = "transfter all luggage from its zone to another transfer zone"
    # prompt = "move all luggage execpt the heavy one from the pick-up zone to the drop-off zone"
    prompt = "move the overweighted luggage from the pick-up zone to the drop-off zone"
    # prompt = "move the overweighted luggage to the drop-off zone"
    # prompt = "grasp the overweighted luggage"


    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    system_prompt = build_system_prompt()
    plan = ask_gpt4o(prompt, system_prompt, image_paths=[image_path])
    # plan = apply_action_ids(plan, action_id_mapping)
    # print(plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
