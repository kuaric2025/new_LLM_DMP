from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image


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


def build_system_prompt(width: int, height: int, actions: Sequence[str], action_id_mapping: dict[str, str]) -> str:
    return (
        "You are a robotic assistant for long-horizon grasping tasks. "
        f"Given a task and an image with original size of ({height}, {width}), generate a step-by-step action plan in JSON format.\n"
        f"There are several actions that the robot can perform, including: {actions}, when generating the plan, you should only use the actions that are listed here. \n"
        "Please use common, everyday object names (only one word) that fit naturally into the context.\n"
        "Actions are divided into two types: \n"
        "1) Manipulation actions: reach, move_to.\n"
        "   - These actions require a target_object with name and optionally attributes.\n"
        "2) Motion / task actions: grasp, release, pour, wiping\n"
        "   - For 'grasp', include target_object with name and optionally 'part' field.\n"
        "   - For 'pour', include both source_object (with ref_id after first grounding) and target_object.\n"
        "   - For 'release', include target_object.\n"
        "   - For 'wiping', should first grasp the sponge for wiping task, then do wiping action.\n"
        "One complete and independent pick and place task should follow the action: reach to -> grasp -> reach to -> release.\n"
        "Additional planning constraints: \n"
        "- The plan MUST include interaction with the task goal object.\n"
        "- For transfer actions such as 'pour', the robot MUST move to the target object before performing the action.\n"
        "- After an object is first identified, use 'ref_id' to reference it in subsequent steps (e.g., 'obj_kettle_1').\n"
        "- The 'ref_id' format should be 'obj_{name}_{number}' (e.g., 'obj_kettle_1', 'obj_mug_1').\n"
        "2D location inference:\n"
        "- Infer the 2D pixel location (x, y) of each target_object center from the image.\n"
        "- Return the center of that target in absolute pixel coordinates.\n"
        "- Pixel origin is the top-left corner of the image.\n"
        "- Give coordinates [xmin,ymin,xmax,ymax] of the bounding box of the object in the image.\n"
        # f"- Coordinates: y = vertical (0 = top, {height} = bottom),x = horizontal (0 = left, {width} = right).\n"
        "- Provide [y=(ymin+ymax)/2, x=(xmin+xmax)/2] as the approximate center point of the object in the image.\n"
        "- Include 'location' for target_object in reach, grasp, move_to, pour (target), etc.\n"
        "- For source_object in pour, use ref_id only (no location needed).\n"
        f"Action ID mapping (use these exact action_id values): {json.dumps(action_id_mapping)}\n"
        "Output format: Return ONLY valid JSON array with the following structure:\n"
        "[\n"
        "  {\n"
        "    \"step_id\": 1,\n"
        "    \"action\": \"reach to\",\n"
        "    \"action_id\": \"7\",\n"
        "    \"target_object\": {\n"
        "      \"name\": \"kettle\",\n"
        "      \"location\": [320, 400],\n"
        "      \"bbox\": [300, 380, 340, 420],\n"
        "      \"attributes\": \"red\",\n"
        "      \"ref_id\": null\n"
        "    }\n"
        "  },\n"
        "  {\n"
        "    \"step_id\": 2,\n"
        "    \"action\": \"grasp\",\n"
        "    \"action_id\": \"9\",\n"
        "    \"target_object\": {\n"
        "      \"name\": \"kettle\",\n"
        "      \"part\": \"handle\",\n"
        "      \"location\": [320, 400],\n"
        "      \"bbox\": [300, 380, 340, 420],\n"
        "      \"attributes\": \"red\",\n"
        "      \"ref_id\": \"obj_kettle_1\"\n"
        "    }\n"
        "  },\n"
        "  {\n"
        "    \"step_id\": 3,\n"
        "    \"action\": \"pour\",\n"
        "    \"action_id\": \"5\",\n"
        "    \"source_object\": {\"ref_id\": \"obj_kettle_1\"},\n"
        "    \"target_object\": {\n"
        "      \"name\": \"mug\",\n"
        "      \"location\": [480, 350],\n"
        "      \"bbox\": [460, 330, 500, 370],\n"
        "      \"attributes\": \"blue\"\n"
        "    }\n"
        "  }\n"
        "]\n"
        "Important:\n"
        "- Return ONLY the JSON array, no additional text or explanations.\n"
        "- Ensure all JSON is valid and properly formatted.\n"
        "- Include 'action_id' in every step, using the mapping above.\n"
        "- Include 'location' as [x, y] in target_object when the action requires a spatial target. Infer from the image.\n"
        "- Use 'ref_id' to reference objects after they are first identified.\n"
        "- For 'attributes', use color name strings.\n"
        "- Set 'ref_id' to null for first occurrence of an object.\n"
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
        pil_format = "PNG"
        mime = "image/png"
    elif format.lower() in {"jpg", "jpeg"}:
        pil_format = "JPEG"
        mime = "image/jpeg"
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Avoid cv2.imencode here: in the full demo process OpenCV's native image
    # encoder can segfault after PyBullet/GLX and ROS threads are active.
    image = Image.fromarray(np.ascontiguousarray(rgb_array))
    encoded = BytesIO()
    image.save(encoded, format=pil_format)
    return f"data:{mime};base64," + base64.b64encode(encoded.getvalue()).decode("utf-8")


def _openai_responses_create_raw(payload: dict) -> dict:
    """Call Responses API without SDK model parsing.

    The OpenAI SDK can hit a pydantic schema rebuild bug in this environment
    ("dict_items object is not callable"). Raw HTTP avoids that parser path.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI Responses API failed ({exc.code}): {body}") from exc


def _extract_response_text(response_data: dict) -> str | None:
    if response_data.get("output_text"):
        return response_data["output_text"]

    text_parts: list[str] = []
    for item in response_data.get("output", []):
        for block in item.get("content", []):
            if block.get("type") in {"output_text", "text"} and block.get("text"):
                text_parts.append(block["text"])
    return "\n".join(text_parts) if text_parts else None


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

    response_data = _openai_responses_create_raw(
        {
            "model": "gpt-4o",
            "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {"role": "user", "content": content},
            ],
            "temperature": 0.2,
        }
    )

    response_text = _extract_response_text(response_data)
    
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
        parsed_json = json.loads(response_text.strip())
        if not isinstance(parsed_json, list):
            raise ValueError(f"Expected JSON array, got {type(parsed_json)}")
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response: {e}\nResponse text: {response_text[:500]}")
