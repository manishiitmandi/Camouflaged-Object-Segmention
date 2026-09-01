"""Qwen model inference and prompt templates/JSON parser utilities for DSS."""

import re
import json
import numpy as np
from collections import Counter
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from dss.utils.image import numpy_to_base64


def process_vision_info(messages):
    """
    Extract image inputs from QWen API-like message dictionaries.
    """
    image_inputs = []
    video_inputs = []
    for message in messages:
        if "content" in message:
            for content_item in message["content"]:
                if content_item.get("type") == "image":
                    img_path = content_item["image"]
                    if img_path.startswith("data:image/png;base64,"):
                        # Base64 encoded string
                        import base64
                        import io
                        base64_data = img_path.split(",")[1]
                        img_data = base64.b64decode(base64_data)
                        image_inputs.append(Image.open(io.BytesIO(img_data)).convert("RGB"))
                    else:
                        image_inputs.append(Image.open(img_path).convert("RGB"))
                elif content_item.get("type") == "video":
                    video_inputs.append(content_item["video"])
    return image_inputs, video_inputs


def clean_result_field(result_str):
    """
    Clean the JSON output of Qwen from markdown enclosures like ```json...```.
    """
    result_str = result_str.strip()
    result_str = re.sub(r"^```json|^```|```$", "", result_str, flags=re.MULTILINE).strip()
    try:
        return json.loads(result_str)
    except Exception:
        return result_str


def parse_json_from_markdown(markdown_string):
    """
    Extract and parse JSON content from a markdown-formatted string.
    """
    try:
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, markdown_string, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            return json.loads(json_content)
        else:
            return json.loads(markdown_string.strip())
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


def clean_and_parse_json(text):
    """
    Exhaustively attempt parsing markdown string into a valid list/dict.
    """
    if isinstance(text, list):
        text = ' '.join(text)
    
    try:
        cleaned_text = text.strip()
        result = json.loads(cleaned_text)
        return result
    except json.JSONDecodeError:
        pass
    
    try:
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1)
            result = json.loads(json_str)
            return result
    except (json.JSONDecodeError, AttributeError):
        pass
    
    try:
        array_pattern = r'\[.*\]'
        match = re.search(array_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(0)
            result = json.loads(json_str)
            return result
    except (json.JSONDecodeError, AttributeError):
        pass
    
    raise ValueError("Could not parse valid JSON from text")


def query_qwen_for_selection(image, mask1, mask2, QWen_model, QWen_processor, device):
    """
    Query Qwen Model with pair of candidate masks to choose the superior overlay.
    """
    def encode_img(img):
        return numpy_to_base64(img)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        img64, mask1_b64, mask2_b64 = executor.map(encode_img, [image, mask1, mask2])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The image is this."},
                    {"type": "image", "image": f"data:image/png;base64,{img64}"},
                    {"type": "text", "text": "The MASK A is this."},
                    {"type": "image", "image": f"data:image/png;base64,{mask1_b64}"},
                    {"type": "text", "text": "The MASK B is this."},
                    {"type": "image", "image": f"data:image/png;base64,{mask2_b64}"},
                    {"type": "text", "text": f"""
                    CAMOUFLAGE MASK COMPARISON TASK
                    IMAGE: The image may contain a few animal/insect or human whose shape, color, texture, pattern and movement closely resemble its surroundings.
                    MASK A: Current best mask
                    MASK B: New candidate mask
 
                    MASK INTERPRETATION:
                    - White areas = potential camouflaged objects
                    - Black areas = background  
 
                    CRITICAL REQUIREMENT: The chosen mask MUST match your object analysis.
 
                    STEP-BY-STEP PROCESS:
 
                    1. OBJECT IDENTIFICATION:
                    - Carefully examine the image
                    - Identify all hidden/concealed objects and their exact locations
 
                    2. MASK EVALUATION:
                    Mask A: Does the white region cover all identified objects? How much extra background?
                    Mask B: Does the white region cover all identified objects? How much extra background?
 
                    3. SELECTION CRITERIA:
                    - PRIMARY: Choose the mask that covers ALL identified objects completely
                    - SECONDARY: Among masks that meet primary criterion, choose the one with least background
                    - If no mask covers all objects, choose the one that covers the most objects
 
                    4. CONSISTENCY CHECK:
                    - Ensure the chosen mask's white regions align with your identified objects
                    - If analysis says "2 camouflaged insects", the mask should have 2 corresponding white regions
 
                    OUTPUT JSON (DO NOT ADD ANY EXTRA INFO, JUST JSON!):
                    [{{
                        "better_mask": "Mask A" / "Mask B", 
                    }}]
                    """},
                ],
            } 
        ]
    
    text = QWen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = QWen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    import torch
    with torch.no_grad():
        generated_ids = QWen_model.generate(
            **inputs, 
            max_new_tokens=64, 
            output_scores=True, 
            do_sample=True,
            temperature=0.5, 
            return_dict_in_generate=True
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
    ]
    
    output_text = QWen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    try:
        data = clean_and_parse_json(output_text)
        if isinstance(data, list):
            data = data[0]
        selected_mask = data["better_mask"]
        return selected_mask
    except Exception:
        return "Mask X"


def pairwise_selection_from_QWen(image, mask_list, score_list, QWen_model, QWen_processor, device="cuda:0"):
    """
    Run iterative pairwise comparison selection over a list of candidate masks using QWen.
    """
    mask_score_pairs = list(zip(mask_list, score_list))
    mask_score_pairs.sort(key=lambda x: x[1])
    
    if len(mask_score_pairs) == 1:
        return mask_score_pairs[0][0]
    
    current_list = mask_score_pairs.copy()
    
    while len(current_list) > 1:
        mask1, score1 = current_list[0]
        mask2, score2 = current_list[1]
        remaining_masks = current_list[2:]
        
        masked_img1 = mask1[:, :, np.newaxis] * image
        masked_img2 = mask2[:, :, np.newaxis] * image
        
        votes = []
        for i in range(1):
            selected_mask_name = query_qwen_for_selection(image, masked_img1, masked_img2, QWen_model, QWen_processor, device)
            votes.append(selected_mask_name.lower())
        
        vote_count = Counter(votes)
        selected_mask_name = vote_count.most_common(1)[0][0]

        if "mask a" in selected_mask_name:
            winner = (mask1, score1)
        elif "mask b" in selected_mask_name:
            winner = (mask2, score2)
        else:
            winner = (mask2, score2)
            
        new_list = [winner] + remaining_masks
        new_list.sort(key=lambda x: x[1])
        current_list = new_list.copy()
    
    return current_list[0][0]


def get_pred_from_QWen(img_mask_grid, new_masks, QWen_model, QWen_processor, device="cuda:1"):
    """
    Identify the best mask choice directly from a grid-stitched collage.
    """
    mask_b64 = numpy_to_base64(img_mask_grid)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/png;base64,{mask_b64}"},
                {"type": "text", "text": f"""CAMOUFLAGE MASK SELECTION TASK
                    IMAGE: Contains camouflaged objects that blend with surroundings.
                    MASKS: Candidate masks with bounding boxes.
 
                    MASK INTERPRETATION:
                    - White areas = potential hidden/concealed objects
                    - Black areas = background  
                    - Boxes = object location hints
 
                    CRITICAL REQUIREMENT: The chosen mask MUST match your object analysis.
 
                    STEP-BY-STEP PROCESS:
 
                    1. OBJECT IDENTIFICATION:
                    - Carefully examine the image
                    - Identify all hidden/concealed/camouflaged objects and their exact locations
                    
                    2. MASK EVALUATION (for each mask):
                    Mask X: Does the white region cover all identified objects? How much extra background?
                    Mask Y: Does the white region cover all identified objects? How much extra background?
 
                    3. SELECTION CRITERIA:
                    - PRIMARY: Choose the mask that covers ALL identified objects completely
                    - SECONDARY: Among masks that meet primary criterion, choose the one with least background
                    - If no mask covers all objects, choose the one that covers the most objects
 
                    4. CONSISTENCY CHECK:
                    - Ensure the chosen mask's white regions align with your identified objects
                    - If analysis says "2 camouflaged insects", the mask should have 2 corresponding white regions
 
                    OUTPUT JSON (DO NOT ADD ANY EXTRA INFO, JUST JSON!):
                    [{{
                        "best_mask": "Mask X" [Mask X should be one of the candidate masks], 
                    }}]
                """}
            ]
        }
    ]
    text = QWen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = QWen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    import torch
    with torch.no_grad():
        generated_ids = QWen_model.generate(
            **inputs, 
            max_new_tokens=512, 
            output_scores=True, 
            do_sample=True,
            temperature=0.2, 
            return_dict_in_generate=True
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
    ]
    output_text = QWen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    try:
        data = clean_and_parse_json(output_text)
        if isinstance(data, list):
            data = data[0]
    except Exception:
        data = clean_and_parse_json(output_text)
        
    try:
        selected_mask = data["best_mask"].split(' ')[1]
    except Exception:
        selected_mask = None
        
    return selected_mask
