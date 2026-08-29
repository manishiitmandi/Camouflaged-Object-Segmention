"""QWen parallel inference runner pipeline for DSS."""

import os
import re
import json
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from multiprocessing import Process
import torch.multiprocessing as mp
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from dss.qwen.qwen_utils import process_vision_info, clean_result_field


def infer_worker(rank, gpu_id, image_paths, dataset_name, output_dir, model_dir):
    """
    Worker process running inference on a chunk of images using Qwen2.5-VL.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    results = []
    start = time.time()
    
    for img_path in tqdm(image_paths, desc=f"[Rank {rank}, GPU {gpu_id}] Processing", position=rank, leave=True):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": """The image may contain a few animal/insect or human whose shape, color, texture, pattern and movement 
                            closely resemble its surroundings. Please identify them and provide their locations in the format of coordinates, as precisely 
                            as possible. Also provide me with the confidence of your localisation result. The output should be in JSON format, eg: 
                            "{"
                               "bbox_2d": [[x1, y1, x2, y2],[x1, y1, x2, y2]],
                               "label": "dog",
                               "confidence": "confidence value in between [0,1]"
                            "}". 
                            If you can not locate the camouflaged/concealed/hidden objects, the format of output should be in JSON format, eg:
                            "{"
                                "description": "suggest possible camouflaged/concealed/hidden ainmals/insects/human"
                            }. DO NOT ADD ANY EXTRA INFO, JUST JSON!"""}
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        with torch.no_grad():
            try:
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        results.append({
            "image": img_path,
            "result": output_text[0] if output_text else ""
        })
        torch.cuda.empty_cache()

    out_path = os.path.join(output_dir, f"infer_{dataset_name}_QWen_7B_{rank}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    end = time.time()
    print(f"[GPU {gpu_id}] Inference time: {end - start:.2f} seconds")
    print(f"[GPU {gpu_id}] Results saved to {out_path}")


def split_list(lst, n):
    """Split a list into n approximately equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


def run_qwen_inference(image_dir, model_dir, output_dir, dataset="COD10K", gpus="0", seed=42):
    """
    Launch parallelized QWen inference processes.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    os.makedirs(output_dir, exist_ok=True)
    gpu_list = gpus.split(',')

    image_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_list.sort()
    
    if "COD10K" in dataset:
        image_list = image_list[:2026]

    chunks = split_list(image_list, len(gpu_list))

    processes = []
    for rank, (gpu_id, chunk) in enumerate(zip(gpu_list, chunks)):
        if not chunk:
            continue
        p = Process(target=infer_worker, args=(rank, gpu_id, chunk, dataset, output_dir, model_dir))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    # Merge results from rank files
    all_results = []
    for rank in range(len(gpu_list)):
        part_file = os.path.join(output_dir, f"infer_{dataset}_QWen_7B_{rank}.json")
        if os.path.exists(part_file):
            with open(part_file, "r", encoding="utf-8") as f:
                all_results.extend(json.load(f))
            os.remove(part_file)

    merged_path = os.path.join(output_dir, f"infer_{dataset}_QWen_7B.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"All results merged to {merged_path}")

    # Clean JSON fields
    output_path = os.path.join(output_dir, f"infer_{dataset}_QWen_7B_clean.json")
    for item in all_results:
        if "result" in item and isinstance(item["result"], str):
            item["result"] = clean_result_field(item["result"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    os.remove(merged_path)
    print(f"Cleaned results saved to: {output_path}")
