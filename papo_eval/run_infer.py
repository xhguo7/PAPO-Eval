# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
from typing import Optional

import fire
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

import os

def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    model_subfolder: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    batch_size: int = 1024,
    save_every: int = 1,
    loop_n: int = 1,
):
    if "/" in save_name:
        os.makedirs(os.path.dirname(save_name), exist_ok=True)

    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    # Determine the actual model path to use
    actual_model_path = model_name_or_path
    if model_subfolder:
        print(f"Using model subfolder: {model_subfolder}")
        actual_model_path = model_name_or_path  # Keep original for LlamaFactory
        vllm_model_path = f"{model_name_or_path}/{model_subfolder}"  # Use subfolder for vLLM
    else:
        vllm_model_path = model_name_or_path

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=actual_model_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    
    # Load tokenizer with subfolder support if needed
    if model_subfolder:
        # Import AutoTokenizer directly for subfolder support
        from transformers import AutoTokenizer, AutoProcessor
        try:
            print(f"Loading tokenizer from subfolder: {model_subfolder}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                subfolder=model_subfolder,
                trust_remote_code=True
            )
            
            # Also load the processor for vision-language models
            print(f"Loading processor from subfolder: {model_subfolder}")
            try:
                processor = AutoProcessor.from_pretrained(
                    model_name_or_path,
                    subfolder=model_subfolder,
                    trust_remote_code=True
                )
                print("Successfully loaded both tokenizer and processor with subfolder parameter")
            except Exception as proc_error:
                print(f"Could not load processor: {proc_error}")
                print("Trying to load processor without subfolder...")
                try:
                    processor = AutoProcessor.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True
                    )
                    print("Successfully loaded processor from main repo")
                except Exception as proc_error2:
                    print(f"Could not load processor at all: {proc_error2}")
                    processor = None
            
            # Create the tokenizer module dict that LlamaFactory expects
            tokenizer_module = {"tokenizer": tokenizer, "processor": processor}
            
        except Exception as e:
            print(f"Failed to load tokenizer with subfolder parameter: {e}")
            print("Falling back to standard tokenizer loading...")
            # Fallback to standard loading
            tokenizer_module = load_tokenizer(model_args)
    else:
        tokenizer_module = load_tokenizer(model_args)
    
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    # Set up vLLM engine arguments
    engine_args = {
        "model": vllm_model_path,  # Use the full path for vLLM
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 10, "video": 2, "audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    print(f"Initializing vLLM with model path: {vllm_model_path}")
    try:
        llm = LLM(**engine_args)
        print("Successfully initialized vLLM engine")
    except Exception as e:
        print(f"Error loading model: {e}")
        if model_subfolder:
            print("Trying alternative approach with download and local loading...")
            # Alternative: Try to download the model files locally first
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
                import tempfile
                import shutil
                
                # Create a temporary directory
                temp_dir = tempfile.mkdtemp()
                print(f"Downloading model to temporary directory: {temp_dir}")
                
                # List all files in the subfolder
                try:
                    repo_files = list_repo_files(repo_id=model_name_or_path)
                    subfolder_files = [f for f in repo_files if f.startswith(f"{model_subfolder}/")]
                    print(f"Found {len(subfolder_files)} files in subfolder {model_subfolder}")
                except Exception as list_error:
                    print(f"Could not list repo files: {list_error}")
                    # Fallback: try to download common model files including processor files
                    subfolder_files = [
                        f"{model_subfolder}/config.json",
                        f"{model_subfolder}/tokenizer.json",
                        f"{model_subfolder}/tokenizer_config.json",
                        f"{model_subfolder}/special_tokens_map.json",
                        f"{model_subfolder}/vocab.json",
                        f"{model_subfolder}/merges.txt",
                        f"{model_subfolder}/pytorch_model.bin",
                        f"{model_subfolder}/model.safetensors",
                        f"{model_subfolder}/generation_config.json",
                        f"{model_subfolder}/preprocessor_config.json",
                        f"{model_subfolder}/processor_config.json",
                        # Add more processor-related files for vision models
                        f"{model_subfolder}/image_processor_config.json",
                        # Add safetensors files (might be split)
                        f"{model_subfolder}/model-00001-of-00001.safetensors",
                        f"{model_subfolder}/model-00001-of-00002.safetensors",
                        f"{model_subfolder}/model-00002-of-00002.safetensors",
                        f"{model_subfolder}/model.safetensors.index.json",
                        # Add other potential files
                        f"{model_subfolder}/added_tokens.json",
                    ]
                
                # Download each file
                downloaded_files = []
                for file_path in subfolder_files:
                    try:
                        # Remove the subfolder prefix for the local filename
                        local_filename = file_path.replace(f"{model_subfolder}/", "")
                        local_file_path = os.path.join(temp_dir, local_filename)
                        
                        # Download the file
                        downloaded_file = hf_hub_download(
                            repo_id=model_name_or_path,
                            filename=file_path,
                            local_dir=temp_dir,
                            local_dir_use_symlinks=False
                        )
                        
                        # Move file to remove subfolder structure
                        if downloaded_file != local_file_path:
                            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                            shutil.move(downloaded_file, local_file_path)
                        
                        downloaded_files.append(local_filename)
                        print(f"Downloaded: {local_filename}")
                        
                    except Exception as file_error:
                        print(f"Could not download {file_path}: {file_error}")
                        continue
                
                if downloaded_files:
                    print(f"Successfully downloaded {len(downloaded_files)} files")
                    print(f"Model files available in: {temp_dir}")
                    
                    # Try to load processor from the downloaded directory
                    try:
                        from transformers import AutoProcessor
                        processor = AutoProcessor.from_pretrained(temp_dir, trust_remote_code=True)
                        print("Successfully loaded processor from downloaded directory")
                        # Update tokenizer_module to include the local processor
                        if "processor" not in tokenizer_module or tokenizer_module["processor"] is None:
                            tokenizer_module["processor"] = processor
                    except Exception as proc_error:
                        print(f"Could not load processor from downloaded directory: {proc_error}")
                    
                    # Update engine args to use local directory
                    engine_args["model"] = temp_dir
                    llm = LLM(**engine_args)
                    print("Successfully initialized vLLM with downloaded model")
                else:
                    raise Exception("No files were successfully downloaded")
                
            except Exception as e2:
                print(f"Alternative approach also failed: {e2}")
                raise e
        else:
            raise e

    # load datasets
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    # Load original dataset to get IDs
    original_dataset = None
    dataset_path = os.path.join(data_args.dataset_dir, f"{dataset}.json")
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            original_dataset = json.load(f)
        print(f"Loaded original dataset with {len(original_dataset)} samples")

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    # Store all results in these lists
    all_indices, all_test_ids, all_prompts, all_preds, all_labels = [], [], [], [], []

    print("Repeat the inference process for {} times.".format(loop_n))
    global_idx = 0
    for _ in range(loop_n):
        # Add batch process to avoid the issue of too many files opened
        for i in tqdm(range(0, len(train_dataset), batch_size), desc="Processing batched inference"):
            test_ids, vllm_inputs, prompts, labels = [], [], [], []
            batch = train_dataset[i : min(i + batch_size, len(train_dataset))]

            for j in range(len(batch["input_ids"])):
                if batch["images"][j] is not None:
                    image = batch["images"][j]
                    multi_modal_data = {
                        "image": template_obj.mm_plugin._regularize_images(
                            image, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
                        )["images"]
                    }
                elif batch["videos"][j] is not None:
                    video = batch["videos"][j]
                    multi_modal_data = {
                        "video": template_obj.mm_plugin._regularize_videos(
                            video,
                            image_max_pixels=image_max_pixels,
                            image_min_pixels=image_min_pixels,
                            video_fps=video_fps,
                            video_maxlen=video_maxlen,
                        )["videos"]
                    }
                elif batch["audios"][j] is not None:
                    audio = batch["audios"][j]
                    audio_data = template_obj.mm_plugin._regularize_audios(
                        audio,
                        sampling_rate=16000,
                    )
                    multi_modal_data = {"audio": zip(audio_data["audios"], audio_data["sampling_rates"])}
                else:
                    multi_modal_data = None

                vllm_inputs.append({"prompt_token_ids": batch["input_ids"][j], "multi_modal_data": multi_modal_data})
                prompts.append(tokenizer.decode(batch["input_ids"][j], skip_special_tokens=skip_special_tokens))
                labels.append(
                    tokenizer.decode(
                        list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                        skip_special_tokens=skip_special_tokens,
                    )
                )
                
                # Get ID from original dataset
                current_data_idx = i + j
                if original_dataset and current_data_idx < len(original_dataset):
                    test_id = original_dataset[current_data_idx].get('id', f"test_{current_data_idx}")
                else:
                    test_id = f"test_{current_data_idx}"
                if test_id == "not specified":
                    test_id = f"test_{current_data_idx}"
                
                test_ids.append(test_id)

            results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
            preds = [result.outputs[0].text for result in results]

            # Accumulate results
            batch_indices = list(range(global_idx, global_idx + len(prompts)))
            global_idx += len(prompts)
            all_indices.extend(batch_indices)
            all_test_ids.extend(test_ids)
            all_prompts.extend(prompts)
            all_preds.extend(preds)
            all_labels.extend(labels)
            gc.collect()

            if (i + 1) % save_every == 0:
                # save intermeidate
                with open(save_name, "w", encoding="utf-8") as f:
                    for idx, test_id, text, pred, label in zip(all_indices, all_test_ids, all_prompts, all_preds, all_labels):
                        f.write(json.dumps({"idx": idx, "id": test_id, "prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    # Write all results at once outside the loop
    print("length of all results:", len(all_prompts))
    print(f"Expected: {len(train_dataset)} samples × {loop_n} rollouts = {len(train_dataset) * loop_n}")
    with open(save_name, "w", encoding="utf-8") as f:
        for idx, test_id, text, pred, label in zip(all_indices, all_test_ids, all_prompts, all_preds, all_labels):
            f.write(json.dumps({"idx":idx, "id": test_id, "prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(all_prompts)} total generated results have been saved at {save_name}.")
    print(f"Each of the {len(train_dataset)} test samples has {loop_n} predictions.")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(vllm_infer)