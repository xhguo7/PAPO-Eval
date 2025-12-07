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
    model_subfolder: str = "last_step",
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

    # Load model
    actual_model_path = model_name_or_path
    vllm_model_path = model_name_or_path
    using_subfolder = False

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
    
    # Load tokenizer
    tokenizer_module = None
    tokenizer_load_error = None
    tokenizer_using_subfolder = False
    
    try:
        print(f"Attempting to load tokenizer from root: {model_name_or_path}")
        tokenizer_module = load_tokenizer(model_args)
        print("Successfully loaded tokenizer from root directory")
    except Exception as e:
        tokenizer_load_error = e
        print(f"Failed to load tokenizer from root: {e}")
        
        if model_subfolder:
            print(f"Attempting to load tokenizer from subfolder: {model_subfolder}")
            from transformers import AutoTokenizer, AutoProcessor
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    subfolder=model_subfolder,
                    trust_remote_code=True
                )
                print(f"Successfully loaded tokenizer from subfolder: {model_subfolder}")
                
                # Try to load processor
                processor = None
                try:
                    processor = AutoProcessor.from_pretrained(
                        model_name_or_path,
                        subfolder=model_subfolder,
                        trust_remote_code=True
                    )
                    print(f"Successfully loaded processor from subfolder: {model_subfolder}")
                except Exception as proc_error:
                    print(f"Could not load processor from subfolder: {proc_error}")
                    try:
                        processor = AutoProcessor.from_pretrained(
                            model_name_or_path,
                            trust_remote_code=True
                        )
                        print("Successfully loaded processor from root")
                    except Exception:
                        print("Could not load processor")
                
                tokenizer_module = {"tokenizer": tokenizer, "processor": processor}
                tokenizer_using_subfolder = True
                
            except Exception as subfolder_error:
                print(f"Failed to load tokenizer from subfolder: {subfolder_error}")
                raise Exception(f"Could not load tokenizer from root or subfolder: {tokenizer_load_error}")
        else:
            raise Exception(f"Could not load tokenizer and no subfolder specified: {tokenizer_load_error}")
    
    if tokenizer_module is None:
        raise Exception("Failed to load tokenizer from any source")
    
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    # Set up vLLM engine arguments
    engine_args = {
        "model": vllm_model_path,
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
    llm = None
    vllm_load_error = None
    
    try:
        llm = LLM(**engine_args)
        print("Successfully initialized vLLM engine from root directory")
    except Exception as e:
        vllm_load_error = e
        print(f"Error loading model from root {vllm_model_path}: {e}")
        
        # If loading from root failed and we have a subfolder specified, try downloading it
        if model_subfolder:
            print(f"Root loading failed. Attempting to download and load from subfolder: {model_subfolder}")
            
            # vLLM doesn't support HF Hub subfolder syntax, so we need to download it
            try:
                from huggingface_hub import snapshot_download
                import tempfile
                
                temp_dir = tempfile.mkdtemp()
                print(f"Downloading model subfolder to temporary directory: {temp_dir}")
                
                # Download all files from the subfolder
                downloaded_path = snapshot_download(
                    repo_id=model_name_or_path,
                    allow_patterns=f"{model_subfolder}/*",
                    local_dir=temp_dir,
                    local_dir_use_symlinks=False
                )
                
                # The downloaded files will be in temp_dir/model_subfolder/
                local_model_path = os.path.join(temp_dir, model_subfolder)
                
                if os.path.exists(local_model_path):
                    print(f"Successfully downloaded model to: {local_model_path}")
                    
                    # Verify that config.json exists
                    config_path = os.path.join(local_model_path, "config.json")
                    if not os.path.exists(config_path):
                        raise Exception(f"config.json not found in {local_model_path}")
                    
                    print(f"Found config.json at: {config_path}")
                    
                    # Try to load processor from the downloaded directory if we don't have one
                    if tokenizer_module.get("processor") is None:
                        try:
                            from transformers import AutoProcessor
                            processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
                            print("Successfully loaded processor from downloaded directory")
                            tokenizer_module["processor"] = processor
                        except Exception as proc_error:
                            print(f"Could not load processor from downloaded directory: {proc_error}")
                    
                    # Update engine args to use local directory
                    engine_args["model"] = local_model_path
                    vllm_model_path = local_model_path
                    
                    try:
                        llm = LLM(**engine_args)
                        print("Successfully initialized vLLM with downloaded model from subfolder")
                        using_subfolder = True
                    except Exception as local_load_error:
                        print(f"Failed to load from downloaded directory: {local_load_error}")
                        raise Exception(f"Could not load model even after downloading. Error: {local_load_error}")
                else:
                    raise Exception(f"Downloaded path does not exist: {local_model_path}")
                    
            except Exception as download_error:
                print(f"Subfolder download and load failed: {download_error}")
                raise Exception(f"All attempts to load model failed. Root error: {vllm_load_error}, Subfolder download error: {download_error}")
        else:
            raise Exception(f"Could not load model from {vllm_model_path} and no subfolder specified: {vllm_load_error}")
    
    if llm is None:
        raise Exception("Failed to initialize vLLM engine")

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