# **PAPO: Perception-Aware Policy Optimization for Multimodal Reasoning**

This is the evalaution module for our work [Perception-Aware Policy Optimization for Multimodal Reasoning](https://mikewangwzhl.github.io/PAPO/)
- This module is also embedded into [PAPO](https://github.com/MikeWangWZHL/PAPO) for convenient inference and evaluation
- Feel free to directly use [PAPO](https://github.com/MikeWangWZHL/PAPO) for complete training-evaluation workflow!



# 🚀 **Evaluation for PAPO**

## **1. Env Setup**
We follow the environment setup instructions from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):
```bash
cd PAPO-Eval
pip install -e ".[torch,metrics]" --no-build-isolation
```

## **2. Data Preprocessing**

Prepare evaluation dataset for PAPO evaluation.

- Set the specific dataset(s) you would like to use for evaluation:
    - `AUTO_UNZIP` (bool): Whether to automatically upzip images
        - If set to `true`, the downloaded image ZIP file will be automatically unzipped, and the ZIP file will be removed
    - `SPLIT_NAME` (str): Which dataset to use for evalaution. Current available datasets:
        - *hiyouga/geometry3k*: `SPLIT_NAME="hiyouga_geometry3k"`
        - *AI4Math/MathVerse*: `SPLIT_NAME="AI4Math_MathVerse"`
        - *AI4Math/MathVista*: `SPLIT_NAME="AI4Math_MathVista"`
        - *We_Math/We_Math*: `SPLIT_NAME="We_Math"`
        - Vision-dependent subset of *AI4Math/MathVerse*: `SPLIT_NAME="AI4Math_MathVerse_vision_dependent"`
        - *BUAADreamer/clevr_count_70k*: `SPLIT_NAME="BUAADreamer_clevr_count_70k"`
        - *lscpku/LogicVista*: `SPLIT_NAME="lscpku_LogicVista"`
        - *MMMU/MMMU_Pro*: `SPLIT_NAME="MMMU_MMMU_Pro`

- Run data preprocessing
```bash
cd PAPO-Eval
bash papo_eval/preprocess/preprocess.sh
```

## **3. Run Evaluation**

### **3.1 Run Model Inference**
- Please set the dataset and other eval parameters in `PAPO-Eval/papo_eval/run_infer.sh`
    - `DATASET` (str): The dataset you would like to run inference on
        - *hiyouga/geometry3k*: `DATASET="hiyouga_geometry3k"`
        - *AI4Math/MathVerse*: `DATASET="AI4Math_MathVerse"`
        - *AI4Math/MathVista*: `DATASET="AI4Math_MathVista"`
        - *We_Math/We_Math*: `DATASET="We-Math_We-Math"`
        - Vision-dependent subset of *AI4Math/MathVerse*: `DATASET="AI4Math_MathVerse_vision_dependent"`
        - *BUAADreamer/clevr_count_70k*: `DATASET="BUAADreamer_clevr_count_70k"`
        - *lscpku/LogicVista*: `DATASET="lscpku_LogicVista"`
        - *MMMU/MMMU_Pro*: `DATASET="MMMU_MMMU_Pro"`
    - `Model` (str): PAPO model you would like to run inference
        - For example: `MODEL="PAPOGalaxy/PAPO-G-Qwen2.5-VL-7B"`
        - Our model collection on Hugging Face: [PAPO-Qwen](https://huggingface.co/collections/PAPOGalaxy/papo-qwen-686d92dd3d43b1ce698f851a)
            - PAPO-GRPO model collection: [PAPO-G](https://huggingface.co/collections/PAPOGalaxy/papo-g-688fd55ed6b49f343114ed6e)
            - PAPO-DAPO model collection: [PAPO-D](https://huggingface.co/collections/PAPOGalaxy/papo-d-688fd5917f3a2ffb715adcca)
    - `MODEL_VERSION` (str): Which version to use for inference
        - We saved both *best_val* and *last_step* for evaluation. Feel free to choose one!
        - Choices: `MODEL_VERSION="best_val"` or `MODEL_VERSION="last_step"`

- Run inference:
    ```bash
    cd PAPO-Eval
    bash papo_eval/run_infer.sh
    ```

- Inference outputs will be saved under `PAPO-Eval/infer_outputs`
    - The first and last output line will also show the exact save path

### **3.2 Run Evaluation On Model Inference**
- Please set the dataset and other eval parameters in `PAPO-Eval/papo_eval/run_eval.sh`
    - `JSONL_PATH` (str): Path to your to-be-eval inference results
        - JSONL path: Directly give the JSONL path if evaluate *accuracy* of a specific dataset inference results
        - Model dir: Give only model dir without JSONL path if evaluate vision-dependent accuracy
    - `N_ROLLOUT` (int): Number of rollout 
        - We set `N_ROLLOUT=8` in our [paper](https://arxiv.org/abs/2507.06448)

- Run evaluation:
    ```bash
    cd PAPO-Eval
    bash papo_eval/run_eval.sh
    ```

- Detailed results will be saved to `./eval_results/<eval_output_name>.json`
    - Results will also be printed out in the final section of the output, together with the exact save path of evaluation results



## 🥰 **Acknowledgement**
Huge thanks for providing this awesome codebase!
- We thank [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) team for providing this foundational codebase that we adapted to implement model inference and evaluation for PAPO.



## 📝 **Citation**
```bibtex
@article{wang2025perception,
  title={Perception-Aware Policy Optimization for Multimodal Reasoning},
  author={Wang, Zhenhailong and Guo, Xuehang and Stoica, Sofia and Xu, Haiyang and Wang, Hongru and Ha, Hyeonjeong and Chen, Xiusi and Chen, Yangyi and Yan, Ming and Huang, Fei and others},
  journal={arXiv preprint arXiv:2507.06448},
  year={2025}
}
```