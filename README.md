# IL3D: A Large-scale indoor layout dataset for LLM-Driven 3D Scene Generation

## 🏠 [Project Page](https://wenxuzhou.github.io/project/IL3D/) | [ArXiv](https://arxiv.org/abs/2510.12095) | [Code](https://github.com/WenxuZhou/IL3D) | [Data](https://huggingface.co/datasets/WenxuZhou/IL3D)

In this study, we present IL3D, a large-scale dataset meticulously designed for large language model (LLM)-driven 3D scene generation, addressing the pressing demand for diverse, high-quality training data in indoor layout design. Comprising 27,816 indoor layouts across 18 prevalent room types and a library of 29,215 high-fidelity 3D object assets, IL3D is enriched with instance-level natural language annotations to support robust multimodal learning for vision-language tasks. We establish rigorous benchmarks to evaluate LLM-driven scene generation. Experimental results show that supervised fine-tuning (SFT) of LLMs on IL3D significantly improves generalization and surpasses the performance of SFT on other datasets. IL3D offers flexible multimodal data export capabilities, including point clouds, 3D bounding boxes, multiview images, depth maps, normal maps, and semantic masks, enabling seamless adaptation to various visual tasks. As a versatile and robust resource, IL3D significantly advances research in 3D scene generation and embodied intelligence, by providing high-fidelity scene data to support environment perception tasks of embodied agents. 

## 1. Requirements
CUDA 11.8, python 3.11

```bash
conda create -n IL3D python=3.11
conda activate IL3D
pip install -r requirements.txt
pip install ms-swift -U
```

## 2. Preparations
Download the [dataset](./data/dataset.md) and [checkpoints](./ckpts/checkpoints.md) to the specified directory，then build the vector database for retrieval，with the following structure:

```
|-- ckpts
    |-- Qwen3-1.7B
    |-- Qwen3-4B
    |-- Qwen3-8B
    |-- Qwen3-14B
    |-- clip-vit-base-patch32
|-- data
    |-- 3D-FRONT
    |-- HSSD
    |-- layout
    |-- room
    |-- qdrant
    |-- text_emb
    |-- assets.json
    |-- labels.json
```

## 3. Supervised Fine-Tuning (SFT)
Construct the dataset required for SFT, run:
```bash
python sft_dataset.py
```

We train the model based on the [Swift](https://github.com/modelscope/ms-swift) framework, run:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> \
swift sft \
    --model <path_to_Qwen3_checkpoint> \
    --ddp_find_unused_parameters true \
    --train_type lora \
    --dataset <path_to_sft_dataset> \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate <sft_learning_rate> \
    --lora_rank <lora_rank> \
    --lora_alpha <lora_alpha> \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir <path_to_output_dir> \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
```


## 4. Scene Generation
Enter a scene description to generate a JSON-formatted retrival results, run:
```bash
CUDA_VISIBLE_DEVICES=<GPU> \
python llm_retrieve.py \
    --model_path <path_to_qwen_checkpoint> \
    --res_dir <path_to_scene_res>
```

Generate USDA format 3D scene based on retrival results, run:
```bash
CUDA_VISIBLE_DEVICES=<GPU> \
python llm_design.py \
    --model_path <path_to_qwen_checkpoint> \
    --lora_checkpoint <path_to_lora_checkpoint> \
    --retrieval <path_to_retrieval_results> \
    --res_dir <path_to_scene_res>
```

## 5. Visualization
Sample point clouds and corresponding bounding boxes from the scene, run:

```bash
python scripts/sample_scan_point.py \
    --input_folder <path_to_usda_scene> \
    --output_dir <path_to_sample_res> \
    --num_points <num_of_points>
```

We provide rendering scripts for scene images, depth, normals, and semantic masks based on the physical simulation engine [Orca3d](http://www.orca3d.cn/).
For details, please see the [relevant instructions](./render/README.md).

## Reference

```
@article{zhou2025il3d,
  title={IL3D: A Large-Scale Indoor Layout Dataset for LLM-Driven 3D Scene Generation},
  author={Zhou, Wenxu and Nie, Kaixuan and Du, Hang and Yin, Dong and Huang, Wei and Guo, Siqiang and Zhang, Xiaobo and Hu, Pengbo},
  journal={arXiv preprint arXiv:2510.12095},
  year={2025}
}
```
