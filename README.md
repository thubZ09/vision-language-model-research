<div align='center'>

# Multimodal Research Hub - Vision-Language Models (VLMs)
<strong>A living resource for Vision-Language Models & multimodal learning  
(papers, models, datasets, benchmarks, tools, ethical challenges, research directions)</strong>

[![Last Updated](https://img.shields.io/badge/Updated-January%202026-brightpurple)](https://github.com/your-username/awesome-vision-language-models/commits/main)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/your-username/awesome-vision-language-models/blob/main/LICENSE)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-yellow)](https://github.com/your-username/awesome-vision-language-models/blob/main/LICENSE)

</div>

## Table of contents

- [Seminal models (Post-2021)](#seminal-models)
  - [2025](#sm-2025)
  - [2024](#sm-2024)
  - [2023](#sm-2023)
  - [2022 & Prior](#sm-2022-prior)
- [Datasets](#datasets)
  - [Core Training Datasets](#core-training-datasets)
  - [Image Classification](#image-classification)
  - [Object Detection](#object-detection)
  - [Semantic Segmentation](#semantic-segmentation)
  - [Action Recognition](#action-recognition)
  - [Image-Text Retrieval](#image-text-retrieval)
  - [Visual Question Answering (VQA)](#visual-question-answering-vqa)
  - [Instruction Tuning](#instruction-tuning)
  - [Bias](#bias)
  - [Video Understanding](#video-understanding)
  - [Additional Datasets](#additional-datasets)
- [Benchmarks](#benchmarks)
- [Research Directions](#research-directions)
- [Ethical Challenges](#ethical-challenges)
- [Privacy Protection Framework](#privacy-protection-framework)
- [Tools](#tools)
- [Emerging Applications](#emerging-applications)


## 🔗 Seminal models (Post-2021)
<!-- Seminal models -->
<a id="seminal-models"></a>

### 2025
<a id="sm-2025"></a>

- **MiniCPM-o 2.6** - [An 8B parameter multimodal model capable of understanding and generating content across vision, speech, and language modalities](https://huggingface.co/openbmb/MiniCPM-o-2_6)
- **Janus-Pro-7B** - [autoregressive framework that unifies multimodal understanding and generation](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
- **Gemma3-4b-it** - [One of the smallest multimodal models with a 128k token context window, supporting 140+ languages](https://huggingface.co/google/gemma-3-4b-it)
- **Qwen2.5-VL-3B-Instruct** - [capable of handling various tasks ranging from localization to document understanding and agentic tasks, with a context length of up to 32k tokens](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- **GLM-4.6V** - [GLM-4.6V series model includes two versions: GLM-4.6V (106B), a foundation model designed for cloud and high-performance cluster scenarios, and GLM-4.6V-Flash (9B), a lightweight model optimized for local deployment and low-latency applications](https://huggingface.co/zai-org/GLM-4.6V)
- **MiniCPM-V** - [On-device friendly (<6 GB RAM), scores 70% on MMBench, supports 30+ languages & 1.8 M-pixel images](https://github.com/OpenBMB/MiniCPM-V.git)
- **CogVLM2-Video** - [Temporal-aware, based on CogVLM2, adds 3D RoPE for long video clips, reaches 63% on Video-MME](https://cogvlm2-video.github.io/)
- **Ovis** - [Decoder-only, native dynamic tiling (AnyRes++), 128 k context, SOTA on OCRBench & DocVQA](https://github.com/AIDC-AI/Ovis.git)
- **MobileCLIP2** - [Image-text pair encoder designed as drop-in vision tower for FastVLM-style pipelines, 2× faster than CLIP-L at same zero-shot top-1](https://huggingface.co/collections/apple/mobileclip2)
- **FastVLM** - [Uses a hybrid vision encoder "FastViTHD" to reduce token count and encoding latency for high-resolution images, achieves up to 85× faster Time-to-First-Token (TTFT) relative to LLaVA-OneVision, with a vision encoder that is ~3.4× smaller](https://github.com/apple/ml-fastvlm) 
- **PerceptionLM (PLM)** - [Focuses on detailed image & video understanding, includes “Perception Encoder (PE)” variants that outperform SigLIP2 (image) and InternVideo2 (video) in benchmarks](https://github.com/facebookresearch/perception_models)
- **DeepSeek-VL2** - [MoE, 3B/16B params (3B active). Efficient for field IoT, scientific diagnostics, outperforms Cambrian-1 on MMMU at half the latency](https://github.com/deepseek-ai/DeepSeek-VL2.git)
- **Eve** - [Efficient multimodal model with “Elastic Visual Experts,”, 1.8B parameters, How to bridge the performance gap between encoder-free and encoder-based VLMs?](https://github.com/baaivision/EVE)
- **Kwai Keye-VL** - [Decoder-only, 8B params, Qwen-3-8B backbone. Tailored for short-form vertical-video understanding & mobile-first VQA. Released by Kuaishou, Apache-2.0](https://kwai-keye.github.io/)
- **Qwen2.5-Omni** - [End-to-end 7B any-to-any model (audio-image-video-text). Shares Qwen2-Audio encoder + Qwen2.5-VL ViT, with unified tokenizer for all modalities, Apache-2.0](https://github.com/QwenLM/Qwen2.5-Omni.git)
- **OmniGen2** - [Hybrid decoder + VAE. Trained on LLaVA-OneVision & SAM-LLaVA mixes, supports any-to-any generation](https://github.com/VectorSpaceLab/OmniGen2.git)
- **BLIP3-o** - [Decoder-only, 4B & 8B variants. Fine-tuned on 60K GPT-4o-generated instruction-image pairs, excels at open-ended captioning and creative editing](https://github.com/JiuhaiChen/BLIP3o.git)
- **InternVL3** - [Follows the "ViT-MLP-LLM", integrating InternViT with various pre-trained LLMs. Comprises seven models ranging from 1B to 78B parameters. Supports text, images, and videos simultaneously. Features Variable Visual Position Encoding (V2PE) for better long-context understanding. Utilizes pixel unshuffling and dynamic resolution strategy for efficient image processing. Training data includes tool usage, GUI operations, scientific diagrams, etc. Can be deployed as an OpenAI-compatible API via LMDeploy's api_server](https://huggingface.co/OpenGVLab/InternVL3-78B)
- **Falcon-2 11B VLM** - [Decoder-only, CLIP ViT-L/14 dynamic encoder, Apache-2.0. Built for multilingual OCR & fine-grained visual grounding](https://huggingface.co/tiiuae/falcon-11B-vlm)
- **Eagle 2.5** - [Decoder-only architecture with 8B parameters, designed for long-context multimodal learning, featuring a context window that can be dynamically adjusted based on input length. Utilizes progressive post-training to gradually increase the model's context window. Achieves 72.4% on Video-MME with 512 input frames, matching the results of top-tier commercial models like GPT-4o and large-scale open-source models like Qwen2.5-VL-72B and InternVL2.5-78B](https://github.com/NVlabs/EAGLE?tab=readme-ov-file)
- **Llama 3.2 Vision** - [Decoder-only architecture with 11B and 90B parameters, supports 8 languages - English, German, Italian, Portuguese, Hindi, Spanish, and Thai. NOTE: for image+text applications, English is the only language supported. Features a 128k-token context window, optimized for multimodal complex reasoning](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)
- **VLM-R1** - [Based on DeepSeek's R1 method, combined with Qwen2.5-VL, excels in Representational Expression Comprehension (REC), such as locating specific targets in images. Uses RL and SFT to enhance performance. Supports joint image and text processing. Provides open-source training code and dataset support](https://github.com/om-ai-lab/VLM-R1)
- **Kimi-VL** - [An efficient open-source MoE based VLM. Activates only 2.8B parameters in its language decoder (Kimi-VL-A3B). Demonstrates strong performance in multi-turn agent interaction tasks and diverse vision language tasks. Features a 128K extended context window for processing long inputs. Utilizes native-resolution vision encoder (MoonViT) for ultra-high-resolution visual inputs. Introduces Kimi-VL-Thinking variant for strong long-horizon reasoning capabilities](https://github.com/MoonshotAI/Kimi-VL/blob/main/Kimi-VL.pdf)
- **Molmo Series** - [Four models by Ai2: MolmoE-1B, Molmo-7B-O, Molmo-7B-D, and Molmo-72B. MolmoE-1B based on OLMoE-1B-7B, Molmo-7B-O on OLMo-7B-1024, Molmo-7B-D on Qwen2 7B, and Molmo-72B on Qwen2 72B. Known for high performance in vision-language tasks, with Molmo-72B outperforming many proprietary systems]()
- **Skywork-R1V** - [Open-sourced multimodal reasoning model with visual chain-of-thought capabilities. Supports mathematical and scientific analysis, and cross-modal understanding. Released quantized version (AWQ) for efficient inference](https://huggingface.co/Skywork/Skywork-R1V-38B)
- **DeepSeek Janus Series** - [Unified transformer architecture with decoupled visual encoding pathway, available in Janus-Pro (7B and 1B), Janus (1.3B), and JanusFlow (1.3B) parameter sizes, supports English and Chinese languages, designed to handle sequence lengths of up to 4,096 tokens](https://github.com/deepseek-ai/Janus)
- **Pixtral 12B & Pixtral Large** - [Decoder-only architecture with 12B (Pixtral 12B) and 124B (Pixtral Large) parameters, combines a 12B multimodal decoder with a 400M vision encoder. Pixtral Large integrates a 1B visual encoder with Mistral Large 2. Supports dozens of languages and features a 128k-token context window. Developed by Mistral AI, known for its efficiency in handling interleaved image and text data](https://huggingface.co/mistralai/Pixtral-12B-2409)
- **Gemma3** - [Decoder-only architecture with 1B/4B/12B/27B parameters, supporting multimodality and 140+ languages, featuring a 128k-token context window and function calling capabilities, based on Google's Gemini 2.0 architecture](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
- **PH4 Mini** - [Decoder-only architecture with specialized parameters, designed for efficient completion tasks, using optimized transformer layers and Microsoft's Phi-4 framework as pretrained backbone](https://arxiv.org/pdf/2503.01743)
- **Aya Vision 32B** - [Decoder-only architecture with 32B parameters, designed for advanced visual language understanding, supporting 23 languages and featuring dynamic image processing capabilities](https://huggingface.co/CohereForAI/aya-vision-32b)
- **Ola** - [Decoder-only architecture with 7B parameters, using OryxViT for vision encoder and Qwen-2.5-7B, SigLIP-400M, Whisper-V3-Large, BEATs-AS2M(cpt2) as pretrained backbone](https://arxiv.org/pdf/2502.04328)
- **Qwen2.5-VL** - [Decoder-only architecture with 3B/7B/72B parameters, using redesigned ViT for vision encoder and Qwen2.5 as pretrained backbone](https://arxiv.org/pdf/2502.13923)
- **Ocean-OCR** - [Decoder-only architecture with 3B parameters, using NaViT for vision encoder and pretrained from scratch](https://arxiv.org/pdf/2501.15558)
- **SmolVLM** - [Decoder-only architecture with 250M & 500M parameters, using SigLIP for vision encoder and SmolLM as pretrained backbone](https://huggingface.co/blog/smolervlm)

### 2024
<a id="sm-2024"></a>

- **Video-LLaVA** - [Video understanding model based on LLaVA architecture, designed to process and understand visual information in video format](https://arxiv.org/pdf/2311.10122)
- **PaliGemma** - [Multimodal understanding and generation model, designed to handle complex interactions between different data modalities](https://arxiv.org/pdf/2407.07726)
- **Emu3** - [Decoder-only architecture with 7B parameters, using MoVQGAN for vision encoder and LLaMA-2 as pretrained backbone](https://arxiv.org/pdf/2409.18869)
- **NVLM** - [Encoder-decoder architecture with 8B-24B parameters, using custom ViT for vision encoder and Qwen-2-Instruct as pretrained backbone](https://arxiv.org/pdf/2409.11402
)
- **Qwen2-VL** - [Decoder-only architecture with 7B-14B parameters, using EVA-CLIP ViT-L for vision encoder and Qwen-2 as pretrained backbone](https://arxiv.org/pdf/2409.12191)
- **Pixtral** - [Decoder-only architecture with 12B parameters, using CLIP ViT-L/14 for vision encoder and Mistral Large 2 as pretrained backbone](https://arxiv.org/pdf/2410.07073)
- **CoPali** - [Code generation model that focuses on programming assistance, designed to understand natural language instructions and generate corresponding code](https://arxiv.org/pdf/2407.01449)
- **SAM 2** - [Foundation model for image segmentation, designed to handle diverse segmentation tasks with minimal prompting](https://arxiv.org/pdf/2408.00714)
- **LLaMA 3.2-vision** - [Decoder-only architecture with 11B-90B parameters, using CLIP for vision encoder and LLaMA-3.1 as pretrained backbone](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)
- **Baichuan Ocean Mini** - [Decoder-only architecture with 7B parameters, using CLIP ViT-L/14 for vision encoder and Baichuan as pretrained backbone](https://arxiv.org/html/2410.08565v1)
- **DeepSeek-VL2** - [Decoder-only architecture with 4.5B x 74 parameters, using SigLIP/SAMB for vision encoder and DeepSeekMoE as pretrained backbone](https://arxiv.org/pdf/2412.10302)
- **Qwen-VL** - [Encoder-decoder architecture with 7B parameters, using a Vision Transformer (ViT) for vision encoding and Qwen (a Transformer-based LLM) as the pretrained text backbone](https://arxiv.org/pdf/2308.12966)
- **SigLip** - [Specialized architecture for sign language interpretation, using advanced gesture and motion recognition techniques](https://arxiv.org/pdf/2303.15343)

### 2023
<a id="sm-2023"></a>

- **ImageBind** - [Multi-encoder architecture with modality-specific encoders (ViT-H for vision, ~632M parameters) trained to align embeddings across 6 modalities (image, text, audio, depth, etc.)](https://arxiv.org/pdf/2305.05665)
- **InstructBLIP** - [Encoder-decoder architecture with 13B parameters, using ViT for vision encoder and Flan-T5, Vicuna as pretrained backbone](https://arxiv.org/pdf/2305.06500)
- **InternVL** - [Encoder-decoder architecture with 7B/20B parameters, using Eva CLIP ViT-g for vision encoder and QLLaMA as pretrained backbone](https://arxiv.org/pdf/2312.14238)
- **CogVLM** - [Encoder-decoder architecture with 18B parameters, using CLIP ViT-L/14 for vision encoder and Vicuna as pretrained backbone](https://arxiv.org/pdf/2311.03079)
- **Kosmos-2** - [Multimodal architecture that can handle images, text, and other modalities, designed for comprehensive understanding and generation across different data types](https://arxiv.org/pdf/2306.14824)
- **BLIP-2** - [Encoder-decoder architecture with 7B-13B parameters, using ViT-g for vision encoder and Open Pretrained Transformer (OPT) as pretrained backbone](https://arxiv.org/pdf/2301.12597)
- **PaLM-E** - [Decoder-only architecture with 562B parameters, using ViT for vision encoder and PaLM as pretrained backbone](https://arxiv.org/pdf/2303.03378)
- **LLaVA-1.5** - [Decoder-only architecture with 13B parameters, using CLIP ViT-L/14 for vision encoder and Vicuna as pretrained backbone](https://arxiv.org/pdf/2304.08485)

### 2022 & Prior
<a id="sm-2022-prior"></a>

- **Flamingo** - [Decoder-only architecture with 80B parameters, using custom vision encoder and Chinchilla as pretrained backbone](https://arxiv.org/pdf/2204.14198)
- **BLIP** - [Encoder-decoder architecture using ViT-B/L/g for vision encoder and pretrained from scratch for language encoder](https://arxiv.org/pdf/2201.12086)
- **CLIP** - [Dual-encoder architecture with ~400M parameters, using a Vision Transformer (ViT, e.g., ViT-L/14) for vision encoding and a Transformer for text encoding. Trained contrastively on 400M image-text pairs for multimodal alignment](https://arxiv.org/pdf/2103.00020)    
- **FLAVA** - [Unified transformer architecture with separate image and text encoders plus a multimodal encoder, designed to handle vision, language, and multimodal reasoning tasks simultaneously](https://arxiv.org/pdf/2112.04482)

## 📊 Datasets
<a id="datasets"></a>

#### **Core Training Datasets**
<a id="core-training-datasets"></a>

- **COCO** - [Contains 328K images, each paired with 5 captions for image captioning and VQA](https://huggingface.co/datasets/HuggingFaceM4/COCO)  
- **Conceptual Captions 3M** - [3M web-mined image-text pairs for pretraining VLMs](https://huggingface.co/datasets/conceptual_captions)
- **Conceptual Captions 12M** - [12M web-mined image-text pairs for pretraining VLMs](https://github.com/google-research-datasets/conceptual-12m)
- **Coyo-700M** - [700 M English-centric pairs, aggressive near-dedup + safety filter, used to train FastVLM & InternVL-3](https://huggingface.co/datasets/kakaobrain/coyo-700m)
- **LAION400M** - [400 million image-text pairs](https://laion.ai/blog/laion-400-open-dataset/)  
- **LAION-5B** - [5B image-text pairs from Common Crawl for large-scale pretraining](https://laion.ai/blog/laion-5b/)  
- **ALIGN** - [1.8B noisy alt-text pairs for robust multimodal alignment](https://research.google/blog/align-scaling-up-visual-and-vision-language-representation-learning-with-noisy-text-supervision/)  
- **SBU Caption** - [1M image-text pairs from web pages](https://huggingface.co/datasets/sbu_captions)  
- **Visual Genome** - [5.4M object/relationship annotations](https://www.kaggle.com/datasets/mathurinache/visual-genome)  
- **WuKong** - [100M Chinese image-text pairs](https://wukong-dataset.github.io/wukong-dataset/) 
- **Localized Narratives** - [0.87 million image-text pairs](https://paperswithcode.com/dataset/localized-narratives)
- **Wikipedia-based Image Text** - [37.6 million image-text pairs, 108 languages](https://github.com/google-research-datasets/wit)
- **Red Caps** - [12 million image-text pairs](https://huggingface.co/datasets/kdexd/red_caps)
- **FILIP300M** - [300 million image-text pairs](https://openreview.net/forum?id=cpDhcsEDC2)
- **WebLI** - [12 billion image-text pairs](https://paperswithcode.com/dataset/webli)
- **MMC4-Core** - [101M interleaved image-text sequences (≤10 images/doc) extracted from Common-Crawl, permissive ODC-BY licence](https://github.com/allenai/mmc4)
- **OBELICS** - [An open and curated collection of interleaved image-text web documents used by the Idefics series. contains 141M documents and is suitable for multimodal pretraining](https://huggingface.co/datasets/HuggingFaceM4/OBELICS)

#### Image Classification
<a id="image-classification"></a>

- **ImageNet-1k** - [1.28M training images across 1,000 classes](https://huggingface.co/datasets/imagenet-1k)  
- **CIFAR-10/100** - [60K low-resolution images for small-scale testing](https://huggingface.co/datasets/cifar10)  
- **Food-101** - [101 food categories with 1,000 images each](https://huggingface.co/datasets/food101)
- **MNIST** - [Handwritten digit database](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- **Caltech-101** - [101 object categories](https://paperswithcode.com/dataset/caltech-101)
- **JourneyDB** - [4M QA, caption, and text prompting tasks based on Midjourney images](https://huggingface.co/datasets/JourneyDB/JourneyDB)

#### Object Detection
<a id="object-detection"></a>

- **COCO Detection** - [118K images with 80 object categories](https://cocodataset.org/#detection-2017)  
- **LVIS** - [1,203 long-tail object categories](https://www.lvisdataset.org/)
- **ODinW** - [Object detection in the wild](https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw) 

#### Semantic Segmentation
<a id="semantic-segmentation"></a>

- **Cityscapes** - [5,000 urban scene images with pixel-level labels](https://www.cityscapes-dataset.com/)  
- **ADE20k** - [25K images with 150 object/part categories](https://www.kaggle.com/datasets/awsaf49/ade20k-dataset)
- **PASCAL VOC** - [Visual Object Classes dataset](https://paperswithcode.com/dataset/pascal-voc)  

#### Action Recognition and 
<a id="action-recognition"></a>

- **UCF101** - [13K video clips across 101 actions](https://www.crcv.ucf.edu/data/UCF101.php)  
- **Kinetics700** - [500K video clips covering 700 human actions](https://paperswithcode.com/dataset/kinetics-700)
- **RareAct** - [Dataset for recognizing rare actions](https://paperswithcode.com/dataset/rareact) 

#### Image-Text Retrieval
<a id="image-text-retrieval"></a>

- **GRIT** (2023) - [a large-scale dataset of Grounded Image-Text pairs](https://huggingface.co/datasets/zzliang/GRIT)
- **Flickr30k** (2014) - [31K images with dense textual descriptions](https://www.kaggle.com/datasets/adityajn105/flickr30k)  
- **COCO Retrieval** (2015) - [Standard benchmark for cross-modal matching](https://cocodataset.org/#home)

#### Visual Question Answering (VQA)
<a id="visual-question-answering-vqa"></a>

- **Situat3DChange** - [developed for perception-action modeling in dynamic 3D environments](https://huggingface.co/datasets/lrp123/Situat3DChange)
- **ReasonVQA** - [ Multi-hop Reasoning Benchmark with Structural Knowledge for VQA](https://duong-tr.github.io/ReasonVQA/)
- **ClearVQA** - [Ambiguous visual questions categorized into three types: referential ambiguity, intent underspecification, and spelling ambiguity](https://huggingface.co/datasets/jian0418/ClearVQA)
- **MedFrameQA** - [A Multi-Image Medical VQA Benchmark for Clinical Reasoning](https://ucsc-vlaa.github.io/MedFrameQA/)
- **VQA** - [Visual Question Answering dataset](https://visualqa.org/)
- **GQA** - [Dataset for compositional question answering](https://paperswithcode.com/dataset/gqa)
- **OK-VQA** - [11K open-ended questions requiring external knowledge](https://okvqa.allenai.org/)
- **ScienceQA** - [21K science questions for multimodal reasoning](https://scienceqa.github.io/)
- **TextVQA** - [Dataset to read and reason about text in images](https://textvqa.org/)
- **VisualReasoner-1M** - [Contains approximately 1 million cases and can be used for training visual reasoning tasks](https://huggingface.co/datasets/orange-sk/VisualReasoner-1M)
 
#### **Instruction Tuning**
<a id="instruction-tuning"></a>

- **M³IT** - [Comprises 40 tasks with 400 human-written instructions, containing 2.4M instruction-image instances](https://m3-it.github.io/)
- **LLaVA Instruct** - [260K image-conversation pairs for instruction fine-tuning](https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft)
- **LLaVA-CoT-100k** - [from paper - LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k)
- **MIMICIT** - [2.8M multimodal instruction-response pairs](https://huggingface.co/datasets/pufanyi/MIMICIT)

#### **Bias**
<a id="bias"></a>

- **MM-RLHF** - [Contains 120,000 finely annotated preference comparison pairs, including scores, rankings, textual descriptions of reasons, and tie annotations across three dimensions](https://mm-rlhf.github.io/)
- **Chicago Face Dataset (CFD)** - [Provides facial images of different races and genders, helps analyzing how models classify and identify faces of various demographic groups and uncover potential biases](https://www.chicagofaces.org/)
- **SocialCounterfactuals** - [Dataset containing 171,000 image-text pairs generated using an over-generate-then-filter method](https://huggingface.co/datasets/Intel/SocialCounterfactuals)
- **StereoSet** - [Dataset targeting stereotypical bias in multimodal models](https://huggingface.co/datasets/McGill-NLP/stereoset)

#### **Video Understanding**
<a id="video-understanding"></a>

- **VideoMathQA** - [Benchmarking Mathematical Reasoning via Multimodal Understanding in Videos](https://mbzuai-oryx.github.io/VideoMathQA/#leaderboard-2)
- **VideoChat2-IT** - [From the paper - VideoChat: Chat-Centric Video Understanding](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT)

#### **Additional Datasets**
<a id="additional-datasets"></a>

- **SpatialLadder-26k** - [A multimodal dataset containing 26,610 samples spanning object localization, single-image, multi-view, and video spatial reasoning tasks](https://huggingface.co/datasets/hongxingli/SpatialLadder-26k)
- **FragFake** - [A Dataset for Fine-Grained Detection of Edited Images with Vision Language Models](https://huggingface.co/datasets/Vincent-HKUSTGZ/FragFake)
- **The P3 dataset** - [Pixels, Points and Polygons for Multimodal Building Vectorization](https://github.com/raphaelsulzer/PixelsPointsPolygons.git)
- **ReXGradient-160K** - [ A Large-Scale Publicly Available Dataset of Chest Radiographs with Free-text Reports](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K)
- **Open Images** - [9M images with multi-label annotations](https://docs.ultralytics.com/datasets/detect/open-images-v7/)  
- **Hateful Memes** - [10K memes for hate speech detection](https://paperswithcode.com/dataset/hateful-memes)  
- **EuroSAT** - [27K satellite images for land use classification](https://github.com/phelber/eurosat)
- **MathVista** - [Evaluating Math Reasoning in Visual Contexts](https://mathvista.github.io/)
- **Multimodal ArXiv** - [A Dataset for Improving Scientific Comprehension of Large Vision-Language Models](https://mm-arxiv.github.io/)

## 🏆 Benchmarks
<a id="benchmarks"></a>

- **GEOBench-VLM** (2025) - [For Geospatial Tasks](https://huggingface.co/datasets/aialliance/GEOBench-VLM)
- **EgoExoBench** (2025) - [Designed to evaluate cross-view video understanding in MLLMs, contains paired egocentric–exocentric videos and over 7,300 MCQs across 11 subtasks](https://arxiv.org/abs/2507.18342)
- **AV-Reasoner** (2025) - [Improving and Benchmarking Clue-Grounded Audio-Visual Counting for MLLMs](https://av-reasoner.github.io/)
- **EOC-Bench** (2025) - [Can MLLMs Identify, Recall, and Forecast Objects in an Egocentric World?](https://circleradon.github.io/EOCBench/)
- **RBench-V** (2025) - [A Primary Assessment for Visual Reasoning Models with Multi-modal Outputs](https://evalmodels.github.io/rbenchv/)
- **LENS** (2025) - [A multi-level benchmark explicitly designed to assess MLLMs across three hierarchical tiers—Perception, Understanding, and Reasoning—encompassing 8 tasks and 12 real-world scenarios.](https://github.com/Lens4MLLMs/LENS.git)
- **On Path to Multimodal Generalist: General-Level and General-Bench** (2025) - [Does higher performance across tasks indicate a stronger capability of MLLM, and closer to AGI?](https://arxiv.org/pdf/2505.04620)
- **VisuLogic** (2025) - [Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models](https://visulogic-benchmark.github.io/VisuLogic/) 
- **V2R-Bench** (2025) - [Holistically Evaluating LVLM Robustness to Fundamental Visual Variations](https://arxiv.org/pdf/2504.16727)
- **Open VLM Leaderboard** [HF(2025)](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- **SVLTA** (2025) - [Benchmarking Vision-Language temporal alignment via synthetic video situation](https://svlta-ai.github.io/SVLTA/)
- **Fewshot-VLM** (2025) - [Evaluates adaptation to new tasks with limited examples](https://arxiv.org/html/2501.02189v3) 
- **OCRBench & OCRBench v2** (2024) - [OCR capabilities of Large Multimodal Models](https://github.com/Yuliang-Liu/MultimodalOCR)
- **MMMU-Pro** - [contains 28k college-level questions delivered as screenshots with varying fonts/layouts](https://huggingface.co/datasets/MMMU/MMMU_Pro)
- **MMT-Bench** (2024) - [comprises 31,325 curated multi-choice visual questions from various multimodal scenarios such as vehicle driving and embodied navigation, covering 32 core meta-tasks and 162 subtasks in multimodal understanding](https://mmt-bench.github.io/)
- **ZeroShot-VLM** (2023) - [Tests generalization to unseen tasks without task-specific training](https://arxiv.org/html/2305.14196v3)  
- **MetaPrompt** (2023) - [Measures domain generalization with unseen prompts/domains](https://ai.meta.com/research/publications/unibench-visual-reasoning-requires-rethinking-vision-language-beyond-scaling/)
- **MMBench** (2023) - [Multimodal benchmark](https://github.com/open-compass/MMBench) 
- **MMMU** (2023) - [Massive Multi-discipline Multimodal Understanding benchmark](https://mmmu-benchmark.github.io/)
- **ChartQA** (2022) - [Question Answering about charts with visual and logical reasoning](https://github.com/vis-nlp/ChartQA.git)

#### **Video-Language Benchmarks**
- **RTV-Bench** (2025) - [Real-time video analysis, contains 552 videos (167.2 hours) and 4,631 QA pairs](https://ljungang.github.io/RTV-Bench/)
- **SciVideoBench** (2025) - [Benchmark for scientific video reasoning, covering disciplines in Physics, Chemistry, Biology, and Medicine](https://huggingface.co/datasets/groundmore/scivideobench)
- **Video-MME v2** (2025) - [Updated 900 → 1800 videos (508 h), adds audio-only & subtitle-only splits, now the default long-video benchmark for GPT-5 & Gemini-2.5-Pro](https://video-mme.github.io/home_page.html)
- **MT-Video-Bench** (2025) - [Video Understanding Benchmark for Evaluating Multimodal LLMs in Multi-Turn Dialogues](https://mt-video-bench.github.io/)
- **Video-MMLU** (2025) - [A Massive Multi-Discipline Lecture Understanding Benchmark](https://enxinsong.com/Video-MMLU-web/)
- **VLM²-Bench** (2024) - [Evaluates multi-image/video linking capabilities (9 subtasks, 3K+ test cases)](https://vlm2-bench.github.io/)  
- **ViLCo-Bench** (2024) - [Continual learning for video-text tasks](https://github.com/cruiseresearchgroup/ViLCo)  

#### **Dynamic Evaluation**
- **VLM4D** (2025) - [4-D spatiotemporal reasoning benchmark](https://vlm4d.github.io/)
- **OmniSpatial** (2025) - [Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](https://qizekun.github.io/omnispatial/)
- **SPBench** (2025) - [Spatial Perception and Reasoning Benchmark](https://huggingface.co/datasets/hongxingli/SPBench)
- **MMSI-Bench** (2025) - [A Benchmark for Multi-Image Spatial Intelligence](https://runsenxu.com/projects/MMSI_Bench/)
- **Spatial-DISE** (2025) - [Evaluating Spatial Reasoning in Vision-Language Models](https://arxiv.org/pdf/2510.13394)
- **LiveXiv** (2024) - [Monthly-changing benchmark to prevent overfitting, estimates true model capabilities](https://arxiv.org/abs/2410.10783)
- **MM-Vet** (2023) - [Evaluating Large Multimodal Models for Integrated Capabilities](https://github.com/yuweihao/MM-Vet)  

#### **Specialized Tasks**
- **ScienceQA** (2022) - [21K science questions for multimodal reasoning](https://scienceqa.github.io/)  
- **OK-VQA** (2021) - [11K open-ended questions requiring external knowledge](https://okvqa.allenai.org/)    

## 🔍 Research Directions
<a id="research-directions"></a>

### Regression Tasks  
- VLMs often struggle with numerical reasoning, counting, measurement tasks.
- Hybrid numeric modules, symbolic-differentiable integration, specialized heads should be suggested.  

### Diverse Visual Data  
- Expansion to non-RGB modalities - multispectral, depth, LiDAR, thermal, medical imaging. 
- Dealing with domain shift, modality gap, alignment strategies.  

### Multimodal Output Beyond Text  
- Generation of images, videos, 3D scenes (like with diffusion + VLM couplings)  
- Dense prediction tasks - segmentation, layout generation, vision + language fused output  
- Challenge - coherence across modalities, consistency, speed  

### Multitemporal & Continual Learning  
- Lifelong / online VLMs that adapt over time  
- Avoid catastrophic forgetting across visual domains  
- Enable adaptation to evolving visual environments  

### Efficient Edge / Deployment  
- Quantization, pruning, distillation, adapters, LoRA for vision-language models.  
- Mobile / ARM / GPU runtimes, optimized kernels. 
- Example - [FastVLM](https://machinelearning.apple.com/research/fast-vision-language-models?utm_source=chatgpt.com) is a major step in efficient inference design.   

### Multimodal Alignment & Fusion  
- Methods - gated cross-attention, mixture-of-prompts, dynamic modality weighting, feature modulation.  
- [LaVi (vision-modulated layernorm)](https://arxiv.org/abs/2506.16691) would be a good example 
- Address density mismatch - dense visual features vs discrete text tokens  

### Embodied AI / Vision-Language-Action (VLA)  
- VLMs extended to output robot actions given vision + instructions  
- Classic example- [RT-2](https://arxiv.org/abs/2307.15818) (Google DeepMind) as early vision–language–action foundation  
- [LiteVLP](https://hustvl.github.io/LiteVLP/) - lightweight, memory-based vision-language policy for robot tasks, showing improvements over existing baselines. 
- Challenges - sample efficiency, latency, generalization, sim-to-real  

### Temporal Reasoning  
- Video question answering, event prediction, cross-frame consistency.  
- Need memory modules, temporal fusion, temporal attention. 

### Medical & Domain-Specific VLMs  
- Domain shift, hallucination, safety-critical constraints. 
- Combine domain-specific modules, calibration, prompt tuning.  
- Require robust uncertainty estimation, out-of-distribution detection. 


##  ⚠️ Ethical Challenges
<a id="ethical-challenges"></a>

| Bias Type          | Prevalence | High-Risk Domains     | Mitigation Effectiveness                     |
|--------------------|------------|-----------------------|----------------------------------------------|
| Gender             | 23%        | Career images         | 63% reduction (Counterfactual)               |
| Racial             | 18%        | Beauty standards      | 58% (Adversarial)                            |
| Cultural           | 29%        | Religious symbols     | 41% (Data Filtering)                         |
| Hallucination      | 34%        | Medical reports       | 71% (CHAIR metric)                           |
| Spatial Reasoning  | High       | Scene understanding   | Requires further research                    |
| Counting           | Moderate   | Object detection      | Requires specialized techniques              |
| Attribute Recognition| Moderate   | Detailed descriptions | Needs improved mechanisms                    |
| Prompt Ignoring    | Moderate   | Task-specific prompts | Requires better understanding of intent      |

- **Visual privacy & identity leakage** - reconstructing faces or sensitive information from embeddings  
- **Adversarial / backdoor attacks in visuals** - e.g. adversarial patches  
- **Copyright / image usage / dataset licensing**  
- **Geographic / demographic bias in visual datasets**  
- **Multimodal RLHF / preference bias** - mixed visual + language feedback  
- **Explainability in multimodal decisions**  
- **Misuse / dual-use risk** - surveillance, deepfakes, misinformation  
- **Mitigation strategies** - adversarial robustness, counterfactual data, filtering, human oversight  


## 🔒 Privacy Protection Framework
<a id="privacy-protection-framework"></a>

```bash
graph TD
    A[Raw Data] --> B{Federated Learning?}
    B -->|Yes| C[Differential Privacy]
    C --> D[Secure Training]
    B -->|No| E[Reject]
```
VLMs often process sensitive data (medical images, personal photos, etc.). This framework prevents data leakage while maintaining utility:

→ **Federated Learning Check**

  -  Purpose: Train models on decentralized devices without raw data collection
  -  Benefit: Processes user photos/text locally (e.g., mobile camera roll analysis)
  - Why Required: 34% of web-scraped training data contains private info (LAION audit)

→  **Differential Privacy (DP)**
 ```python
 # DP-SGD Implementation for Medical VLMs
optimizer = DPAdam(
    noise_multiplier=1.3,  
    l2_norm_clip=0.7,      
    num_microbatches=32
)
```
  - Guarantees formal privacy (ε=3.8, δ=1e-5)
- Prevents memorization of training images/text

→ **Secure Training**

- Homomorphic Encryption: Process encrypted chest X-rays/patient notes
- Trusted Execution Environments: Isolate retinal scan analysis
- Prevents: Model inversion attacks that reconstruct training images

→ **Reject Pathway**

- Triggered for:
  - Web data without consent (23% of WebLI dataset rejected)  
  - Protected health information (HIPAA compliance)
  - Biometric data under GDPR

↓ **Real-World Impact**
| Scenario              | Without Framework       | With Framework          |
|-----------------------|-------------------------|-------------------------|
| Medical VLM Training  | 12% patient ID leakage | 0.03% leakage risk    |
| Social Media Photos   | Memorizes user faces   | Anonymous embeddings    |
| Autonomous Vehicles | License plate storage   | Local processing only |


## 🛠️ Tools 
<a id="tools"></a>

- **VLMEvalKit** - [Unified evaluation framework supporting 20+ benchmarks (Apache-2.0)](https://github.com/open-compass/VLMEvalKit)   
- **VLM-BiasCheck** - [Bias auditing toolkit covering 15+ dimensions (MIT)](https://arxiv.org/abs/2306.17202)

- **Prompt tuning / adapter frameworks** - [CoOp/CoCoOp](https://arxiv.org/pdf/2203.05557), [VLSM-Adapter](https://arxiv.org/pdf/2405.06196), [FedMVP](https://arxiv.org/pdf/2504.20860), [CHROME](https://arxiv.org/pdf/2408.06610) 

#### **Optimization Toolkit**
- **4-bit QAT** - [Quantization-aware training via bitsandbytes (3.2× speedup)](https://github.com/TimDettmers/bitsandbytes)  
- **Flash Attention** - [Memory-efficient attention with xFormers (2.8× speedup)](https://github.com/Dao-AILab/flash-attention)  
- **Layer Dropping** - [Structural pruning via torch.prune (1.9× speedup)](https://paperswithcode.com/method/layerdrop)
 
  
## 📌Emerging Applications 
<a id="emerging-applications"></a>

- **Robotics / embodied agents** (vision + language + action)  
- **AR / XR / smart glasses** - real-time vision + language understanding overlays  
- **Document / UI understanding** - combining layout, OCR, semantics  
- **Scientific / diagram reasoning** - charts, formulas, visuals + language  
- **Satellite / geospatial + textual metadata fusion**  
- **Medical imaging + clinical note fusion**  
- **Assistive technologies** - interactive image description + QA for visually impaired  
- **Visual-to-code / UI generation** - sketches or mockups → code  
- **Video summarization / captioning / QA over long videos**


## 🤝Contributing

Thank you for considering contributing to this repository! The goal is to create a comprehensive, community-driven resource for Multimodal and VLM researchers. Contributions ranging from updates to models, datasets, and benchmarks, to new code examples, ethical discussions, and research insights would be welcomed:)





 



