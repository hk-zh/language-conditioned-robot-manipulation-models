# Bridging Language and Action: Awesome Language-Conditioned Robot Manipulation Models [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

[![arXiv](https://img.shields.io/badge/arXiv-2312.10807-b31b1b.svg)](https://arxiv.org/abs/2312.10807)
<a href="https://github.com/hk/language-conditioned-robot-manipulation-models/stargazers"><img src="https://img.shields.io/github/stars/hk-zh/language-conditioned-robot-manipulation-models" alt="Stars Badge"/></a> 
<a href="https://github.com/hk-zh/language-conditioned-robot-manipulation-models/watchers">
  <img src="https://img.shields.io/github/watchers/hk-zh/language-conditioned-robot-manipulation-models?style=social" alt="Watchers Badge"/>
</a>
<a href="https://github.com/hk-zh/language-conditioned-robot-manipulation-models/network/members">
  <img src="https://img.shields.io/github/forks/hk-zh/language-conditioned-robot-manipulation-models?style=social" alt="Forks Badge"/>
</a>
<a href="https://github.com/hk-zh/language-conditioned-robot-manipulation-models/pulls"><img src="https://img.shields.io/github/issues-pr/hk-zh/language-conditioned-robot-manipulation-models" alt="Pull Requests Badge"/></a>
<a href="https://github.com/hk-zh/language-conditioned-robot-manipulation-models/issues"><img src="https://img.shields.io/github/issues/hk-zh/language-conditioned-robot-manipulation-models" alt="Issues Badge"/></a>
<a href="https://github.com/hk-zh/language-conditioned-robot-manipulation-models/blob/main/LICENSE"><img src="https://img.shields.io/github/license/hk-zh/language-conditioned-robot-manipulation-models" alt="License Badge"/></a>
> A curated, taxonomy-driven collection of **language-conditioned robot manipulation** papers, code, simulators, and benchmarks – tracking the literature behind  
> **“Bridging Language and Action: A Survey of Language-Conditioned Robot Manipulation” (arXiv:2312.10807).**

If you find this repo useful, please:

- ⭐ **Star** the repo  
- 👀 **Watch** for updates  
- 🧑‍💻 **Open a PR** to add missing papers or fixes  

so more people can discover and build on this survey!

<p align="center">
  <img src="graphs/overview.png" alt="Overview" width="80%">
</p>

---

## News

- **[November 20, 2025]** Further extension of the survey paper with **new structure** (language roles taxonomy) and **more recent works (2024–2025)**.
- **[November 30, 2024]** Extended survey paper is available.
- **[October 02, 2024]** Cutting-edge papers in 2024 are available.

---

## Table of Contents

- [Survey Paper](#survey-paper)
- [Taxonomy: How Language Bridges Perception and Control](#taxonomy-how-language-bridges-perception-and-control)
- [Language as a Policy Condition – Paper Collections](#language-as-a-policy-condition--paper-collections)
  - [Language-Conditioned Reinforcement Learning](#language-conditioned-reinforcement-learning)
  - [Language-Conditioned Imitation Learning](#language-conditioned-imitation-learning)
  - [Diffusion-Based Policies](#diffusion-policy)
- [Language for Cognitive Planning and Reasoning – Paper Collections](#language-for-cognitive-planning-and-reasoning--paper-collections)
  - [Neural-Symbolic Methods](#neuralsymbolic)
  - [Empowered by LLMs](#empowered-by-llms)
  - [Empowered by VLMs](#empowered-by-vlms)
- [Comparative Analysis](#comparative-analysis)
  - [Benchmarks](#benchmarks)
  - [Models](#models)
- [How to Contribute](#how-to-contribute)
- [Citation](#citation)


## Survey Paper

This repository is built around the survey:

**[Bridging Language and Action: A Survey of Language-Conditioned Robot Manipulation](https://arxiv.org/abs/2312.10807)**  
*Hongkuan Zhou, Xiangtong Yao, Oier Mees, Yuan Meng, Ted Xiao, Yonatan Bisk, Jean Oh, Edward Johns, Mohit Shridhar, Dhruv Shah, Jesse Thomason, Kai Huang, Joyce Chai, Zhenshan Bing, Alois Knoll*


## Taxonomy: How Language Bridges Perception and Control

The **structure of the survey** organizes methods by the *role* language plays in the system.

At a high level, language can:

1. **Evaluate what the robot is doing**  
   -> *Language for state evaluation* 
   Language becomes a **reward, cost, or scoring function**, used to measure task progress, preferences, or goal satisfaction.

2. **Specify how the robot should act**  
   -> *Language as a policy condition* (Sec. 5)  
   Language is fed **directly into the policy**, shaping the action distribution at each step (e.g., language-conditioned RL, BC, diffusion policies).

3. **Help the robot think and plan**  
   -> *Language for cognitive planning and reasoning* (Sec. 6)  
   Language is used as an **internal reasoning medium**: planning, decomposition, querying knowledge bases, or manipulating symbolic structures.

Below, we briefly summarize each role and show how it maps to the sections in this repo.

---

## 4. Language for State Evaluation

### 4.1 Reward Design / Learning

#### 4.1.1 Reward Designing

- Zero-Shot Reward Specification via Grounded Natural Language [[paper]](https://proceedings.mlr.press/v162/mahmoudieh22a.html) 
- Trajectory Improvement and Reward Learning from Comparative Language Feedback [[paper]](https://proceedings.mlr.press/v270/yang25e.html)[[code]](https://github.com/USC-Lira/language-preference-learning)
- PixL2R: Guiding Reinforcement Learning Using Natural Language by Mapping Pixels to Rewards [[paper]](https://proceedings.mlr.press/v155/goyal21a.html)

#### 4.1.2 Reward Learning
- Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization [[paper]](https://proceedings.mlr.press/v48/finn16.html) 
- From Language to Goals: Inverse Reinforcement Learning for Vision-Based Instruction Following [[paper]](https://openreview.net/forum?id=r1lq1hRqYQ)
- Grounding English Commands to Reward Functions [[paper]](https://cs.brown.edu/~jmacglashan/pubpdfs/rss_commands.pdf)
- Model-Based Inverse Reinforcement Learning from Visual Demonstrations [[paper]](https://proceedings.mlr.press/v155/das21a.html)
- From Language to Goals: Inverse Reinforcement Learning for Vision-Based Instruction Following [[paper]](https://arxiv.org/abs/1902.07742)
- Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation [[paper]](https://arxiv.org/abs/2109.01115)[[code]](https://github.com/suraj-nair-1/lorel)

#### 4.1.3 FM-driven Reward Design / Learning

- Reward Design with Language Models [[paper]](https://openreview.net/pdf?id=10uNUgI5Kl) [[code]](https://github.com/minaek/reward_design_with_llms)
- Language Reward Modulation for Pretraining Reinforcement Learning [[paper]](https://openreview.net/forum?id=zzHUJYe3Py) [[code]](https://github.com/ademiadeniji/lamp)
- Language to Rewards for Robotic Skill Synthesis [[paper]](https://proceedings.mlr.press/v229/yu23a.html) [[code]](https://github.com/google-deepmind/language_to_reward_2023)
- RoboGen: towards unleashing infinite data for automated robot learning via generative simulation [[paper]](https://dl.acm.org/doi/abs/10.5555/3692070.3694197) [[code]](https://github.com/Genesis-Embodied-AI/RoboGen)
- Text2Reward: Reward Shaping with Language Models for Reinforcement Learning [[paper]](https://openreview.net/forum?id=tUM39YTRxH) [[code]](https://github.com/xlang-ai/text2reward)
- Eureka: Human-Level Reward Design via Coding Large Language Models [[paper]](https://openreview.net/forum?id=IEduRUO55F) [[code]](https://github.com/eureka-research/Eureka) 
- Learning reward for robot skills using large language models via self-alignment [[paper]](https://dl.acm.org/doi/10.5555/3692070.3694478) [[code]](https://github.com/friolero/self_aligned_reward_learning)
- R*: Efficient Reward Design via Reward Structure Evolution and Parameter Alignment Optimization with Large Language Models [[paper]](https://proceedings.mlr.press/v267/li25v.html)
- RLingua: Improving Reinforcement Learning Sample Efficiency in Robotic Manipulations With Large Language Models [[paper]](https://ieeexplore.ieee.org/document/10529514)
- ELEMENTAL: Interactive Learning from Demonstrations and Vision-Language Models for Reward Design in Robotics [[paper]](https://proceedings.mlr.press/v267/chen25at.html)
- Video-Language Critic: Transferable Reward Functions for Language-Conditioned Robotics [[paper]](https://arxiv.org/abs/2405.19988) [[code]](https://github.com/minttusofia/video_language_critic)
- Guiding reinforcement learning with shaping rewards provided by the vision–language model [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0952197625010048)
- ReWiND: Language-Guided Rewards Teach Robot Policies without New Demonstrations [[paper]](https://proceedings.mlr.press/v305/zhang25a.html)[[code]](https://github.com/rewind-reward/ReWiND)


### 4.2 Cost Functions Mapping

- Correcting Robot Plans with Natural Language Feedback [[paper]](https://arxiv.org/pdf/2204.05186)
- VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models [[paper]](https://voxposer.github.io/voxposer.pdf) [[code]](https://github.com/huangwl18/VoxPoser)
- Language-Conditioned Path Planning [[paper]](https://proceedings.mlr.press/v229/xie23b/xie23b.pdf) [[code]](https://github.com/amberxie88/lapp)
- IMPACT: Intelligent Motion Planning with Acceptable Contact Trajectories via Vision-Language Models [[paper]](https://arxiv.org/abs/2503.10110) [[code]](https://impact-planning.github.io/)



## 5. Language as the Policy Conditions

### 5.1 Reinforcement Learning

- Language-Conditioned Goal Generation: a New Approach to Language Grounding for RL [[paper]](https://arxiv.org/abs/2006.07043) 
- LanCon-Learn: Learning With Language to Enable Generalization in Multi-Task Manipulation [[paper]](https://ieeexplore.ieee.org/document/9667188)
- Meta-Reinforcement Learning via Language Instructions [[paper]](https://ieeexplore.ieee.org/document/10160626) [[code]](https://github.com/yaoxt3/MILLION)
- Learning from Symmetry: Meta-Reinforcement Learning with Symmetrical Behaviors and Language Instructions [[paper]](https://ieeexplore.ieee.org/document/10341769) [[code]]
- Natural Language Instruction-following with Task-related Language Development and Translation [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/1dc2fe8d9ae956616f86bab3ce5edc59-Paper-Conference.pdf)
- Task-Oriented Language Grounding for Robot via Learning Object Mask [[paper]](https://ieeexplore.ieee.org/document/10911272)
- Preserving and combining knowledge in robotic lifelong reinforcement learning [[paper]](https://www.nature.com/articles/s42256-025-00983-2)
- FLaRe: Achieving Masterful and Adaptive Robot Policies with Large-Scale Reinforcement Learning Fine-Tuning [[paper]](https://ieeexplore.ieee.org/document/11127934) [[code]](https://github.com/JiahengHu/FLaRe) 
- Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance [[paper]](https://openreview.net/forum?id=6FGlpzC9Po) [[code]](https://github.com/nakamotoo/V-GPS)
- LIMT: Language-Informed Multi-Task Visual World Models [[paper]](https://ieeexplore.ieee.org/document/11128817)


### 5.2 Behavioral Cloning
- Pay Attention! - Robustifying a Deep Visuomotor Policy Through Task-Focused Visual Attention [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Abolghasemi_Pay_Attention_-_Robustifying_a_Deep_Visuomotor_Policy_Through_Task-Focused_CVPR_2019_paper.html) 
- Language-Conditioned Imitation Learning for Robot Manipulation Tasks [[paper]](https://proceedings.neurips.cc/paper/2020/file/9909794d52985cbc5d95c26e31125d1a-Paper.pdf) 
- CLIPORT: What and Where Pathways for Robotic Manipulation [[paper]](https://proceedings.mlr.press/v164/shridhar22a/shridhar22a.pdf) [[code]](https://cliport.github.io/)
- Language Conditioned Imitation Learning over Unstructured Data [[paper]](https://www.roboticsproceedings.org/rss17/p047.pdf) [[code]](https://language-play.github.io/)
- BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning [[paper]](https://proceedings.mlr.press/v164/jang22a/jang22a.pdf) [[code]](https://www.kaggle.com/datasets/google/bc-z-robot)
- MimicPlay: Long-Horizon Imitation Learning by Watching Human Play [[paper]](https://openreview.net/pdf/b5fa74c8444234a8be8561eefeff3103bad2be96.pdf) [[code]](https://github.com/j96w/MimicPlay)
- Instruction-driven history-aware policies for robotic manipulations [[paper]](https://openreview.net/pdf?id=h0Yb0U_-Tki) [[code]](https://github.com/vlc-robot/hiveformer)
- PERCEIVER-ACTOR: A Multi-Task Transformer for Robotic Manipulation [[paper]](https://proceedings.mlr.press/v205/shridhar23a/shridhar23a.pdf) [[code]](https://github.com/peract/peract)
- Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation [[paper]](https://proceedings.mlr.press/v229/gervet23a/gervet23a.pdf) [[code]](https://github.com/zhouxian/act3d-chained-diffuser)
- GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields [[paper]](https://proceedings.mlr.press/v229/ze23a/ze23a.pdf) [[code]](https://github.com/YanjieZe/GNFactor)
- RVT: Robotic View Transformer for 3D Object Manipulation [[paper]](https://proceedings.mlr.press/v229/goyal23a/goyal23a.pdf) [[code]](https://github.com/nvlabs/rvt)
- Contrastive Imitation Learning for Language-guided Multi-Task Robotic Manipulation [[paper]](https://openreview.net/forum?id=9HkElMlPbU) [[code]](https://github.com/TeleeMa/Sigma-Agent)
- What Matters in Language Conditioned Robotic Imitation Learning Over Unstructured Data [[paper]](https://ieeexplore.ieee.org/document/9849097) [[code]](https://github.com/lukashermann/hulc)
- Grounding Language with Visual Affordances over Unstructured Data [[paper]](https://ieeexplore.ieee.org/document/10160396) [[code]](https://github.com/mees/hulc2)
- Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware [[paper]](https://www.roboticsproceedings.org/rss19/p016.pdf) [[code]](https://github.com/tonyzhaozh/aloha)
- RoboAgent: Generalization and Efficiency in Robot Manipulation via
Semantic Augmentations and Action Chunking [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10611293) [[code]](https://github.com/robopen/roboagent/)
- SRT-H: A Hierarchical Framework for Autonomous Surgery via Language Conditioned Imitation Learning [[paper]](https://arxiv.org/abs/2505.10251)



### 5.3 Diffusion-based Policy
- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion [[paper]](https://diffusion-policy.cs.columbia.edu/diffusion_policy_ijrr.pdf) [[code]](https://github.com/real-stanford/diffusion_policy)
- Imitating Human Behaviour with Diffusion Models [[paper]](https://openreview.net/forum?id=Pv1GPQzRrC8) [[code]](https://github.com/microsoft/Imitating-Human-Behaviour-w-Diffusion)
- Movement Primitive Diffusion: Learning Gentle Robotic Manipulation of Deformable Objects [[paper]](https://ieeexplore.ieee.org/document/10480552)
- Octo: An Open-Source Generalist Robot Policy [[paper]](https://www.roboticsproceedings.org/rss20/p090.pdf) [[code]](https://github.com/octo-models/octo)
- Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics [[paper]](https://www.roboticsproceedings.org/rss20/p096.pdf)
- The Ingredients for Robotic Diffusion Transformers [[paper]](https://dit-policy.github.io/resources/paper.pdf) [[code]](https://github.com/sudeepdasari/dit-policy)
- ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation [[paper]](https://proceedings.mlr.press/v229/xian23a.html) [[code]](https://github.com/zhouxian/act3d-chained-diffuser)
- DNAct: Diffusion Guided Multi-Task 3D Policy Learning [[paper]](https://arxiv.org/abs/2403.04115) 
- Render and Diffuse: Aligning Image and Action Spaces for Diffusion-based Behaviour Cloning [[paper]](https://arxiv.org/abs/2405.18196) 
- 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations [[paper]](https://roboticsconference.org/2024/program/papers/67/) [[code]](https://github.com/YanjieZe/3D-Diffusion-Policy)
- Diffusion Co-Policy for Synergistic Human-Robot Collaborative Tasks [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10310116) 
- Inference-Time Policy Steering Through Human Interactions [[paper]](https://ieeexplore.ieee.org/document/11127931) [[code]](https://github.com/yanweiw/itps) 
- Pick-and-place Manipulation Across Grippers Without Retraining: A Learning-optimization Diffusion Policy Approach [[paper]](https://arxiv.org/abs/2502.15613) [[code]](https://github.com/yaoxt3/GADP)
- Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation [[paper]](https://www.roboticsproceedings.org/rss21/p052.pdf) [[code]](https://github.com/xiaoxiaoxh/reactive_diffusion_policy)
- Generative Skill Chaining: Long-Horizon Skill Planning with Diffusion Models [[paper]](https://proceedings.mlr.press/v229/mishra23a/mishra23a.pdf) [[code]](https://github.com/generative-skill-chaining/gsc-code)
- Goal-Conditioned Imitation Learning using Score-based Diffusion Policies [[paper]](https://www.roboticsproceedings.org/rss19/p028.pdf) [[code]](https://github.com/intuitive-robots/beso)
- PlayFusion: Skill Acquisition via Diffusion from Language-Annotated Play [[paper]](https://proceedings.mlr.press/v229/chen23c/chen23c.pdf) 
- Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition [[paper]](https://proceedings.mlr.press/v229/ha23a/ha23a.pdf) [[code]](https://github.com/real-stanford/scalingup)
- Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals [[paper]](https://www.roboticsproceedings.org/rss20/p121.pdf) [[code]](https://github.com/intuitive-robots/mdt_policy)
- DISCO: Language-Guided Manipulation With Diffusion Policies and Constrained Inpainting [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11106699) 
- 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations [[paper]](https://openreview.net/forum?id=gqCQxObVz2&noteId=gqCQxObVz2) [[code]](https://github.com/nickgkan/3d_diffuser_actor)
- Language Control Diffusion: Efficiently Generalizing through Space, Time, and Tasks [[paper]](https://proceedings.iclr.cc/paper_files/paper/2024/file/e04dac2533abd18724142190530d556c-Paper-Conference.pdf) [[code]](https://github.com/ezhang7423/language-control-diffusion)
- PlayFusion: Skill Acquisition via Diffusion from Language-Annotated Play [[paper]](https://proceedings.mlr.press/v229/chen23c/chen23c.pdf) 
- Rethinking Mutual Information for Language Conditioned Skill Discovery on Imitation Learning [[paper]](https://arxiv.org/abs/2402.17511) 
- Discrete Policy: Learning Disentangled Action Space for Multi-Task Robotic Manipulation [[paper]](https://ieeexplore.ieee.org/document/11127630) 
- StructDiffusion: Language-Guided Creation of Physically-Valid Structures using Unseen Objects [[paper]](https://www.roboticsproceedings.org/rss19/p031.pdf) [[code]](https://github.com/StructDiffusion/StructDiffusion)
- PoCo: Policy Composition from and for Heterogeneous Robot Learning [[paper]](https://www.roboticsproceedings.org/rss20/p127.pdf)
- RoLD: Robot Latent Diffusion for Multi-task Policy Modeling [[paper]](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_25) [[code]](https://github.com/AlbertTan404/RoLD)
- GR-MG: Leveraging Partially-Annotated Data via Multi-Modal Goal-Conditioned Policy [[paper]](https://ieeexplore.ieee.org/document/10829675) [[code]](https://github.com/bytedance/GR-MG/tree/main)

## 6. Language for Cognitive Planning and Reasoning

### 6.1 Neuro-symbolic Approaches

- Hierarchical understanding in robotic manipulation: A knowledge-based framework [[paper]](https://www.mdpi.com/2076-0825/13/1/28)
- Semantic Grasping Via a Knowledge Graph of Robotic Manipulation: A Graph Representation Learning Approach [[paper]](https://ieeexplore.ieee.org/iel7/7083369/7339444/09830861.pdf)
- Knowledge Acquisition and Completion for Long-Term Human-Robot Interactions using Knowledge Graph Embedding [[paper]](https://arxiv.org/pdf/2301.06834)

- Tell me dave: Context-sensitive grounding of natural language to manipulation instructions [[paper]](https://www.semanticscholar.org/paper/Tell-me-Dave%3A-Context-sensitive-grounding-of-to-Misra-Sung/8cb52a0424992807dceeaf2af740364b2e80c438)
- Neuro-symbolic procedural planning with commonsense prompting [[paper]](https://arxiv.org/abs/2206.02928)
- Reinforcement Learning Based Navigation with Semantic Knowledge of Indoor Environments [[paper]](https://ieeexplore.ieee.org/abstract/document/8919366/?casa_token=7x7LciTVSGYAAAAA:Ou51YDO9Zz6Ozk_7XTjvhdlW2IL5gOv8g9XK5tlrTOLvE2bRsuZvD2E7MRSCyIZ4c2zm-EvDJSI)
- Learning Neuro-Symbolic Skills for Bilevel Planning [[paper]](Learning Neuro-Symbolic Skills for Bilevel Planning)

- Learning Neuro-symbolic Programs for Language Guided Robot Manipulation [[paper]](https://arxiv.org/abs/2211.06652) [[code]](https://github.com/dair-iitd/nsrmp) 
- Long-term robot manipulation task planning with scene graph and semantic knowledge [[paper]](https://www.emerald.com/insight/content/doi/10.1108/RIA-09-2022-0226/full/html)
- Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition [[paper]](https://proceedings.mlr.press/v229/ha23a/ha23a.pdf) [[code]](https://github.com/real-stanford/scalingup)
- ProgPrompt: program generation for situated robot task planning using large language models [[paper]](https://link.springer.com/article/10.1007/s10514-023-10135-3)
- Data-Agnostic Robotic Long-Horizon Manipulation with Vision-Language-Guided Closed-Loop Feedback [[paper]](https://www.arxiv.org/abs/2503.21969v1) [[code]](https://github.com/Ghiara/DAHLIA)
- Growing with Your Embodied Agent: A Human-in-the-Loop Lifelong Code Generation Framework for Long-Horizon Manipulation Skills [[paper]](https://arxiv.org/abs/2509.18597) [[code]]()
### 6.2 Empowered by LLMs 

#### 6.2.1 Planning
##### 6.2.1.1 Open-loop Planning
- Do As I Can, Not As I Say: Grounding Language in Robotic Affordances [[paper]](https://proceedings.mlr.press/v205/ichter23a/ichter23a.pdf) [[code]](https://say-can.github.io/)
- Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners [[paper]](https://proceedings.mlr.press/v229/ren23a/ren23a.pdf) [[code]](https://github.com/google-research/google-research/tree/master/language_model_uncertainty)
- Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents [[paper]](Language Models as Zero-Shot Planners:
Extracting Actionable Knowledge for Embodied Agents) [[code]](https://github.com/huangwl18/language-planner)
- Embodied Task Planning with Large Language Models [[paper]](https://arxiv.org/abs/2307.01848) [[code]](https://github.com/Gary3410/TaPA)

##### 6.2.1.2 Closed-loop Planning
- Text2Motion: from natural language instructions to feasible plans [[paper]](https://link.springer.com/article/10.1007/s10514-023-10131-7) 
- AlphaBlock: Embodied Finetuning for Vision-Language Reasoning in Robot Manipulation [[paper]](https://arxiv.org/pdf/2305.18898) 
- Learning to reason over scene graphs: a case study of finetuning GPT-2 into a robot language model for grounded task planning [[paper]](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1221739/full) 
- SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning [[paper]](https://proceedings.mlr.press/v229/rana23a.html)
- Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition [[paper]](https://proceedings.mlr.press/v229/ha23a/ha23a.pdf) [[code]](https://github.com/real-stanford/scalingup)
- Inner Monologue: Embodied Reasoning through Planning with Language Models [[paper]](https://proceedings.mlr.press/v205/huang23c.html) 
-  Language Models as Zero-Shot Trajectory Generators [[paper]](https://ieeexplore.ieee.org/document/10549793) [[code]](https://github.com/kwonathan/language-models-trajectory-generators)
- SELP: Generating Safe and Efficient Task Plans for Robot Agents with Large Language Models [[paper]](https://arxiv.org/abs/2409.19471) [[code]](https://github.com/lt-asset/selp)
- Human–robot interaction through joint robot planning with large language models [[paper]](https://link.springer.com/article/10.1007/s11370-024-00570-1) 
#### 6.2.2 Reasoning
##### 6.2.2.1 Summarization
- Rearrangement: A Challenge for Embodied AI [[paper]](https://arxiv.org/pdf/2011.01975) [[code]](https://github.com/haosulab/CSE291-G00-SP20)
- The ThreeDWorld Transport Challenge: A Visually Guided Task-and-Motion Planning Benchmark Towards Physically Realistic Embodied AI [[paper]](https://ieeexplore.ieee.org/document/9812329)
- Housekeep: Tidying Virtual Households Using Commonsense Reasoning [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-19842-7_21) [[code]](https://github.com/yashkant/housekeep)
- TidyBot: personalized robot assistance with large language models [[paper]](https://link.springer.com/article/10.1007/s10514-023-10139-z) [[code]](https://github.com/jimmyyhwu/tidybot)
##### 6.2.2.2 Eliciting reasoning via prompt engineering
- Building Cooperative Embodied Agents Modularly with Large Language Models [[paper]](https://openreview.net/forum?id=EnXJfQqy0K) [[code]](https://github.com/UMass-Embodied-AGI/CoELA)
- Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language [[paper]](https://openreview.net/forum?id=G2Q2Mh3avow) [[code]](https://github.com/google-research/google-research/tree/master/socraticmodels)
- Robotic Control via Embodied Chain-of-Thought Reasoning [[paper]](https://openreview.net/forum?id=S70MgnIA0v) [[code]](https://github.com/MichalZawalski/embodied-CoT/)
- Training Strategies for Efficient Embodied Reasoning [[paper]](https://arxiv.org/abs/2505.08243)
-  Scaling up and distilling down: Language-guided robot skill acquisition [[paper]](https://proceedings.mlr.press/v229/ha23a/ha23a.pdf) [[code]](https://github.com/real-stanford/scalingup)
- 

##### 6.2.2.3 Code-generation
- Voyager: An Open-Ended Embodied Agent with Large Language Models [[paper]](https://neurips.cc/virtual/2023/83083) 
- Code as Policies: Language Model Programs for Embodied Control [[paper]](https://ieeexplore.ieee.org/document/10160591) [[code]](https://github.com/google-research/google-research/tree/master/code_as_policies)
- ProgPrompt: program generation for situated robot task planning using large language models [[paper]](https://link.springer.com/article/10.1007/s10514-023-10135-3) [[code]](https://github.com/NVlabs/progprompt-vh)
- EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/4ec43957eda1126ad4887995d05fae3b-Paper-Conference.pdf) [[code]](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch)
- Alchemist: LLM-Aided End-User Development of Robot Applications [[paper]](https://ieeexplore.ieee.org/document/10660797)
- RoboCodeX: Multimodal Code Generation for Robotic Behavior Synthesis [[paper]](https://proceedings.mlr.press/v235/mu24a.html) [[code]](https://github.com/RoboCodeX-source/RoboCodeX_code)
- Data-Agnostic Robotic Long-Horizon Manipulation with Vision-Language-Guided Closed-Loop Feedback [[paper]](https://www.arxiv.org/abs/2503.21969v1) [[code]](https://ghiara.github.io/DAHLIA/)
- Growing with Your Embodied Agent: A Human-in-the-Loop Lifelong Code Generation Framework for Long-Horizon Manipulation Skills [[paper]](https://openreview.net/forum?id=1su9RkTVT9)
##### 6.2.2.4 Iterative reasoning
- Inner Monologue: Embodied Reasoning through Planning with Language Models [[paper]](https://proceedings.mlr.press/v205/huang23c.html)
- REFLECT: Summarizing Robot Experiences for FaiLure Explanation and CorrecTion [[paper]](https://proceedings.mlr.press/v229/liu23g/liu23g.pdf) [[code]](https://github.com/real-stanford/reflect)
- HiCRISP: An LLM-based Hierarchical Closed-Loop Robotic Intelligent
Self-Correction Planner [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10865457) [[code]](https://github.com/ming-bot/HiCRISP)
- Autonomous Interactive Correction MLLM for Robust Robotic Manipulation [[paper]](https://proceedings.mlr.press/v270/xiong25a.html) [[code]](https://sites.google.com/view/aic-mllm)
- Self-Corrected Multimodal Large Language Model for End-to-End Robot Manipulation [[paper]](https://arxiv.org/abs/2405.17418v1)
#### 6.2.3 LLMs-driven structured planning
##### 6.2.3.1 Combining with symbolic system
- Neuro-Symbolic Procedural Planning with Commonsense Prompting [[paper]](https://openreview.net/forum?id=iOc57X9KM54) [[code]](https://github.com/YujieLu10/CLAP)
- Hierarchical Understanding in Robotic Manipulation: A Knowledge-Based Framework [[paper]](https://www.mdpi.com/2076-0825/13/1/28)
- RoboEXP: Action-Conditioned Scene Graph via Interactive Exploration for Robotic Manipulation [[paper]](https://openreview.net/pdf?id=UHxPZgK33I) [[code]](https://github.com/Jianghanxiao/RoboEXP)
- Robot Task Planning and Situation Handling in Open Worlds [[paper]](https://arxiv.org/abs/2210.01287) [[code]](https://github.com/yding25/GPT-Planner)
- Translating Natural Language to Planning Goals with Large-Language Models [[paper]](https://arxiv.org/abs/2302.05128) 
- A framework for neurosymbolic robot action planning using large language models [[paper]](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1342786/full) [[code]](https://github.com/alessiocpt/teriyaki)
- Instruction-Augmented Long-Horizon Planning: Embedding Grounding Mechanisms in Embodied Mobile Manipulation [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33610) [[code]](https://nicehiro.github.io/IALP/#)
- LEMMo-Plan: LLM-Enhanced Learning from Multi-Modal Demonstration for Planning Sequential Contact-Rich Manipulation Tasks [[paper]](https://ieeexplore.ieee.org/document/11127842) 
- Bootstrapping Object-Level Planning with Large Language Models [[paper]](https://ieeexplore.ieee.org/document/11127365) [[code]](https://github.com/davidpaulius/olp_llm)

##### 6.2.3.2 Combining LLMs with behavior trees
- A survey of Behavior Trees in robotics and AI [[paper]](https://www.sciencedirect.com/science/article/pii/S0921889022000513?via%3Dihub) 
- LLM-BT: Performing Robotic Adaptive Tasks based on Large Language Models and Behavior Trees [[paper]](https://ieeexplore.ieee.org/document/10610183) [[code]](https://github.com/henryhaotian/LLM-BT)
- Integrating Intent Understanding and Optimal Behavior Planning for Behavior Tree Generation from Human Instructions [[paper]](https://www.ijcai.org/proceedings/2024/0755.pdf) [[code]](https://github.com/DIDS-EI/LLM-OBTEA)
- Automatic Behavior Tree Expansion with LLMs for Robotic Manipulation [[paper]](https://ieeexplore.ieee.org/document/11127942) [[code]](https://github.com/jstyrud/BETR-XP-LLM)
- LLM-as-BT-Planner: Leveraging LLMs for Behavior Tree Generation in Robot Task Planning [[paper]](https://ieeexplore.ieee.org/document/11128454) [[code]](https://github.com/ProNeverFake/kios)
### 6.3 Empowered by VLMs
#### 6.3.1 Contrastive Learning
- CLIPORT: What and Where Pathways for Robotic Manipulation [[paper]](https://proceedings.mlr.press/v164/shridhar22a/shridhar22a.pdf) [[code]](https://github.com/cliport/cliport)
- Dream2Real: Zero-Shot 3D Object Rearrangement with Vision-Language Models [[paper]](https://ieeexplore.ieee.org/document/10611220) [[code]](https://github.com/FlyCole/Dream2Real)
- CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory [[paper]](https://www.roboticsproceedings.org/rss19/p074.pdf) [[code]](https://github.com/notmahi/clip-fields)
- Simple but Effective: CLIP Embeddings for Embodied AI [[paper]](https://ieeexplore.ieee.org/document/9879778) [[code]](https://github.com/allenai/embodied-clip)
- Instruct2Act: Mapping Multi-modality Instructions to Robotic Actions with Large Language Model [[paper]](https://arxiv.org/abs/2305.11176) [[code]](https://github.com/OpenGVLab/Instruct2Act)
- Language Reward Modulation for Pretraining Reinforcement Learning [[paper]](https://arxiv.org/abs/2308.12270) [[code]](https://github.com/ademiadeniji/lamp)
- R3M: A Universal Visual Representation for Robot Manipulation [[paper]](https://proceedings.mlr.press/v205/nair23a/nair23a.pdf) [[code]](https://github.com/facebookresearch/r3m)
- Open-World Object Manipulation using Pre-Trained Vision-Language Models [[paper]](https://proceedings.mlr.press/v229/stone23a/stone23a.pdf)
- Simple Open-Vocabulary Object Detection with Vision Transformers [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700714.pdf) [[code]](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
- Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models [[paper]](https://www.roboticsproceedings.org/rss19/p029.pdf) 
#### 6.3.2 Generative Approaches
##### 6.3.2.1 Text Generation
- Pretrained Language Models as Visual Planners for Human Assistance [[paper]](https://ieeexplore.ieee.org/document/10377131) [[code]](https://github.com/facebookresearch/vlamp)
- Learning Universal Policies via Text-Guided Video Generation [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/1d5b9233ad716a43be5c0d3023cb82d0-Paper-Conference.pdf) 
- Learning to reason over scene graphs: a case study of finetuning GPT-2 into a robot language model for grounded task planning [[paper]](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1221739/full) 
- RoboPoint: A Vision-Language Model for Spatial Affordance Prediction in Robotics [[paper]](https://openreview.net/forum?id=GVX6jpZOhU&noteId=GVX6jpZOhU) [[code]](https://github.com/wentaoyuan/RoboPoint)
- Scaling Robot Policy Learning via Zero-Shot Labeling with Foundation Models [[paper]](https://arxiv.org/abs/2410.17772) [[code]](https://github.com/intuitive-robots/NILS)
- PaLM-E: An Embodied Multimodal Language Model [[paper]](https://proceedings.mlr.press/v202/driess23a/driess23a.pdf)
- Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language [[paper]](https://openreview.net/forum?id=G2Q2Mh3avow) [[code]](https://github.com/google-research/google-research/tree/master/socraticmodels)
- PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs [[paper]](https://openreview.net/pdf/c3e6b7c07bd8a22c950e8a6036aeecf048f9588a.pdf) 
##### 6.3.2.2 Image Generation
- DALL-E-Bot: Introducing Web-Scale Diffusion Models to Robotics [[paper]](https://ieeexplore.ieee.org/document/10114570) 
- Zero-Shot Robotic Manipulation with Pre-Trained Image-Editing Diffusion Models [[paper]](https://openreview.net/forum?id=c0chJTSbci) [[code]](https://github.com/kvablack/susie)
- Semantically controllable augmentations for generalizable robot learning [[paper]](https://journals.sagepub.com/doi/10.1177/02783649241273686)
- GR-MG: Leveraging Partially-Annotated Data via Multi-Modal Goal-Conditioned Policy [[paper]](https://ieeexplore.ieee.org/document/10829675) [[code]](https://github.com/bytedance/GR-MG/tree/main)
- General Flow as Foundation Affordance for Scalable Robot Learning[[paper]](https://openreview.net/forum?id=nmEt0ci8hi&noteId=nmEt0ci8hi) [[code]](https://github.com/michaelyuancb/general_flow)
### 6.4 Vision-Language-Action (VLA) Models 
#### 6.4.1 Optimization for perception 
##### 6.4.1.2 Data sources and augmentation
- EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos [[paper]](https://arxiv.org/abs/2507.12440) [[code]](https://github.com/quincy-u/Ego_Humanoid_Manipulation_Benchmark)
- H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation [[paper]](https://arxiv.org/abs/2507.23523) 
- π0: A Vision-Language-Action Flow Model for General Robot Control [[paper]](https://arxiv.org/abs/2410.24164)
- RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation [[paper]](RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation) 
- Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation [[paper]](https://arxiv.org/abs/2508.06426) [[code]](https://github.com/Lucky-Light-Sun/shortcut-learning-in-grps)
##### 6.4.1.2 3D scene representation and grounding
- SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model [[paper]](https://arxiv.org/abs/2501.15830) [[code]](https://github.com/SpatialVLA/SpatialVLA)
- PointVLA: Injecting the 3D World into Vision-Language-Action Models [[paper]](https://arxiv.org/abs/2503.07511) 
- BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models [[paper]](https://arxiv.org/abs/2506.07961) [[code]](https://github.com/BridgeVLA/BridgeVLA)
- GeoVLA: Empowering 3D Representations in Vision-Language-Action Models [[paper]](https://arxiv.org/abs/2508.09071) 

##### 6.4.1.3 Multimodal sensing and fusion
- VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation [[paper]](https://arxiv.org/abs/2505.09577) 
- Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization [[paper]](https://arxiv.org/abs/2507.09160) 
- OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned Tactile Sensing [[paper]](https://arxiv.org/abs/2508.08706) 
- Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding [[paper]](https://arxiv.org/abs/2501.04693) [[code]](https://github.com/fuse-model/FuSe)
- ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation [[paper]](https://openreview.net/forum?id=2845H8Ua5D) [[code]](https://github.com/ft-robotic/ForceVLA/)

#### 6.4.2 Optimization for reasoning
##### 6.4.2.1 Long-horizon planning
- LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon Embodied Tasks [[paper]](https://arxiv.org/abs/2506.00411) 
- DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control [[paper]](https://arxiv.org/abs/2502.05855) [[code]](https://github.com/juruobenruo/DexVLA)
- Long-VLA: Unleashing Long-Horizon Capability of Vision Language Action Model for Robot Manipulation [[paper]](https://arxiv.org/abs/2508.19958) 
- MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation [[paper]](https://arxiv.org/abs/2508.19236)

##### 6.4.2.2 Preserving foundational VLM capabilities
- Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning [[paper]](https://arxiv.org/abs/2412.03293) 
- ChatVLA-2: Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge [[paper]](https://arxiv.org/abs/2505.21906)
- Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better [[paper]](https://arxiv.org/abs/2505.23705) 
- InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation [[paper]](https://arxiv.org/abs/2507.17520) 
- GR-3 Technical Report [[paper]](https://arxiv.org/abs/2507.15493) 

##### 6.4.2.2 Leveraging world model
- Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation [[paper]](https://arxiv.org/abs/2412.15109) [[code]](https://github.com/InternRobotics/Seer)
- CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models [[paper]](https://arxiv.org/abs/2503.22020) 
- WorldVLA: Towards Autoregressive Action World Model [[paper]](https://arxiv.org/abs/2506.21539) [[code]](https://github.com/alibaba-damo-academy/RynnVLA-002)
- DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge [[paper]](https://arxiv.org/abs/2507.04447) [[code]](https://github.com/Zhangwenyao1/DreamVLA)
#### 6.4.3 Optimization for action
- π0: A vision-language-action flow model for general robot control. [[paper]](https://arxiv.org/abs/2410.24164) 
- FAST: Efficient Action Tokenization for Vision-Language-Action Models [[paper]](https://arxiv.org/abs/2501.09747) [[code]](https://huggingface.co/physical-intelligence/fast)
- π0.5: a Vision-Language-Action Model with Open-World Generalization [[paper]](https://arxiv.org/abs/2504.16054)
- Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies [[paper]](https://arxiv.org/abs/2508.20072) 
#### 6.4.4 Optimization for learning and adaption
- Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success [[paper]](https://arxiv.org/abs/2502.19645) [[code]](https://github.com/moojink/openvla-oft)
- ControlVLA: Few-shot Object-centric Adaptation for Pre-trained Vision-Language-Action Models [[paper]](https://arxiv.org/abs/2506.16211) [[code]](https://github.com/ControlVLA/ControlVLA)
- ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy [[paper]](https://arxiv.org/abs/2502.05450) 
- Interactive Post-Training for Vision-Language-Action Models [[paper]](https://arxiv.org/abs/2505.17016)
- Reinforcement Learning for Long-Horizon Interactive LLM Agents [[paper]](https://arxiv.org/abs/2502.01600)


| Optimization (Direction) | Article | Time | Observation | Action Generation | CoT | FP | MEM | MD | Pretraining CE | Scenarios MS | Scenarios RW | Execution CE |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Data Source Augmentation | [EgoVLA](https://arxiv.org/abs/2507.12440) | 2025-07 | RGB, ROB, TX | DP | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| | [H-RDT](https://arxiv.org/abs/2507.23523) | 2025-08 | RGB, ROB, TX | FM | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [Shortcut](https://arxiv.org/abs/2508.06426) | 2025-08 | RGB, TX | - | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Spatial Understanding | [SpatialVLA](https://arxiv.org/abs/2501.15830) | 2025-01 | RGB, ROB, TX | AR | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [PointVLA](https://arxiv.org/abs/2503.07511) | 2025-05 | RGBD, ROB, TX | DM | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [BridgeVLA](https://arxiv.org/abs/2506.07961) | 2025-06 | RGB, ROB, TX | DP | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| | [GeoVLA](https://arxiv.org/abs/2508.09071) | 2025-08 | RGBD, ROB, TX | DM | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Multimodal Sensing & Fusion | [VTLA](https://arxiv.org/abs/2505.09577) | 2025-05 | RGB, TX | AR | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| | [Tactile-VLA](https://arxiv.org/abs/2507.09160) | 2025-07 | RGB, ROB, TX | FM | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| | [OmniVTLA](cheng2https://arxiv.org/abs/2508.08706025omnivtla) | 2025-08 | RGB, ROB, TX | FM | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| | [ForceVLA](https://arxiv.org/abs/2505.22159) | 2025-09 | RGBD, ROB, TX | FM | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| | [FuSe VLA](https://arxiv.org/abs/2501.04693) | 2025-01 | RGBD, ROB, TX | AR | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Long-horizon task solving | [LoHoVLA](yang2025lohovla) | 2025-05 | RGB, ROB, TX | AR | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| | [Long-VLA](https://arxiv.org/abs/2508.19958) | 2025-08 | RGB, TX | DM | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| | [DexVLA](https://arxiv.org/abs/2502.05855) | 2025-08 | RGB, ROB, TX | DM | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [MemoryVLA](https://arxiv.org/abs/2508.19236) | 2025-08 | RGB, TX | DM | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [DiffusionVLA](https://openreview.net/forum?id=VdwdU81Uzy&noteId=KMyfyxiVhr) | 2024-12 | RGBD, TX | DM + AR | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Knowledge Preserving | [ChatVLA](https://arxiv.org/abs/2502.14420) | 2025-02 | RGB, TX | DM | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| | [ChatVLA-2](https://arxiv.org/abs/2505.21906) | 2025-05 | RGB, TX | DM | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| | [Insulating](https://openreview.net/forum?id=cb0xbZ3APM) | 2025-05 | RGB, ROB, TX | FM + AR | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [InstructVLA](https://arxiv.org/abs/2507.17520) | 2025-07 | RGB, ROB, TX | FM | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| | [GR-3](https://arxiv.org/abs/2507.15493) | 2025-07 | RGB, ROB, TX | FM | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| Reasoning & World Models | [Seer](https://arxiv.org/abs/2412.15109) | 2024-12 | RGB, ROB, TX | DP | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [CoT-VLA](https://arxiv.org/abs/2503.22020) | 2025-05 | RGB, ROB, TX | AR | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [WorldVLA](https://arxiv.org/abs/2506.21539) | 2025-06 | RGB, ROB, TX | AR | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| | [DreamVLA](https://openreview.net/forum?id=PK07eretkF&referrer=%5Bthe%20profile%20of%20Runpei%20Dong%5D(%2Fprofile%3Fid%3D~Runpei_Dong1)) | 2025-08 | RGB, ROB, TX | DM | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| | [ECoT VLA] (https://openreview.net/forum?id=S70MgnIA0v) | 2024-07 | RGB, TX | AR | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| Policy Execution | [PI-0](https://arxiv.org/abs/2410.24164) | 2024-10 | RGB, ROB, TX | FM | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [PI-Fast](https://arxiv.org/abs/2501.09747) | 2025-01 | RGB, ROB, TX | FM | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [PI-0.5](https://arxiv.org/abs/2504.16054) | 2025-04 | RGB, ROB, TX | FM | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [DisDiffVLA](https://arxiv.org/abs/2508.20072) | 2025-08 | RGB, ROB, TX | DM | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Adaptation & Fine-Tuning | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | 2025-02 | RGB, ROB, TX | DP | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | [ConRFT](https://arxiv.org/abs/2502.05450) | 2025-04 | RGB, ROB, TX | DM | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| | [RIPT-VLA](https://arxiv.org/abs/2505.17016) | 2025-05 | RGB, ROB, TX | AR | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| | [ControlVLA](https://arxiv.org/abs/2506.16211) | 2025-06 | RGB, ROB, TX | DM | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

**Notes**
- Observation: RGB images; D = depth information (e.g., point clouds); ROB = robot proprioception; TX = text (e.g., prompt, language goal).
- Action generation: FM = Flow Matching; DM = Diffusion Model; AR = Autoregressive; DP = Direct Prediction; - = Not Applicable.
- CoT = Chain-of-Thought; FP = Future Prediction; MEM = Memory Mechanisms; MD = Multiple Datasets.
- Pretraining CE = Cross-embodiment Data; MS = Multi-scenario; RW = Real-world Deployment; Execution CE = Cross-embodiment Execution.

## Comparative Analysis
### Benchmarks
| Benchmark          | Simulation Engine or Real-world Dataset | Embodiment      | Data Size | RGB | Depth | Masks | Tool used | Multi-agents | Long-horizon |
|--------------------|-----------------------------------------|-----------------|----------:|:---:|:------:|:-----:|:---------:|:------------:|:------------:|
| CALVIN             | PyBullet                                | Franka Panda    | 2400k     | ✅  | ✅    | ❌    | ❌        | ❌           | ✅           |
| Meta-world         | MuJoCo                                  | Sawyer          | -         | ✅  | ❌    | ❌    | ❌        | ❌           | ❌           |
| RLBench            | CoppeliaSim                             | Franka Panda    | -         | ✅  | ✅    | ✅    | ❌        | ❌           | ✅           |
| VIMAbench          | PyBullet                                | UR5             | 650k      | ✅  | ❌    | ❌    | ❌        | ❌           | ✅           |
| LoHoRavens         | PyBullet                                | UR5             | 15k       | ✅  | ✅    | ❌    | ❌        | ❌           | ✅           |
| ARNOLD             | NVIDIA Omniverse                        | Framka Panda    | 10k       | ✅  | ✅    | ✅    | ❌        | ❌           | ✅           |
| RoboGen            | PyBullet                                | Multiple        | -         | ✅  | ❌    | ❌    | ✅        | ✅           | ✅           |
| LIBERO             | MuJoCo                                  | Franka Panda    | 6.5k      | ✅  | ❌    | ❌    | ❌        | ❌           | ✅           |
| Open X-Embodiment  | Real-world Dataset                      | Multiple        | 2419k     | ✅  | ✅    | ❌    | ❌        | ❌           | ✅           |
| DROID              | Real-world Dataset                      | Franka Panda    | 76k       | ✅  | ✅    | ❌    | ❌        | ❌           | ✅           |
| Galaxea Open-world | Real-world Dataset                      | Galaxea R1 Lite | 100k      | ✅  | ✅    | ❌    | ❌        | ❌           | ✅           |

### Models

| Models | Years | Benchmark | Simulation Engine | Language Module | Perception Module | Real world experiments | FMs | RL | IL | MP |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [IPRO](https://ieeexplore.ieee.org/document/8460699) | 2018 | # | - | LSTM | CNN | ✅ | ❌ | ❌ | ❌ | ✅ |
| [MaestROB](https://ieeexplore.ieee.org/document/8462870) | 2018 | # | - | IBM Watson | Artoolkit | ✅ | ❌ | ❌ | ❌ | ✅ |
| [exePlan](https://www.sciencedirect.com/science/article/abs/pii/S0950705117304926) | 2018 | # | - | coreNLP | * | ✅ | ❌ | ❌ | ❌ | ✅ |
| [TLC](https://ieeexplore.ieee.org/document/8403899) | 2018 | # | - | CCG | * | ✅ | ❌ | ❌ | ✅ | ❌ |
| [Cut\&recombine](https://journals.sagepub.com/doi/full/10.1177/0278364919865594)  | 2019 | # | - | Parser | * | ✅ | ❌ | ❌ | ❌ | ✅ |
| [DREAMCELL](https://ieeexplore.ieee.org/document/8794441)  | 2019 | # | - | LSTM | * | ❌ | ❌ | ❌ | ✅ | ❌ |
| [ICR](https://ttic.edu/ripl/assets/publications/patki19.pdf) | 2019 | # | - | Parser, DCG | YOLO9000 | ✅ | ❌ | ❌ | ❌ | ✅ |
| [GroundedDA](https://dl.acm.org/doi/10.1109/ICRA.2019.8794287)  | 2019 | # | - | CCG | RANSAC | ✅ | ❌ | ❌ | ❌ | ✅ |
| [MEC](https://journals.sagepub.com/doi/abs/10.1177/0278364920917755)  | 2020 | # | - | Parser, ADCG | Mask RCNN | ✅ | ❌ | ❌ | ❌ | ✅ |
| [LMCR](https://arxiv.org/abs/1904.12907) | 2020 | # | - | RNN | Mask RCNN | ✅ | ❌ | ❌ | ❌ | ✅ |
| [PixL2R](https://proceedings.mlr.press/v155/goyal21a.html) | 2020 | Meta-World | MuJoCo | LSTM | CNN | ❌ | ❌ | ✅ | ❌ | ❌ |
| [Concept2Robot](https://www.roboticsproceedings.org/rss16/p082.pdf) | 2020 | # | PyBullet | BERT | ResNet-18 | ❌ | ❌ | ❌ | ✅ | ❌ |
| [LanguagePolicy](https://proceedings.neurips.cc/paper/2020/file/9909794d52985cbc5d95c26e31125d1a-Paper.pdf)| 2020 | # | CoppeliaSim | GLoVe | Faster RCNN | ❌ | ❌ | ❌ | ✅ | ❌ |
| [LOReL](https://openreview.net/pdf?id=tfLu5W6SW5J) | 2021 | Meta-World | MuJoCo | distillBERT | CNN | ✅ | ❌ | ✅ | ❌ | ❌ |
| [CARE](https://proceedings.mlr.press/v139/sodhani21a.html) | 2021 | Meta-World | MuJoCo | RoBERTa | * | ❌ | ✅ | ✅ | ❌ | ❌ |
| [MCIL](https://www.roboticsproceedings.org/rss17/p047.pdf) | 2021 | # | MuJoCo | MUSE | CNN | ❌ | ❌ | ❌ | ✅ | ❌ |
| [BC-Z](https://proceedings.mlr.press/v164/jang22a/jang22a.pdf) | 2021 | # | - | MUSE | ResNet18 | ✅ | ❌ | ❌ | ✅ | ❌ |
| [CLIPort](https://proceedings.mlr.press/v164/shridhar22a/shridhar22a.pdf) | 2021 | # | PyBullet | CLIP | CLIP | ✅ | ❌ | ❌ | ✅ | ❌ |
| [LanCon-Learn](https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/d/958/files/2022/01/lanConLearn.pdf) | 2022 | Meta-World | MuJoCo | GLoVe | * | ❌ | ❌ | ✅ | ✅ | ❌ |
| [MILLION](https://ieeexplore.ieee.org/document/10160626) | 2022 | Meta-World | MuJoCo | GLoVe | * | ✅ | ❌ | ✅ | ❌ | ❌ |
| [PaLM-SayCan](https://proceedings.mlr.press/v205/ichter23a) | 2022 | # | - | PaLM | ViLD | ✅ | ✅ | ✅ | ✅ | ❌ |
| [ATLA](https://proceedings.mlr.press/v205/ren23a) | 2022 | # | PyBullet | BERT-Tiny | CNN | ❌ | ✅ | ✅ | ❌ | ❌ |
| [HULC](https://www.roboticsproceedings.org/rss17/p047.pdf) | 2022 | CALVIN | PyBullet | MiniLM-L3-v2 | CNN | ❌ | ❌ | ❌ | ✅ | ❌ |
| [PerAct](https://proceedings.mlr.press/v205/shridhar23a) | 2022 | RLbench | CoppelaSim | CLIP | ViT | ✅ | ❌ | ❌ | ✅ | ❌ |
| [RT-1](https://www.roboticsproceedings.org/rss19/p025.pdf) | 2022 | # | - | USE | EfficientNet-B3 | ✅ | ✅ | ❌ | ✅ | ❌ |
| [LATTE](https://arxiv.org/abs/2208.02918) | 2023 | # | CoppeliaSim | distillBERT, CLIP | CLIP | ✅ | ❌ | ❌ | ❌ | ✅ |
| [DIAL](https://www.roboticsproceedings.org/rss19/p029.pdf) | 2022 | # | - | CLIP | CLIP | ✅ | ✅ | ❌ | ✅ | ❌ |
| [R3M](https://proceedings.mlr.press/v205/nair23a/nair23a.pdf) | 2022 | # | - | distillBERT | ResNet18,34,50 | ✅ | ❌ | ❌ | ✅ | ❌ |
| [Inner Monologue](https://proceedings.mlr.press/v205/huang23c)  | 2022 | # | - | CLIP | CLIP | ✅ | ✅ | ❌ | ❌ | ✅ |
| [NLMap](https://arxiv.org/abs/2209.09874) | 2023 | # | - | CLIP | ViLD | ✅ | ✅ | ❌ | ✅ | ❌ |
| [Code as Policies](https://ieeexplore.ieee.org/document/10160591) | 2023 | # | - | GPT3, Codex | ViLD | ✅ | ✅ | ❌ | ❌ | ✅ |
| [Progprompt](https://ieeexplore.ieee.org/document/10161317) | 2023 | Virtualhome | Unity3D | GPT-3 | * | ✅ | ✅ | ❌ | ❌ | ✅ |
| [Language2Reward](https://proceedings.mlr.press/v229/yu23a.html) | 2023 | # | MuJoCo MPC | GPT-4 | * | ✅ | ✅ | ✅ | ❌ | ❌ |
| [LfS](https://ieeexplore.ieee.org/document/10341769) | 2023 | Meta-World | MuJoCo | Cons. Parser | * | ✅ | ❌ | ✅ | ❌ | ❌ |
| [HULC++](https://ieeexplore.ieee.org/document/10160396) | 2023 | CALVIN | PyBullet | MiniLM-L3-v2 | CNN | ✅ | ❌ | ❌ | ✅ | ❌ |
| [ALOHA](https://aclanthology.org/2022.emnlp-main.83/)  | 2023 | # | - | Transformer | CNN | ✅ | ❌ | ❌ | ✅ | ❌ |
| [LEMMA](https://ieeexplore.ieee.org/document/10243083) | 2023 | LEMMA | NVIDIA Omniverse | CLIP | CLIP | ❌ | ❌ | ❌ | ✅ | ❌ |
| [SPIL](https://ieeexplore.ieee.org/document/10685120) | 2023 | CALVIN | PyBullet | MiniLM-L3-v2 | CNN | ✅ | ❌ | ❌ | ✅ | ❌ |
| [PaLM-E](https://proceedings.mlr.press/v202/driess23a/driess23a.pdf)  | 2023 | # | PyBullet | PaLM | ViT | ✅ | ✅ | ❌ | ✅ | ❌ |
| [LAMP](https://arxiv.org/abs/2308.12270) | 2023 | RLbench | CoppelaSim | ChatGPT | R3M | ❌ | ✅ | ✅ | ❌ | ❌ |
| [MOO](https://proceedings.mlr.press/v229/stone23a) | 2023 | # | - | OWL-ViT | OWL-ViT | ✅ | ❌ | ❌ | ✅ | ❌ |
| [Instruction2Act](https://arxiv.org/abs/2305.11176) | 2023 | VIMAbench | PyBullet | ChatGPT | CLIP | ❌ | ✅ | ❌ | ❌ | ✅ |
| [VoxPoser](https://openreview.net/forum?id=9_8LF30mOC) | 2023 | # | SAPIEN | GPT-4 | OWL-ViT | ✅ | ✅ | ❌ | ❌ | ✅ |
| [SuccessVQA](https://proceedings.mlr.press/v232/du23b.html) | 2023 | # | IA Playroom | Flamingo | Flamingo | ✅ | ✅ | ❌ | ✅ | ❌ |
| [VIMA](https://proceedings.mlr.press/v202/jiang23b) | 2023 | VIMAbench | PyBullet | T5 | ViT | ✅ | ✅ | ❌ | ✅ | ❌ |
| [TidyBot](https://ieeexplore.ieee.org/document/10341577/) | 2023 | # | - | GPT-3 | CLIP | ✅ | ✅ | ❌ | ❌ | ✅ |
| [Text2Motion](https://link.springer.com/article/10.1007/s10514-023-10131-7) | 2023 | # | - | GPT-3, Codex | * | ✅ | ✅ | ✅ | ❌ | ❌ |
| [LLM-GROP](https://ieeexplore.ieee.org/document/10342169) | 2023 | # | Gazebo | GPT-3 | * | ✅ | ✅ | ❌ | ❌ | ✅ |
| [Scaling Up](https://proceedings.mlr.press/v229/ha23a) | 2023 | # | MuJoCo | CLIP, GPT-3 | ResNet-18 | ✅ | ✅ | ❌ | ✅ | ❌ |
| [Socratic Models](https://openreview.net/forum?id=G2Q2Mh3avow) | 2023 | # | - | RoBERTa, GPT-3 | CLIP | ✅ | ✅ | ❌ | ❌ | ✅ |
| [SayPlan](https://proceedings.mlr.press/v229/rana23a.html) | 2023 | # | - | GPT-4 | * | ✅ | ✅ | ❌ | ❌ | ✅ |
| [RT-2](https://proceedings.mlr.press/v229/zitkovich23a.html) | 2023 | # | - | PaLI-X, PaLM-E | PaLI-X, PaLM-E | ✅ | ✅ | ❌ | ✅ | ❌ |
| [KNOWNO](https://proceedings.mlr.press/v229/ren23a) | 2023 | # | PyBullet | PaLM-2L | * | ✅ | ✅ | ❌ | ❌ | ✅ |
| [MDT](https://www.roboticsproceedings.org/rss20/p121.pdf) | 2023 | CALVIN | PyBullet | CLIP | CLIP | ❌ | ❌ | ❌ | ✅ | ❌ |
| [RT-Trajectory](https://arxiv.org/abs/2311.01977) | 2023 | # | - | PaLM-E | EfficientNet-B3 | ✅ | ✅ | ❌ | ✅ | ❌ |
| [SuSIE](https://neurips.cc/virtual/2023/82816) | 2023 | CALVIN | PyBullet | InstructPix2Pix(GPT3) | InstructPix2Pix | ✅ | ✅ | ❌ | ✅ | ❌ |
| [Playfusion](https://proceedings.mlr.press/v229/chen23c/chen23c.pdf) | 2023 | CALVIN | PyBullet | Sentence-bert | ResNet-18 | ✅ | ❌ | ❌ | ✅ | ❌ |
| [ChainedDiffuser](https://proceedings.mlr.press/v229/xian23a.html) | 2023 | RLbench | CoppelaSim | CLIP | CLIP | ✅ | ❌ | ❌ | ✅ | ❌ |
| [GNFactor](https://proceedings.mlr.press/v229/ze23a) | 2023 | RLbench | CoppelaSim | CLIP | NeRF | ✅ | ❌ | ❌ | ✅ | ❌ |
| [StructDiffusion](https://www.roboticsproceedings.org/rss19/p031.pdf) | 2023 | # | PyBullet | Sentence-bert | PCT | ✅ | ❌ | ❌ | ❌ | ✅ |
| [PoCo](https://www.roboticsproceedings.org/rss20/p127.pdf) | 2024 | Fleet-Tools | Drake | T5 | ResNet-18 | ✅ | ❌ | ❌ | ✅ | ❌ |
| [DNAct](https://arxiv.org/abs/2403.04115) | 2024 | RLbench | CoppelaSim | CLIP | NeRF, PointNext | ✅ | ❌ | ❌ | ✅ | ❌ |
| [3D Diffuser Actor](https://arxiv.org/abs/2402.10885) | 2024 | CALVIN | PyBullet | CLIP | CLIP | ✅ | ❌ | ❌ | ✅ | ❌ |
| [RoboFlamingo]()  | 2024 | CALVIN | Pybullet | OpenFlamingo | OpenFlamingo | ❌ | ✅ | ❌ | ✅ | ❌ |
| [OpenVLA](https://arxiv.org/abs/2406.09246) | 2024 | Open X-Embodiment | - | Llama 2 7B | DINOv2 \& SigLIP | ✅ | ✅ | ❌ | ✅ | ❌ |
| [RT-X](https://neurips.cc/virtual/2023/77259) | 2024 | Open X-Embodiment | - | PaLI-X,PaLM-E | PaLI-X,PaLM-E | ✅ | ✅ | ❌ | ✅ | ❌ |
| [PIVOT](https://dl.acm.org/doi/10.5555/3692070.3693585) | 2024 | Open X-Embodiment | - | GPT-4, Gemini | GPT-4, Gemini | ✅ | ✅ | ❌ | ❌ | ✅ |
| [RT-Hierarchy](https://www.roboticsproceedings.org/rss20/p049.pdf) | 2024 | # | - | PaLI-X | PaLI-X | ✅ | ✅ | ❌ | ✅ | ❌ |
| [3D-VLA](https://dl.acm.org/doi/abs/10.5555/3692070.3694603) | 2024 | RL-Bench \& CALVIN | CoppeliaSim \& PyBullet | 3D-LLM | 3D-LLM | ❌ | ✅ | ❌ | ✅ | ❌ |
| [Octo](https://www.roboticsproceedings.org/rss20/p090.pdf) | 2024 | Open X-Embodiment | - | T5 | CNN | ✅ | ✅ | ❌ | ✅ | ❌ |
| [ECoT](https://arxiv.org/abs/2407.08693) | 2024 | BridgeData V2 | - | Llama 2 7B | DinoV2 \& SigLIP | ✅ | ✅ | ❌ | ✅ | ❌ |
| [LEGION](https://www.nature.com/articles/s42256-025-00983-2) | 2024 | Meta-World | MuJoCo | RoBERTa | * | ✅ | ❌ | ✅ | ❌ | ❌ |
| [RACER](https://ieeexplore.ieee.org/document/11127799) | 2024 | RLbench | CoppelaSim | Llama3-llava-next-8B | LLaVA | ✅ | ✅ | ❌ | ✅ | ❌ |
| [Ground4Act](https://www.sciencedirect.com/science/article/pii/S0262885624003858?via%3Dihub) | 2024 | # | Gazebo | Transformer | ResNet101, BERT | ✅ | ❌ | ✅ | ❌ | ❌ |
| [LOVM](https://ieeexplore.ieee.org/document/10911272) | 2024 | # | - | BiGRU | LOVM | ❌ | ❌ | ✅ | ❌ | ✅ |
| [ECLAIR](https://ieeexplore.ieee.org/document/10803055) | 2024 | # | - | GPT-3-turbo | * | ✅ | ✅ | ✅ | ❌ | ❌ |
| [PR2L](https://openreview.net/forum?id=vQDKYYuqWA) | 2024 | MineDojo | HM3D | InstructBLIP | InstructBLIP | ✅ | ✅ | ✅ | ❌ | ❌ |
| [AHA](https://openreview.net/forum?id=JVkdSi7Ekg) | 2024 | RLBench, ManiSkill | CoppeliaSim, SAPIEN | LLaMA-2-13B | CLIP | ❌ | ✅ | ✅ | ❌ | ✅ |
| [KOI](https://proceedings.mlr.press/v270/lu25a.html) | 2024 |  Meta-World, LIBERO | MuJoCo | GPT-4v | KOI | ✅ | ✅ | ❌ | ✅ | ✅ |
| [GPT-4V(ISION)](https://ieeexplore.ieee.org/document/10711245) | 2024 | # | - | GPT-4 | GPT-4 | ✅ | ✅ | ❌ | ✅ | ❌ |
| [HiRT](https://proceedings.mlr.press/v270/zhang25b.html) | 2024 | Meta-World, Franka-Kitchen | MuJoCo, | InstructBLIP | CNN | ✅ | ✅ | ❌ | ✅ | ❌ |
| [Sentinel](https://openreview.net/forum?id=yqLFb0RnDW) | 2024 | # | - | GPT-4o | PointNet++ | ✅ | ✅ | ❌ | ❌ | ✅ |
| [RoLD](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_25) | 2024 | Open X-E, Robomimic, Meta-World | -, MuJoCo | DistilBERT | DistilBERT | ❌ | ❌ | ❌ | ❌ | ✅ |
| [ITS](https://link.springer.com/article/10.1007/s10015-025-01036-y) | 2025 | * | - | LLaMA | A2C | ❌ | ✅ | ✅ | ❌ | ✅ |
| [SIAMS](https://ieeexplore.ieee.org/document/10856348) | 2025 | Miniworld ｜ Pyglet | LTL | CNN | ❌ | ✅ | ✅ | ❌ | ❌ |
| [CRTO](https://ieeexplore.ieee.org/document/10955186) | 2025 | Continual World | MuJoCo | ChatGPT | * | ❌ | ✅ | ✅ | ❌ | ✅ |
| [LAMARL](https://ieeexplore.ieee.org/abstract/document/11027664) | 2025 | # | - | OpenAI | MADDPG | ✅ | ✅ | ✅ | ❌ | ✅ |
| [ARCHIE](https://arxiv.org/abs/2503.04280) | 2025 | # | - | GPT-4 | * | ✅ | ✅ | ✅ | ❌ | ✅ |
| [RealBEF](https://www.sciencedirect.com/science/article/abs/pii/S0952197625010048) | 2025 | Meta-World | MuJoCo | ALBEF | CNN | ❌ | ✅ | ✅ | ❌ | ❌ |
| [LLMRewardShaping](https://link.springer.com/chapter/10.1007/978-981-96-0783-9_1) | 2025 | Meta-World | MuJoCo | GPT-4 | * | ✅ | ✅ | ✅ | ❌ | ❌ |
| [BOSS](https://ieeexplore.ieee.org/document/11062780)  | 2025 | LIBERO | MuJoCo | OpenVLA | ResNet | ❌ | ✅ | ❌ | ✅ | ✅ |
| [LAV-ACT](https://ieeexplore.ieee.org/document/10977578) | 2025 | # | MuJuCo | Voltron | Voltron | ✅ | ❌ | ❌ | ✅ | ❌ |
| [TPM](https://openreview.net/forum?id=SkGvesJZkU) | 2025 | # | MuJuCo | GPT-4 | ResNet | ✅ | ✅ | ❌ | ✅ | ✅ |
| [Mamba](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10966860) | 2025 | # | - | Mamba | Mamba | ✅ | ✅ | ❌ | ✅ | ✅ |
| [TransformerPolicy](https://ieeexplore.ieee.org/document/10934975) | 2025 | CALVIN | PyBullet | Transformer | Sentence-BERT | ✅ | ❌ | ❌ | ✅ | ✅ |
| [HierarchicalLCL](https://link.springer.com/chapter/10.1007/978-981-96-1614-5_11) | 2025 | CALVIN | PyBullet | OpenFlamingoM-3B | ViT | ❌ | ✅ | ❌ | ✅ | ✅ |
| [BLADE](https://openreview.net/forum?id=fR1rCXjCQX&noteId=fR1rCXjCQX) | 2025 | CALVIN | PyBullet | GPT-4 | PCT | ✅ | ✅ | ❌ | ✅ | ✅ |
| [LES6DPose](https://ieeexplore.ieee.org/document/11075556) | 2025 | # | Isaac Gym |  | GPT-4 | PointNet++ | ✅ | ✅ | ❌ | ❌ | ✅ |
| [SafetyFilter](https://ieeexplore.ieee.org/document/10933541) | 2025 | # | - | GPT-4o | CLIP | ✅ | ✅ | ❌ | ❌ | ✅ |
| [TARAD](https://ieeexplore.ieee.org/document/11124589)| 2025 | RLBench | CoppeliaSim  | GPT-4o | CLIP | ✅ | ✅ | ❌ | ✅ | ✅ |
| [DISCO](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11106699) | 2025 | CALVIN | PyBullet | GPT-4o | * | ✅ | ✅ | ❌ | ❌ | ✅ |
| [TinyVLA](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10900471) | 2025 | Meta-World | MuJoCo | Pythia | MLP | ✅ | ✅ | ❌ | ❌ | ✅ |
| [ASD-QR](https://ieeexplore.ieee.org/document/10889202) | 2025 | ScalingUp | MuJoCo | GPT3 | CLIP | ❌ | ✅ | ✅ | ❌ | ✅ |
| [RDT-1B](https://openreview.net/forum?id=yAzN4tz7oI) | 2025 | # | - | GPT-4-Turbo | T5-XXL | ✅ | ✅ | ❌ | ✅ | ❌ |
| [GRAVMAD](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10161534) | 2025 | RLBench | CoppeliaSim | GPT-4o | CLIP | ✅ | ✅ | ❌ | ✅ | ✅ |
| [GR-MG](https://ieeexplore.ieee.org/document/10829675) | 2025 | CALVIN | PyBullet| Transformer | T5-Base | ✅ | ❌ | ❌ | ❌ | ✅ |
| [LEMMo-Plan](https://arxiv.org/abs/2409.11863) | 2025 | # | - | GPT-4o | * | ✅ | ✅ | ❌ | ❌ | ✅ |
## Citation

If you find this survey or repository useful, please consider citing:

```bibtex
@article{zhou2023language,
  author       = {Hongkuan Zhou and
                  Xiangtong Yao and
                  Oier Mees and
                  Yuan Meng and
                  Ted Xiao and
                  Yonatan Bisk and
                  Jean Oh and
                  Edward Johns and
                  Mohit Shridhar and
                  Dhruv Shah and
                  Jesse Thomason and
                  Kai Huang and
                  Joyce Chai and
                  Zhenshan Bing and
                  Alois Knoll},
  title        = {Bridging Language and Action: A Survey of Language-Conditioned Robot Manipulation},
  journal      = {CoRR},
  volume       = {abs/2312.10807},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2312.10807}
}