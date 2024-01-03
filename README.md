# Language-conditioned Robot Manipulation Models

![alt text](graphs/overview.png)
This architectural framework provides an overview of language-conditioned robot manipulation. The agent comprises three key modules: the language module, the perception module, and the control module. These modules serve the functions of understanding instructions, perceiving the environment's state, and acquiring skills, respectively. The vision-language module establishes a connection between instructions and the surrounding environment to achieve a more profound comprehension of both aspects. This interplay of information from both modalities enables the robot to engage in high-level planning and perform visual question answering tasks, ultimately enhancing its overall performance. The control module has the capability to acquire low-level policies through learning from rewards (reinforcement learning) and demonstrations (imitation learning) which engineered by experts. At times, these low-level policies can also be directly designed or hard-coded, making use of path and motion planning algorithms. There are two key loops to highlight. The interactive loop, located on the left, facilitates human-robot language interaction. The control loop, positioned on the right, signifies the interaction between the agent and its surrounding environment.

## Table of the Content

- Survey Paper
- Language-conditioned Reinforcement Learning
  - Reward Function Learning
- Language-conditioned Imitation Learning
  - Behaviour Cloning
  - Inverse Reinforcement Learning
- Enpowered by LLMs
- Enpowered by VLMs
- Comparative Analysis

## Survey

This paper is basically based on the survey paper

**[Language-conditioned Learning for Robot Manipulation: A Survey](https://arxiv.org/abs/2312.10807)**
<br />
Hongkuan Zhou,
Xiangtong Yao,
Yuan Meng,
Siming Sun,
Zhenshan Bing,
Kai Huang,
Hong Qiao,
Alois Knoll
<br />

```bibtex
@article{zhou2023language,
  title={Language-conditioned Learning for Robotic Manipulation: A Survey},
  author={Zhou, Hongkuan and Yao, Xiangtong and Meng, Yuan and Sun, Siming and BIng, Zhenshan and Huang, Kai and Knoll, Alois},
  journal={arXiv preprint arXiv:2312.10807},
  year={2023}
}
```

## Language-conditioned Reinforcement Learning

## Language-conditioned Imitation Learning

## Empowered by LLMs

## Empowered by VLMs

## Comparative Analysis


| Name | Year | Benchmark | Simulation Engine | Language Module| Perception Module | Real World Experiment | LLM | Reinforcement Learning | Imitation Learning |
| ------ | ------ | :-----------: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|  [DREAMCELL](https://arxiv.org/abs/1903.08309) |  2019  |        #         | -       | LSTM | *   | ❌ | ❌ | ❌ | ✅ |
|  [PixL2R](https://proceedings.mlr.press/v155/goyal21a.html) |  2020  |    Meta-World    | MuJoCo  | LSTM | CNN | ❌ | ❌ | ✅ | ❌ |
| Concept2Robot | 2020 | # | PyBullet | BERT | ResNet-18 | ❌ | ❌ | ❌ | ✅ |
| LanguagePolicy | 2020 | # | CoppeliaSim | GLoVe | Faster RCNN | ❌ | ❌ | ❌ | ✅ |
| LOReL| 2021 | Meta-World | MuJoCo | distillBERT | CNN | ✅ | ❌ | ❌ | ✅ |
| CARE| 2021 | Meta-World | MuJoCo | RoBERTa | * | ❌ | ✅ | ✅ | ❌ |
| MCIL | 2021 | # | MuJoCo | MUSE | CNN | ❌ | ❌ | ❌ | ✅ |
| BC-Z | 2021 | # | - | MUSE | ResNet18 | ✅ | ❌ | ❌ | ✅ |
| CLIPort | 2021 | # | Pybullet | CLIP | CLIP/ResNet | ✅ | ❌ | ❌ | ✅ |
| LanCon-Learn | 2022 | Meta-World | MuJoCo | GLoVe | * | ❌ | ❌ | ✅ | ✅ | 
| MILLON | 2022 | Meta-World| MuJoCo | GLoVe | * | ✅ | ❌ | ✅ | ❌ | 
| PaLM-SayCan | 2022 | # | - | PaLM | ViLD | ✅ | ✅ | ✅ | ✅ |
| ATLA | 2022 | # | PyBullet | BERT-Tiny | CNN | ❌ | ✅ | ✅ | ❌ |
| HULC | 2022 | CALVIN | Pybullet | MiniLM-L3-v2 | CNN | ❌ | ❌ | ❌ | ✅ |
| PerAct | 2022 | RLbench | CoppelaSim | CLIP | ViT | ✅ | ❌ | ❌ | ✅ |
| RT-1 | 2022 | # | - | USE | EfficientNet-B3 |  ✅ | ✅ | ❌ | ❌ |
| DIAL | 2022 | # | - | CLIP | CLIP |  ✅ | ✅ | ❌ | ✅ |
| R3M | 2022 | # | - | distillBERT | ResNet | ✅ | ❌ | ❌ | ✅ |
| Inner Monologue | 2022 | # | - | CLIP | CLIP | ✅ | ✅ | ❌ | ❌ |
| PROGPROMPT |  2023 | Virtualhome | Unity3D | GPT-3 | * | ✅ | ✅ | ❌ | ❌ |
| Language2Reward | 2023 | # | MuJoCo MPC | GPT-4 | * | ✅ | ✅ | ✅ | ❌ |
| LfS | 2023 | Meta-World | MuJoCo | Cons. Parser | * | ✅ | ❌ | ✅ | ❌ |
| HULC++ | 2023 | CALVIN | PyBullet | MiniLM-L3-v2 | CNN | ✅ | ❌ | ❌ | ✅ |
| LEMMA | 2023 | LEMMA | NVIDIA Omniverse | CLIP | CLIP | ❌ | ❌ | ❌ | ✅ |
| SPIL| 2023 | CALVIN | PyBullet | MiniLM-L3-v2 | CNN | ✅ | ❌ | ❌ | ✅ |
| PaLM-E | 2023 | # | PyBullet | PaLM | ViT | ✅ | ✅ | ❌ | ✅ | 
| LATTE | 2023 | # | CoppeliaSim | distillBERT, CLIP | CLIP | ✅ | ❌ |  ❌ | ❌ |
| LAMP | 2023 | RLbench | CoppelaSim  | ChatGPT | R3M | ❌ | ✅ | ✅ | ❌ |
| MOO | 2023 | # | - | OWL-ViT | OWL-ViT | ✅ | ❌ | ❌ | ✅ |
| Instruction2Act | 2023 | VIMAbench | PyBullet | ChatGPT | CLIP | ❌ | ✅ |  ❌ | ❌ |
| VoxPoser | 2023 | # | SAPIEN | CPT-4 | OWL-ViT  | ✅ | ✅ | ❌ | ❌| 
| SuccessVQA | 2023 | # | IA Playroom | Flamingo | Flamingo | ✅ | ✅ | ❌ | ❌| 
| VIMA | 2023 | VIMAbench | PyBullet | T5 model | ViT | ✅ | ✅ | ❌ | ✅| 
| TidyBot | 2023 | # | - | GPT-3 | CLIP | ✅ | ✅ | ❌ | ❌| 
| Text2Motion | 2023 | # | - | GPT-3, Codex | * | ✅ | ✅ | ✅ | ❌| 
| LLM-GROP | 2023 | # | Gazebo | GPT-3 | * | ✅ | ✅ | ❌ | ❌| 
| Scaling Up | 2023 | # | MuJoCo | CLIP, GPT-3 | ResNet-18 | ✅ | ✅ | ❌ | ✅ | 
| Socratic Models | 2023 | # | - | RoBERTa, GPT-3 | CLIP | ✅ | ✅ | ❌ | ❌| 
| SayPlan | 2023 | # | - | GPT-4 | * | ✅ | ✅ | ❌ | ❌ | 
| RT-2 | 2023 | # | - | PaLI-X, PaLM-E | PaLI-X, PaLM-E | ✅ | ✅ | ❌ | ❌ | 
| KNOWNO | 2023 | # | PyBullet | PaLM-2L | * | ✅ | ✅ | ❌ | ❌ | 
| NLMap | 2023 | # | - | CLIP | ViLD | ✅ | ✅ | ❌ | ✅ | 
| Code as Policies | 2023 | # | - | GPT3, Codex | ViLD | ✅ | ✅ | ❌ | ❌ | 

