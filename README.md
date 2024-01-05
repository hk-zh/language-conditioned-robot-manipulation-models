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

### Simulator
| Simulator | Description |
| - | - |
| PyBullet | <div style="width: 350pt">With its origins rooted in the Bullet physics engine, PyBullet transcends the boundaries of conventional simulation platforms, offering a wealth of tools and resources for tasks ranging from robot manipulation and locomotion to computer-aided design analysis.</div> | <div style="width: 250pt"> Shao et al., Mees et al.  leverage pybullet to build a table-top environment to conduct object manipulations tasks. </div>|
| MuJoCo | <div style="width: 350pt"> MuJoCo, short for "Multi-Joint dynamics with Contact", originates from the vision of creating a physics engine tailored for simulating articulated and deformable bodies. It has evolved into an essential tool for exploring diverse domains, from robot locomotion and manipulation to human movement and control. </div>|
| CoppeliaSim | <div style="width: 350pt"> CoppeliaSim is formerly known as V-REP (Virtual Robot Experimentation Platform). It offers a comprehensive environment for simulating and prototyping robotic systems, enabling users to create, analyze, and optimize a wide spectrum of robotic applications. Its origins as an educational tool have evolved into a full-fledged simulation framework, revered for its versatility and user-friendly interface. </div>|
| NVIDIA Omniverse | <div style="width: 350pt"> NVIDIA Omniverse offers real-time physics simulation and lifelike rendering, creating a virtual environment for comprehensive testing and fine-tuning of robotic manipulation algorithms and control strategies, all prior to their actual deployment in the physical realm. </div>|
| Unity | <div style="width: 350pt"> Unity is a cross-platform game engine developed by Unity Technologies. Renowned for its user-friendly interface and powerful capabilities, Unity has become a cornerstone in the worlds of video games, augmented reality (AR), virtual reality (VR), and also simulations. </div>| 

### Benchmarks
<table>
    <tr align="center">
        <th rowspan="2" >Benchmark</th>
        <th rowspan="2">Simulation Engine</th>
        <th rowspan="2">Manipulator</th>
        <td colspan="3"> <b> Observation</td>
        <th rowspan="2">Tool used</th>
        <th rowspan="2">Multi-agents</th>
        <th rowspan="2">Long-horizon</th>
    </tr>
    <tr>
        <th>RGB</th>
        <th>Depth</th>
        <th>Masks</th>
    </tr>
    <tr align="center">
        <td><a href="http://calvin.cs.uni-freiburg.de/">CALVIN</a></td>
        <td>PyBullet</td>
        <td>Franka Panda</td>
        <td>✅</td>
        <td>✅</td>
        <td>❌</td>
        <td>❌</td>
        <td>❌</td>
        <td>✅</td>
    </tr>
    <tr align="center">
        <td><a href="https://meta-world.github.io/">Meta-world</a></td>
        <td>MuJoCo</td>
        <td>Sawyer</td>
        <td>✅</td>
        <td>❌</td>
        <td>❌</td>
        <td>❌</td>
        <td>❌</td>
        <td>❌</td>
    </tr>
    <tr align="center">
        <td><a href="https://arxiv.org/abs/2308.00937">LEMMA</a></td>
        <td>NVIDIA Omniverse</td>
        <td>UR10 & UR5</td>
        <td>✅</td>
        <td>✅</td>
        <td>❌</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr align="center">
        <td><a href="https://github.com/stepjam/RLBench">RLbench</a></td>
        <td>CoppeliaSim</td>
        <td>Franka Panda</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>❌</td>
        <td>❌</td>
        <td>✅</td>
    </tr>
    <tr align="center">
        <td><a href="https://github.com/vimalabs/VIMABench">VIMAbench</a></td>
        <td>Pybullet</td>
        <td>UR5</td>
        <td>✅</td>
        <td>❌</td>
        <td>❌</td>
        <td>❌</td>
        <td>❌</td>
        <td>✅</td>
    </tr>
    <tr align="center">
        <td><a href="https://cisnlp.github.io/lohoravens-webpage/">LoHoRavens</a></td>
        <td>Pybullet</td>
        <td>UR5</td>
        <td>✅</td>
        <td>✅</td>
        <td>❌</td>
        <td>❌</td>
        <td>❌</td>
        <td>✅</td>
    </tr>
</table>


### Models
| Model | Year | Benchmark | Simulation Engine | Language Module| Perception Module | Real World Experiment | LLM | Reinforcement Learning | Imitation Learning |
| ------ | ------ | :-----------: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|  [DREAMCELL](https://arxiv.org/abs/1903.08309) |  2019  |        #         | -       | LSTM | *   | ❌ | ❌ | ❌ | ✅ |
|  [PixL2R](https://proceedings.mlr.press/v155/goyal21a.html) |  2020  |    Meta-World    | MuJoCo  | LSTM | CNN | ❌ | ❌ | ✅ | ❌ |
| [Concept2Robot](https://www.roboticsproceedings.org/rss16/p082.pdf) | 2020 | # | PyBullet | BERT | ResNet-18 | ❌ | ❌ | ❌ | ✅ |
| [LanguagePolicy](https://proceedings.neurips.cc/paper/2020/hash/9909794d52985cbc5d95c26e31125d1a-Abstract.html) | 2020 | # | CoppeliaSim | GLoVe | Faster RCNN | ❌ | ❌ | ❌ | ✅ |
| [LOReL](https://proceedings.mlr.press/v164/nair22a.html)| 2021 | Meta-World | MuJoCo | distillBERT | CNN | ✅ | ❌ | ❌ | ✅ |
| [CARE](https://proceedings.mlr.press/v139/sodhani21a.html) | 2021 | Meta-World | MuJoCo | RoBERTa | * | ❌ | ✅ | ✅ | ❌ |
| [MCIL](https://arxiv.org/abs/2005.07648) | 2021 | # | MuJoCo | MUSE | CNN | ❌ | ❌ | ❌ | ✅ |
| [BC-Z](https://arxiv.org/abs/2202.02005) | 2021 | # | - | MUSE | ResNet18 | ✅ | ❌ | ❌ | ✅ |
| [CLIPort](https://proceedings.mlr.press/v164/shridhar22a.html) | 2021 | # | Pybullet | CLIP | CLIP/ResNet | ✅ | ❌ | ❌ | ✅ |
| [LanCon-Learn](https://ieeexplore.ieee.org/document/9667188) | 2022 | Meta-World | MuJoCo | GLoVe | * | ❌ | ❌ | ✅ | ✅ | 
| [MILLON](https://arxiv.org/abs/2209.04924) | 2022 | Meta-World| MuJoCo | GLoVe | * | ✅ | ❌ | ✅ | ❌ | 
| [PaLM-SayCan](https://arxiv.org/abs/2204.01691) | 2022 | # | - | PaLM | ViLD | ✅ | ✅ | ✅ | ✅ |
| [ATLA](https://arxiv.org/abs/2206.13074) | 2022 | # | PyBullet | BERT-Tiny | CNN | ❌ | ✅ | ✅ | ❌ |
| [HULC](https://arxiv.org/abs/2204.06252) | 2022 | CALVIN | Pybullet | MiniLM-L3-v2 | CNN | ❌ | ❌ | ❌ | ✅ |
| [PerAct](https://arxiv.org/abs/2209.05451) | 2022 | RLbench | CoppelaSim | CLIP | ViT | ✅ | ❌ | ❌ | ✅ |
| [RT-1](https://arxiv.org/abs/2212.06817) | 2022 | # | - | USE | EfficientNet-B3 |  ✅ | ✅ | ❌ | ❌ |
| [LATTE](https://arxiv.org/abs/2208.02918) | 2023 | # | CoppeliaSim | distillBERT, CLIP | CLIP | ✅ | ❌ |  ❌ | ❌ |
| [DIAL](https://arxiv.org/abs/2211.11736) | 2022 | # | - | CLIP | CLIP |  ✅ | ✅ | ❌ | ✅ |
| [R3M](https://arxiv.org/abs/2203.12601) | 2022 | # | - | distillBERT | ResNet | ✅ | ❌ | ❌ | ✅ |
| [Inner Monologue](https://arxiv.org/abs/2207.05608) | 2022 | # | - | CLIP | CLIP | ✅ | ✅ | ❌ | ❌ |
| [NLMap](https://ieeexplore.ieee.org/document/10161534) | 2023 | # | - | CLIP | ViLD | ✅ | ✅ | ❌ | ✅ |
| [Code as Policies](https://ieeexplore.ieee.org/document/10160591) | 2023 | # | - | GPT3, Codex | ViLD | ✅ | ✅ | ❌ | ❌ | 
| [PROGPROMPT](https://arxiv.org/abs/2209.11302) |  2023 | Virtualhome | Unity3D | GPT-3 | * | ✅ | ✅ | ❌ | ❌ |
| [Language2Reward](https://arxiv.org/abs/2306.08647) | 2023 | # | MuJoCo MPC | GPT-4 | * | ✅ | ✅ | ✅ | ❌ |
| [LfS](https://arxiv.org/abs/2209.10656) | 2023 | Meta-World | MuJoCo | Cons. Parser | * | ✅ | ❌ | ✅ | ❌ |
| [HULC++](https://arxiv.org/abs/2210.01911)| 2023 | CALVIN | PyBullet | MiniLM-L3-v2 | CNN | ✅ | ❌ | ❌ | ✅ |
| [LEMMA](https://arxiv.org/abs/2308.00937) | 2023 | LEMMA | NVIDIA Omniverse | CLIP | CLIP | ❌ | ❌ | ❌ | ✅ |
| [SPIL](https://arxiv.org/abs/2305.19075)| 2023 | CALVIN | PyBullet | MiniLM-L3-v2 | CNN | ✅ | ❌ | ❌ | ✅ |
| [PaLM-E](https://proceedings.mlr.press/v202/driess23a.html) | 2023 | # | PyBullet | PaLM | ViT | ✅ | ✅ | ❌ | ✅ | 
| [LAMP](https://arxiv.org/abs/2308.12270) | 2023 | RLbench | CoppelaSim  | ChatGPT | R3M | ❌ | ✅ | ✅ | ❌ |
| [MOO](https://arxiv.org/abs/2303.00905) | 2023 | # | - | OWL-ViT | OWL-ViT | ✅ | ❌ | ❌ | ✅ |
| [Instruction2Act](https://arxiv.org/abs/2305.11176) | 2023 | VIMAbench | PyBullet | ChatGPT | CLIP | ❌ | ✅ |  ❌ | ❌ |
| [VoxPoser](https://arxiv.org/abs/2307.05973) | 2023 | # | SAPIEN | CPT-4 | OWL-ViT  | ✅ | ✅ | ❌ | ❌| 
| [SuccessVQA](https://arxiv.org/abs/2303.07280) | 2023 | # | IA Playroom | Flamingo | Flamingo | ✅ | ✅ | ❌ | ❌| 
| [VIMA](https://arxiv.org/abs/2210.03094) | 2023 | VIMAbench | PyBullet | T5 model | ViT | ✅ | ✅ | ❌ | ✅| 
| [TidyBot](https://arxiv.org/abs/2305.05658) | 2023 | # | - | GPT-3 | CLIP | ✅ | ✅ | ❌ | ❌| 
| [Text2Motion](https://arxiv.org/abs/2303.12153) | 2023 | # | - | GPT-3, Codex | * | ✅ | ✅ | ✅ | ❌| 
| [LLM-GROP](https://arxiv.org/abs/2303.06247) | 2023 | # | Gazebo | GPT-3 | * | ✅ | ✅ | ❌ | ❌| 
| [Scaling Up](https://arxiv.org/abs/2307.14535) | 2023 | # | MuJoCo | CLIP, GPT-3 | ResNet-18 | ✅ | ✅ | ❌ | ✅ | 
| [Socratic Models](https://openreview.net/pdf?id=kdHpWogtX6Y) | 2023 | # | - | RoBERTa, GPT-3 | CLIP | ✅ | ✅ | ❌ | ❌| 
| [SayPlan](https://arxiv.org/abs/2307.06135) | 2023 | # | - | GPT-4 | * | ✅ | ✅ | ❌ | ❌ | 
| [RT-2](https://arxiv.org/abs/2307.15818) | 2023 | # | - | PaLI-X, PaLM-E | PaLI-X, PaLM-E | ✅ | ✅ | ❌ | ❌ | 
| [KNOWNO](https://arxiv.org/abs/2307.01928) | 2023 | # | PyBullet | PaLM-2L | * | ✅ | ✅ | ❌ | ❌ | 
 


