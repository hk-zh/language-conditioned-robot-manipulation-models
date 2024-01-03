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
