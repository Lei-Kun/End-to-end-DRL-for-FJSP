# End-to-end-DRL-for-FJSP
This is the code for our published paper: 'A Multi-action Deep Reinforcement Learning Framework for Flexible Job-shop Scheduling Problem'; Everyone is welcome to use this code and cite our paper:

{Kun Lei, Peng Guo, Wenchao Zhao, Yi Wang, Linmao Qian, Xiangyin Meng, Liansheng Tang,
A multi-action deep reinforcement learning framework for flexible Job-shop scheduling problem,
Expert Systems with Applications,
Volume 205,
2022,
117796,
ISSN 0957-4174,
https://doi.org/10.1016/j.eswa.2022.117796.
(https://www.sciencedirect.com/science/article/pii/S0957417422010624)}

# Motivation 
Most traditional methods, including exact methods based on mathematical programming and metaheuristics, cannot apply to large FJSP instances or real-time FJSP instances due to their time complexity. Some researchers have used DRL to solve combinatorial optimization problems and achieved good results, but FJSP has received less attention. Some DRL-based methods in solving FJSP are designed to select composite dispatching rules instead of directly finding scheduling solutions, whose performance depends on the design of dispatching rules. To the best of our knowledge, there is no research to solve the FJSP via multiple action end-to-end DRL framework without predetermined dispatching rules. 

In this paper, we proposed a novel end-to-end model-free DRL architecture on FJSP and demonstrated that it yields superior performance in terms of solution quality and efficiency. The proposed model-free DRL architecture can be directly applied to arbitrary FJSP scenarios without modeling the environment in advance. That is to say, the transition probability distribution (and the reward function) associated with the Markov decision process (MDP) is not explicitly defined when invoking the environment. Meanwhile, based on the advantages of our design of policy networks, our architecture is not bounded by the instance size

# Graph neural network for disjunctive graph of FJSP
