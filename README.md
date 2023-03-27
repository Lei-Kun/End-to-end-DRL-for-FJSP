# End-to-end-DRL-for-FJSP

#---------------------------------------------------------------------------------

2023/02/15 I've revised the 'PPOwithValue.py' so that it's suitable for a higher version of Pytorch. 

2022/11/03 torch == 1.4.0

2022/09/24 I've uploaded the FJSP_realworld files, you can download it to run the 'validation_realWorld.py' to test on bechmark instances. In particular, you can download more bechmark instances and saved models to test, and in this project I only upload one saved model and benchmark. 

You can download the "FJSP_MultiPPO" project to run 'PPOwithValue' file to train the policies, run the 'validation' file to test/validate on random generated instances.

You can download the other project named 'FJSP-benchmarks' in my github account to test the trained model on real-world instances.

#----------------------------------------------------------------------------------

2022/09/12 Some issues are resolveed, please download the latest code. If you have any question please feel free to mail to me via: kunlei@my.swjtu.edu.cn.

#----------------------------------------------------------------------------------

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

This work can be extend to solve other type of scheduling problems which can be represented by disjunctive graph, e.g., flow shop scheduling problem, dynamic FJSP etc., or FJSP with other objective, e.g., mean comletion time and sequence dependence & setup time. 

The proposed multi-PPO algorithm can be extend to solve other multi-action decision needed combinatorial optimization problem.


# Running the code
You can run the 'PPOwithValue' file to train the policies, run the 'validation' file to test/validate on random generated instances.

# Motivation 
Most traditional methods, including exact methods based on mathematical programming and metaheuristics, cannot apply to large FJSP instances or real-time FJSP instances due to their time complexity. Some researchers have used DRL to solve combinatorial optimization problems and achieved good results, but FJSP has received less attention. Some DRL-based methods in solving FJSP are designed to select composite dispatching rules instead of directly finding scheduling solutions, whose performance depends on the design of dispatching rules. To the best of our knowledge, there is no research to solve the FJSP via multiple action end-to-end DRL framework without predetermined dispatching rules. 

In this paper, we proposed a novel end-to-end model-free DRL architecture on FJSP and demonstrated that it yields superior performance in terms of solution quality and efficiency. The proposed model-free DRL architecture can be directly applied to arbitrary FJSP scenarios without modeling the environment in advance. That is to say, the transition probability distribution (and the reward function) associated with the Markov decision process (MDP) is not explicitly defined when invoking the environment. Meanwhile, based on the advantages of our design of policy networks, our architecture is not bounded by the instance size

# Graph neural network for disjunctive graph of FJSP
The disjunctive graphprovides a complete view of the scheduling states containing numerical and structural information, such as the precedence constraints, processing order on each machine, compatible machine set for each operation, and the processing time of a compatible machine for each operation. It is crucial to extract all state information embedded in the disjunctive graph to achieve effective scheduling performance. It motivates us to embed the complex graph state by exploiting a graph neural network (GNN). We used the Graph Isomorphism Network (GIN) to encode the disjunctive graph.

# Deep reinforcement learning algorithm 
To cope with this kind of multi-action reinforcement problem, we proposed a multi-Proximal Policy Optimization (multi-PPO) algorithm that takes a multiple actor-critic architecture and adopts PPO as its policy optimization method for learning the two sub-policies. The PPO algorithm is a state-of-the-art policy gradient approach with an actor-critic style, which is widely used to deal with both discrete and continuous control tasks . However, the PPO algorithm cannot be directly used to handle a multi-action task since it generally contains one actor to learn one policy that can only control a single action at each timestep. By contrast, the proposed multi-PPO architecture includes two actor networks (job operation and machine encoder-decoders as the two actor networks, respectively).
 
# Cite us
For open access source, please cite the work correctly!!!


