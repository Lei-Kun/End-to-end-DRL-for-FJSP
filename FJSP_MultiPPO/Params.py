import argparse

parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
# args for device
parser.add_argument('--device', type=str, default="cuda", help='Number of jobs of instances')
# args for env
parser.add_argument('--n_j', type=int, default=30, help='Number of jobs of instance')
parser.add_argument('--n_m', type=int, default=20, help='Number of machines instance')
parser.add_argument('--rewardscale', type=float, default=0., help='Reward scale for positive rewards')
parser.add_argument('--init_quality_flag', type=bool, default=False, help='Flag of whether init state quality is 0, True for 0')
parser.add_argument('--low', type=int, default=-99, help='LB of duration')
parser.add_argument('--high', type=int, default=99, help='UB of duration')
parser.add_argument('--np_seed_train', type=int, default=200, help='Seed for numpy for training')
parser.add_argument('--np_seed_validation', type=int, default=200, help='Seed for numpy for validation')
parser.add_argument('--torch_seed', type=int, default=600, help='Seed for torch')
parser.add_argument('--et_normalize_coef', type=int, default=1000, help='Normalizing constant for feature LBs (end time), normalization way: fea/constant')
parser.add_argument('--dr_normalize_coef', type=int, default=100, help='Normalizing constant for feature LBs (end time), normalization way: fea/constant')
parser.add_argument('--wkr_normalize_coef', type=int, default=1, help='Normalizing constant for wkr, normalization way: fea/constant')
# args for network
parser.add_argument('--num_layers', type=int, default=3, help='No. of layers of feature extraction GNN including input layer')
parser.add_argument('--neighbor_pooling_type', type=str, default='average', help='neighbour pooling type')
parser.add_argument('--graph_pool_type', type=str, default='average', help='graph pooling type')
parser.add_argument('--input_dim', type=int, default=2, help='number of dimension of raw node features')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dim of MLP in fea extract GNN')
parser.add_argument('--num_mlp_layers_feature_extract', type=int, default=3, help='No. of layers of MLP in fea extract GNN')
parser.add_argument('--num_mlp_layers_actor', type=int, default=3, help='No. of layers in actor MLP')
parser.add_argument('--hidden_dim_actor', type=int, default=128, help='hidden dim of MLP in actor')
parser.add_argument('--num_mlp_layers_critic', type=int, default=2, help='No. of layers in critic MLP')
parser.add_argument('--Init', type=bool, default=True, help='No. of layers in critic MLP')
parser.add_argument('--hidden_dim_critic', type=int, default=32, help='hidden dim of MLP in critic')
# args for PPO

parser.add_argument('--ppo_step', type=int, default=3, help='No. of envs for training')
parser.add_argument('--batch_size', type=int, default=128, help='No. of envs for training')
parser.add_argument('--num_ins', type=int, default=12800, help='No. of envs for training')

parser.add_argument('--max_updates', type=int, default=10000, help='No. of episodes of each env for training')
parser.add_argument('--lr', type=float, default=1e-3, help='lr')
parser.add_argument('--decayflag', type=bool, default=False, help='lr decayflag')
parser.add_argument('--decay_step_size', type=int, default=500, help='decay_step_size')
parser.add_argument('--decay_ratio', type=float, default=0.96, help='decay_ratio, e.g. 0.9, 0.95')
parser.add_argument('--gamma', type=float, default=1, help='discount factor')
parser.add_argument('--k_epochs', type=int, default=1, help='update policy for K epochs')
parser.add_argument('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
parser.add_argument('--vloss_coef', type=float, default=1, help='critic loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=2, help='policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.01, help='entropy loss coefficient')

configs = parser.parse_args()
