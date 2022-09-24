import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic,MLP
import torch.nn.functional as F
from models.graphcnn_congForSJSSP import GraphCNN
import torch
from Mhattention import ProbAttention
from agent_utils import select_action1,greedy_select_action,select_action2
import numpy as np
from models.attentionEncoder import GraphAttentionEncoder
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)

        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)



        return u_i

class Encoder(nn.Module):
    def __init__(self,num_layers, num_mlp_layers, input_dim,  hidden_dim, learn_eps, neighbor_pooling_type, device):
        super(Encoder,self).__init__()
        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
    def forward(self,x,graph_pool, padded_nei, adj,):
        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)

        return h_pooled,h_nodes

class Job_Actor(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 device
                 ):
        super(Job_Actor, self).__init__()
        # job size for problems, no business with network
        self.n_j = n_j
        #self.bn = torch.nn.BatchNorm1d(input_dim).to(device)
        # machine size for problems, no business with network
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim, bias=False).to(device)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        #self.fc2 = nn.Linear(1, hidden_dim, bias=False).to(device)

        #self.fc3 = nn.Linear(hidden_dim*2, hidden_dim, bias=False).to(device)
        self.encoder = Encoder(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        self._input = nn.Parameter(torch.Tensor(hidden_dim))
        self._input.data.uniform_(-1, 1)


        '''self.actor1 = MLPActor(4, hidden_dim * 2, hidden_dim, 1).to(device)'''
        #self.actor = ProbAttention(8,hidden_dim,hidden_dim).to(device)

        #self.cell = torch.nn.GRUCell(2*hidden_dim, hidden_dim, bias=True).to(device)
        self.attn = Attention(hidden_dim).to(device)

        #self.mlp = MLP(3, hidden_dim*2, hidden_dim*2, self.n_m).to(device)
        #self.MCH_actor = ProbAttention(8, hidden_dim, hidden_dim).to(device)
        self.actor = MLPActor(3, hidden_dim*2, hidden_dim, 1).to(device)
        #self.select_mch = MLPCritic(num_mlp_layers_actor, hidden_dim, hidden_dim*2, self.n_m)
    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                candidate,
                mask,
                pretask,
                firsttask,
                j,
                mask_mch,
                hx,
                mch_node,
                dur,
                job,
                job_time,
                T=1,
                greedy=False
                ):
        #print('sssssssssssssssssssssss',x.size(),graph_pool.size(),padded_nei,adj.size(),candidate.size(),mask.size())
        h_pooled, h_nodes = self.encoder(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        #(batch_size*task,hidden_dim)

        # prepare policy feature: concat omega feature with global feature

        dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))

        batch_node = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)

        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)

        #job_time_feature = self.fc2(job_time.unsqueeze(-1))

        #candidate_feature = self.fc3(torch.cat([candidate_feature,job_time_feature],-1))
        if j == 0:
            _input = self._input[None, :].expand(candidate.size(0), -1).to(self.device)
        else:
            _input = torch.gather(batch_node, 1,
                                  pretask.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1,
                                                                           batch_node.size(2))).squeeze(1)
            #_input = torch.cat([_input, mch_node], -1)
            first_node = torch.gather(batch_node, 1,
                                  firsttask.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1,
                                                                           batch_node.size(2))).squeeze(1)
            pre_nodes = torch.cat([first_node, _input], dim=-1)

        #_input_first = self.fc(_input)
        #pool = self.fc1(h_pooled)
        decoder_input = torch.cat([h_pooled,_input],dim=-1)
        decoder_input = self.fc(decoder_input)
        # -----------------------------------------------------------------------------cat(pool,first_node,current_node)

        '''decoder_input = torch.cat([h_pooled,pre_nodes], dim=-1)
        decoder_input  = self.fc(decoder_input)'''
        # -----------------------------------------------------------------------------------------------------------
        #candidate_scores = self.actor(h_pooled, candidate_feature,0)

#--------------------------------------------------------------------------------------------------------------------------
        '''c'''

# --------------------------------------------------------------------------------------------------------------------------
        '''if j == 0:
            hx = self.cell(_input,h_pooled)
        else:
            hx = torch.cat([hx,h_pooled],-1)
            hx = self.fc3(hx)
            hx = self.cell(_input, hx)'''
        #candidate_scores= self.attn(h_pooled, candidate_feature)

        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        candidate_scores = self.actor(concateFea)

        candidate_scores = candidate_scores * 10
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')

        pi = F.softmax(candidate_scores, dim=1)
        if greedy:
            action = greedy_select_action(pi,candidate)
            log_a = 0
        else:
            action, index, log_a = select_action1(pi, candidate)



        action1 = action.type(torch.long).to(self.device)

        batch_x = dur.reshape(dummy.size(0), self.n_j*self.n_m, -1).to(self.device)

        #mask_mch = mask_mch.reshape(dummy.size(0),-1,self.n_m)

        '''indexs_a = []
        for z in range(dummy.size(0)):

            ins = np.where(job[z] == action[z].item())[0]
            indexs_a.append(ins.tolist())
        indexs_a = torch.tensor(indexs_a).type(torch.long).to(self.device)'''



        mask_mch_action = torch.gather(mask_mch, 1,
                                  action1.unsqueeze(-1).unsqueeze(-1).expand(mask_mch.size(0), -1,
                                                                           mask_mch.size(2)))
        #--------------------------------------------------------------------------------------------------------------------
        action_feature = torch.gather(batch_node, 1,
                                      action1.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1,
                                                                                 batch_node.size(2))).squeeze(1)
        action_node = torch.gather(batch_x, 1,
                                   action1.unsqueeze(-1).unsqueeze(-1).expand(batch_x.size(0), -1,
                                                                              batch_x.size(2))).squeeze()#[:,:-2]


        return action,log_a,action_node.detach(),action_feature.detach(),mask_mch_action.detach(),h_pooled.detach()

class Mch_Actor(nn.Module):
    def __init__(self,n_j,
                 n_m,
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 device):
        super(Mch_Actor,self).__init__()
        self.n_j = n_j
        self.bn = torch.nn.BatchNorm1d(input_dim)
        # machine size for problems, no business with network
        self.n_m = n_m
        self.hidden_dim = hidden_dim
        self.n_ops_perjob = n_m
        self.device = device
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.actor = ProbAttention(8, hidden_dim, hidden_dim).to(device)
        #self.attn = Attention(hidden_dim).to(device)
        #self.cell = torch.nn.GRUCell(hidden_dim, hidden_dim, bias=True).to(device)
        self.fc2 = nn.Linear(2, hidden_dim, bias=False)
        #self.actor = MLPActor(4, hidden_dim * 2, hidden_dim, 1)
        self.init_embed1 = nn.Linear(2, hidden_dim)
        self.mch_encoder = GraphAttentionEncoder(n_heads=8, embed_dim=hidden_dim, n_layers=3,
                                             normalization='batch')

    def forward(self,action_node,hx,mask_mch_action,mch_time,greedy=False):


        feature = torch.cat([mch_time.unsqueeze(-1),action_node.unsqueeze(-1)],-1)
        #action_node,_ = self.mch_encoder(self.init_embed1(feature))


        action_node = self.bn(self.fc2(feature).reshape(-1,self.hidden_dim)).reshape(feature.size(0),-1,self.hidden_dim)

        '''h_pooled_repeated = hx.unsqueeze(1).expand_as(action_node)
        concateFea = torch.cat((action_node, h_pooled_repeated), dim=-1)
        mch_scores = self.actor(concateFea)'''

        mch_scores = self.actor(hx, action_node, mask_mch_action, True)
        #mch_scores = self.attn(hx,action_node)
        # --------------------------------------------------------------------------------------------------------------------
        '''action_feature = torch.gather(batch_node, 1,
                                  action1.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1,
                                                                           batch_node.size(2))).squeeze(1)
        action_feature = torch.cat([action_feature,pool],dim=-1)
        mch_scores = self.mlp(action_feature)'''
        # ---------------------------------------------------------------------------------------------------------------------

        mch_scores = mch_scores * 10

        # mask_reshape = mask_mch_action.reshape(candidate_scores.size())
        mch_scores = mch_scores.masked_fill(mask_mch_action.squeeze(1).bool(), float("-inf"))

        pi_mch = F.softmax(mch_scores, dim=1)
        if greedy:
            log_mch, mch_a = pi_mch.squeeze().max(1)
        else:
            mch_a, log_mch = select_action2(pi_mch)

        mch_node = torch.gather(action_node, 1,
                                  mch_a.unsqueeze(-1).unsqueeze(-1).expand(action_node.size(0), -1,
                                                                           action_node.size(2))).squeeze(1)

        return log_mch,mch_a,mch_node











if __name__ == '__main__':
    print('Go home')