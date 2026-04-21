import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fusion_model import ConcatFusion
from copy import deepcopy
from module.moepp_layer import MoEPlusPlusEncoderLayer, Linear
from module.tamoe import InteractionMoE


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, hidden_dim):
        super(Encoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.final_norm_a = nn.LayerNorm(hidden_dim)
        self.final_norm_v = nn.LayerNorm(hidden_dim)

    def forward(self, src_a, src_v):
        output_a = src_a
        output_v = src_v
        for i in range(self.num_layers):
            output_a, map_aa, map_av = self.layers[i](src_a, src_v)
            output_v, map_vv, map_va = self.layers[i](src_v, src_a)
            src_a = output_a
            src_v = output_v

        return output_a, output_v, [map_aa, map_av, map_vv, map_va]


class SECM(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SECM, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, x, y=None):
        B, T, K, D = x.shape
        x = x.contiguous().view(B * T, K, D).permute(1, 0, 2)
        if y is not None:
            y = y.contiguous().view(B * T, K, D).permute(1, 0, 2)
            out, map1 = self.attn(x, y, y, attn_mask=None, key_padding_mask=None)
        else:
            out, map1 = self.attn(x, x, x, attn_mask=None, key_padding_mask=None)
        out = out.permute(1, 0, 2)
   
        return out, map1.contiguous().view(B, T, K, K)

class LGSF(nn.Module):
    def __init__(self, d_model, is_cross=False):
        super(LGSF, self).__init__()
        self.relu = nn.ReLU()
        if is_cross:
            self.glob = nn.Linear(d_model, d_model)
        else:
            self.glob1 = nn.Linear(d_model, d_model)
            self.glob2 = nn.Linear(d_model, d_model)

    def forward(self, x, y=None):
        B, T, K, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * K, T, D)
        if y is not None:
            y = y.permute(0, 2, 1, 3).contiguous().view(B * K, T, D)
            x_y = self.relu(self.glob(x))
            glob = self.relu(self.glob(y)).mean(-2).unsqueeze(-2)

            relevance = torch.cosine_similarity(x_y, glob, dim=-1)
            relevance = torch.sigmoid(relevance.unsqueeze(-1))
            out = relevance * glob
        else:
            '''和 Cross-modal 的区别：
                    局部特征和全局特征都来自同一模态（x 本身）；
                    使用 glob1 和 glob2 两套线性变换，分别处理局部与全局信息（避免特征混淆）。'''
            x_1 = self.relu(self.glob1(x))
            glob = self.relu(self.glob2(x)).mean(-2).unsqueeze(-2)
            relevance = torch.cosine_similarity(x_1, glob, dim=-1)
            relevance = torch.sigmoid(relevance.unsqueeze(-1))
            out = relevance * glob

        out = out.contiguous().view(B, K, T, D).permute(0, 2, 1, 3)
        out = out.contiguous().view(B * T, K, D)

        return out


class FGSE(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(FGSE, self).__init__()

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)

        self.SECM_1 = SECM(d_model, nhead, dropout=0.1)
        self.LGSF_1 = LGSF(d_model, is_cross=False)
        self.SECM_2 = SECM(d_model, nhead, dropout=0.1)
        self.LGSF_2 = LGSF(d_model, is_cross=True)  # is_cross=True → 局部特征 来自 src_q，全局特征 来自另一模态 src_v；
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, src_q, src_v):
        # B,T,K,D
        B, T, K, D = src_q.shape
        src_q0 = src_q.contiguous().view(B * T, K, D)

        '''intra'''
        src_1_scd, map_aa = self.SECM_1(src_q)
        src_1_scd = src_q0 + src_1_scd  # B * T, K, D
        src_1_scd = src_1_scd.contiguous().view(B, T, K, D)
        src_1_igf = src_q0 + self.LGSF_1(src_1_scd)
        src1 = self.fc1(src_1_igf)

        '''cross'''
        src_2_scd, map_av = self.SECM_2(src_q, src_v)
        src_2_scd = src_q0 + src_2_scd  # B * T, K, D
        src_2_scd = src_2_scd.contiguous().view(B, T, K, D)
        src_2_igf = src_q0 + self.LGSF_2(src_2_scd, src_v)
        src2 = self.fc2(src_2_igf)
        src_q0 = src1 + src2
        src_q0 = self.norm1(src_q0)

        src_q0 = self.linear2(self.dropout(F.relu(self.linear1(src_q0))))

        return src_q0.contiguous().view(B, T, K, D), map_aa, map_av


class Recon(nn.Module):

    def __init__(self, hidden_dim, input_dim):
        super(Recon, self).__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        try:
            x = self.model(x)
        except:
            assert 1 == 2, x.shape
        return x


class Alpha_net(nn.Module):

    def __init__(self, hidden_dim):
        super(Alpha_net, self).__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        try:
            x = self.model(x)
        except:
            assert 1 == 2, x.shape
        return x


class last_Pred(nn.Module):

    def __init__(self):
        super(last_Pred, self).__init__()
        self.fc = nn.ModuleList(
            [nn.Linear(128, 1) for i in range(25)])

    def forward(self, x):
        out = [self.fc[i](x[:, :, :, i, :]) for i in range(25)]
        out = torch.stack(out, dim=-2).squeeze(-1)  # (B, T, 2, k)
        return out


class custom_autograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx,input,theta):
        ctx.save_for_backward(input,theta)
        return input/(1-theta.item())

    @staticmethod
    def backward(ctx,grad_output):
        input,theta=ctx.saved_tensors
        input_grad=1/(1-theta.item())*grad_output.clone()

        return input_grad,theta


class Modality_drop():
    def __init__(self,dim_list,p_exe=0.7,device='cuda'):
        self.dim_list=dim_list
        self.p_exe=p_exe
        self.device=device

    def execute_drop(self,fead_list,q):
        # Step 1
        B = fead_list[0].shape[0]
        D = fead_list[0].shape[1]
        exe_drop = torch.tensor(np.random.rand(1)).to(device=self.device) >= 1-self.p_exe
        if not exe_drop:
            return fead_list, torch.ones([B],dtype=torch.int32,device=self.device)

        # Step 2
        num_mod=len(fead_list)
        d_sum=sum(self.dim_list)
        q_sum=sum(self.dim_list*q)
        theta=q_sum/d_sum

        # Step 3
        mask=torch.distributions.Bernoulli(1-q).sample([B,1]).permute(2,1,0).contiguous().reshape(num_mod,B,-1).to(device=self.device)
        # Step 4
        concat_list=torch.stack(fead_list,dim=0)                # [2,B,D]
        concat_list=torch.mul(concat_list,mask)
        # Step 5
        concat_list=custom_autograd.apply(concat_list,theta)
        # Step 6
        mask=torch.transpose(mask,0,1).squeeze(-1)              # [B,2]
        update_flag=torch.sum(mask,dim=1)>0
        #  Step 7
        cleaned_fea=torch.masked_select(concat_list,update_flag.unsqueeze(-1)).reshape(num_mod,-1,D)
        # Step 8
        cleaned_fea=torch.chunk(cleaned_fea,num_mod,dim=0)
        cleaned_fea=[_.squeeze(0) for _ in cleaned_fea]   # [B,D]
        return cleaned_fea,update_flag


def calcu_q(performance_1,performance_2,q_base,fix_lambda):
    q=torch.tensor([0.0,0.0])
    relu = nn.ReLU(inplace=True)
    performance_1 = torch.tensor(performance_1)
    performance_2 = torch.tensor(performance_2)
    ratio_1=torch.tanh(relu(performance_1/performance_2-1))
    ratio_2=torch.tanh(relu(performance_2/performance_1-1))
    
    lamda = fix_lambda

    
    q[0] = q_base * (1 + lamda * ratio_1) if ratio_1>0 else 0
    q[1] = q_base * (1 + lamda * ratio_2) if ratio_2>0 else 0
    
    q=torch.clip(q,0.0,1.0)
    
    return q


class MoEPlusPlusTransformer(nn.Module):
    def __init__(
        self,
        num_modalities,
        num_patches,
        hidden_dim,
        output_dim,
        num_layers,
        num_experts,
        num_heads=2,
        dropout=0.5,
    ):
        super(MoEPlusPlusTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [
                MoEPlusPlusEncoderLayer(
                    num_experts, hidden_dim, num_heads, num_modalities, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.classification_head = Linear(
            hidden_dim * num_modalities, output_dim
        ).cuda()

        self.pos_embed = nn.Parameter(
            torch.zeros(1, np.sum([num_patches] * num_modalities), hidden_dim)
        )

    def forward(self, inputs, return_latent=False):
        chunk_size = [input.shape[1] for input in inputs]
        x = torch.cat(inputs, dim=1)
        if self.pos_embed != None:
            x += self.pos_embed

        x = torch.split(x, chunk_size, dim=1)

        gate_residual = None
        for idx, layer in enumerate(self.layers):
            layer_outputs, gate_residual = layer(x, gate_residual=gate_residual)
        x = [item.mean(dim=1) for item in layer_outputs]
        x = torch.cat(x, dim=1)
        if return_latent:
            latent = x.clone()
        x = self.classification_head(x)
        if return_latent:
            return x, latent
        return x  # (64,25)



class TAMOE_Net(nn.Module):
    def __init__(self, args, mode='train', device='cuda'):
        super(TAMOE_Net, self).__init__()

        self.fc_a = nn.Linear(args.input_a_dim, args.hidden_dim)

        self.fc_v = nn.Linear(args.input_v_dim, args.hidden_dim)
        self.fc_st = nn.Linear(512, args.hidden_dim)
        self.fc_fusion = nn.Linear(args.hidden_dim * 2, args.hidden_dim)

        self.v_unmix = nn.ModuleList([nn.Sequential(nn.ReLU(), nn.Linear(args.hidden_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 128), nn.ReLU()) for i in range(26)])
        self.a_unmix = nn.ModuleList([nn.Sequential(nn.ReLU(), nn.Linear(args.hidden_dim, 128), nn.ReLU()) for i in range(26)])

        if mode == 'train':
            self.v_recon = Recon(hidden_dim=args.hidden_dim, input_dim=128 * 26)
            self.a_recon = nn.Sequential(
                nn.Linear(128 * 26, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
            )

        self.av_enhance = Encoder(FGSE(d_model=128, nhead=args.nhead, dim_feedforward=512), num_layers=args.num_layers, hidden_dim=128)

        self.fc_prob = last_Pred()
        self.fc_frame_att = last_Pred()
        self.fc_av_att = last_Pred()

        self.reduce_a = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU()
        )
        self.reduce_v = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU()
        )

        self.alpha_a = Alpha_net(args.hidden_dim)
        self.alpha_v = Alpha_net(args.hidden_dim)

        self.norm_where = args.norm_where
        self.input_v_dim = args.input_v_dim
        self.input_a_dim = args.input_a_dim
        self.hidden_dim = args.hidden_dim

        self.device=device
        self.use_adam_drop = args.use_adam_drop
        self.q_base = args.q_base
        self.lam = args.lam

        self.softmax=nn.Softmax(dim=1) 
        self.fusion_model=ConcatFusion(in_c_x=256,in_c_y=256,out_c=25)

        if args.use_adam_drop:
            self.modality_drop=Modality_drop(dim_list=torch.tensor(args.d),p_exe=args.p_exe,device=self.device)

        self.moe_fusion_model = MoEPlusPlusTransformer(
            num_modalities=2,
            num_patches=10,
            hidden_dim=args.hidden_dim,
            output_dim=25,
            num_layers=1,
            num_experts=args.num_experts,
            num_heads=args.moe_nheads,
            dropout=args.dropout,
        ).to(device=self.device)

        self.ensemble_model = InteractionMoE(
            num_modalities=2,
            fusion_model=deepcopy(self.moe_fusion_model),
            fusion_sparse=args.fusion_sparse,
            hidden_dim=args.hidden_dim,
            hidden_dim_rw=args.hidden_dim_rw,
            num_layer_rw=args.num_layer_rw,
            temperature_rw=args.temperature_rw,
        ).to(device=self.device)
        
        self.modality =  args.modality
        self.interaction_loss_weight = args.interaction_loss_weight
        self.gate_loss_weight = args.gate_loss_weight

    def temporal_pool(x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B, T, C) = (batch, time, channels)
        return: (B, 512) by concatenating mean and max over time on 256-dim features
        """
        mean_pool = x_seq.mean(dim=1)                 # (B, 256)
        max_pool, _ = x_seq.max(dim=1)                # (B, 256)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return pooled

    def forward(self, audio, visual, visual_st, label, warm_up=1, mode='train'):
        if audio.size(1) == 64:
            x1 = audio.permute(0, 2, 1).contiguous().view(-1, self.input_a_dim, 2, 32)
            upsampled = F.interpolate(x1, size=(2, 1024), mode='bicubic')
            upsampled = self.fc_a(upsampled.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).mean(dim=2)
            x1 = F.adaptive_avg_pool1d(upsampled, 10).view(-1, self.hidden_dim, 10)
            x1 = x1.permute(0, 2, 1)
        else:
           x1 = self.fc_a(audio)

        # 2d and 3d visual feature fusion
        if visual.size(1) == 80:  # input 2d features are from ResNet152
            vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
            vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        else:  # input 2d features are from CLIP
            vid_s = self.fc_v(visual)

        vid_st = self.fc_st(visual_st)
        x2 = torch.cat((vid_s, vid_st), dim=-1)
        x2 = self.fc_fusion(x2)


        fusion_input = [x1.contiguous(), x2.contiguous()]
        if mode == 'train':
            expert_outputs, routing_weights, outputs, interaction_losses = self.ensemble_model(fusion_input)
            criterion = torch.nn.BCEWithLogitsLoss()
            task_loss = criterion(outputs, label)
            interaction_loss = sum(interaction_losses) / (self.modality + 2)
            loss = task_loss + self.interaction_loss_weight * interaction_loss

        else:
            _, _, outputs = self.ensemble_model.inference(fusion_input)

        features1 = [self.a_unmix[i](x1) for i in range(26)]
        x1_2 = torch.stack(features1, dim=0).permute(1, 2, 0, 3)
        B, T, K, D = x1_2.shape
        features2 = [self.v_unmix[i](x2) for i in range(26)]
        x2_2 = torch.stack(features2, dim=0).permute(1, 2, 0, 3)

        apha_x1 = torch.sigmoid(self.alpha_a(x1).unsqueeze(-1))
        apha_x2 = torch.sigmoid(self.alpha_v(x2).unsqueeze(-1))

        x1_2_bg = x1_2[:, :, -1, :].unsqueeze(-2).repeat(1, 1, 25, 1)
        x2_2_bg = x2_2[:, :, -1, :].unsqueeze(-2).repeat(1, 1, 25, 1)
        x1_2_e = x1_2[:, :, :-1, :]
        x2_2_e = x2_2[:, :, :-1, :]

        x1_new = torch.cat([apha_x1 * x1_2_e, (1 - apha_x1) * x1_2_bg], dim=-1)
        x2_new = torch.cat([apha_x2 * x2_2_e, (1 - apha_x2) * x2_2_bg], dim=-1)

        x1_3 = self.reduce_a(x1_new)
        x2_3 = self.reduce_v(x2_new)

        if mode == 'train':
            loss_a_ort = torch.cosine_similarity(x1_2_e, x1_2_bg, dim=-1).mean()
            loss_v_ort = torch.cosine_similarity(x2_2_e.mean(-2), x2_2[:, :, -1, :], dim=-1).mean()
            x1_ = x1_2.contiguous().view(B, T, K * D)
            x2_ = x2_2.contiguous().view(B, T, K * D)
            x1_rec = self.a_recon(x1_)
            x2_rec = self.v_recon(x2_)
            loss_a_rec = nn.MSELoss()(x1, x1_rec)
            loss_v_rec = nn.MSELoss()(x2, x2_rec)

        x1_3, x2_3, [map_aa, map_av, map_vv, map_va] = self.av_enhance(x1_3, x2_3)

        performance_1=None
        performance_2=None
        t1,t2=None,None

        w = self.fusion_model.fxy.weight.clone().detach()
        b = self.fusion_model.fxy.bias.clone().detach()

        t1_bias = b / 2
        t2_bias = b / 2

        t1 = torch.mm(x1.mean(dim=1), w[:, :256].T) + t1_bias
        t2 = torch.mm(x2.mean(dim=1), w[:, 256:].T) + t2_bias

        performance_1 = F.softmax(t1, dim=1).mul(label).sum().item()
        performance_2 = F.softmax(t2, dim=1).mul(label).sum().item()

        if warm_up == 0 and self.use_adam_drop:
            self.q = calcu_q(performance_1, performance_2, q_base=self.q_base, fix_lambda=self.lam)
            cleaned_fea, update_flag = self.modality_drop.execute_drop([x1.mean(dim=1), x2.mean(dim=1)], self.q)
            _, _, out = self.fusion_model(cleaned_fea[0], cleaned_fea[1])
        else:
            _, _, out = self.fusion_model(x1.mean(dim=1), x2.mean(dim=1))
            update_flag = torch.ones([x1.size(0)], dtype=torch.int32, device=x1.device)

        # prediction
        x = torch.cat([x1_3.unsqueeze(2), x2_3.unsqueeze(2)], dim=2)  # (B, T, 2, K,D)
        frame_logits = self.fc_prob(x)
        frame_prob = torch.sigmoid(frame_logits)  # (B, T, 2, K)

        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)  # (B, T, 2, K)
        av_att = torch.softmax(self.fc_av_att(x), dim=2)  # (B, T, 2, K)
        temporal_prob = (frame_att * frame_prob)
        global_prob = (temporal_prob * av_att).sum(dim=2).sum(dim=1)  # (B, K)

        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)  # (B, K)
        v_prob = temporal_prob[:, :, 1, :].sum(dim=1)  # (B, K).


        if mode == 'train':
            return global_prob, a_prob, v_prob, frame_prob, frame_logits, [loss_a_rec, loss_v_rec, loss_a_ort, loss_v_ort], [map_aa, map_av, map_vv, map_va], (t1, t2, out, update_flag, performance_1, performance_2), loss
        else:
            return global_prob, a_prob, v_prob, frame_prob, frame_logits

