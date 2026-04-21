import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        activation=nn.ReLU(),
        dropout=0.5,
    ):
        super(MLP, self).__init__()
        layers = []
        self.drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(self.drop)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(self.drop)
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPReWeighting(nn.Module):
    """Use MLP to re-weight all interaction experts."""

    def __init__(
        self,
        num_modalities,
        num_branches,
        hidden_dim=256,
        hidden_dim_rw=256,
        num_layers=2,
        temperature=1,
    ):
        """args:
        hidden_dim: hidden dimension of input embeddings.
        hidden_dim_rw: hidden dimension of the re-weighting model.
        """
        super(MLPReWeighting, self).__init__()
        self.temperature = temperature
        self.mlp = MLP(
            hidden_dim * num_modalities,
            hidden_dim_rw,
            num_branches,
            num_layers,
            activation=nn.ReLU(),
            dropout=0.5,
        )

    def temperature_scaled_softmax(self, logits):
        logits = logits / self.temperature
        return torch.softmax(logits, dim=1)

    def forward(self, inputs):
        if inputs[0].dim() == 3:
            x = [item.mean(dim=1) for item in inputs]
            x = torch.cat(x, dim=1)
        else:
            x = torch.cat(inputs, dim=1)
        x = self.mlp(x)
        return self.temperature_scaled_softmax(x)


class InteractionExpert(nn.Module):
    """
    Interaction Expert.
    """

    def __init__(self, fusion_model, fusion_sparse):
        super(InteractionExpert, self).__init__()
        self.fusion_model = fusion_model
        self.fusion_sparse = fusion_sparse

    def forward(self, inputs):
        """
        Forward pass with all modalities present.
        """
        return self._forward_with_replacement(inputs, replace_index=None)

    def forward_with_replacement(self, inputs, replace_index):
        """
        Forward pass with one modality replaced by a random vector.

        Args:
            inputs (list of tensors): List of modality inputs.
            replace_index (int): Index of the modality to replace. If None, no modality is replaced.
        """
        return self._forward_with_replacement(inputs, replace_index=replace_index)

    def _forward_with_replacement(self, inputs, replace_index=None):
        """
        Internal function to handle forward pass with optional modality replacement.
        """
        # Replace specified modality with a random vector
        if replace_index is not None:
            random_vector = torch.randn_like(inputs[replace_index])
            inputs = (
                inputs[:replace_index] + [random_vector] + inputs[replace_index + 1 :]
            )

        x = self.fusion_model(inputs)
        return x  # (64,25)

    def forward_multiple(self, inputs):
        """
        Perform (1 + n) forward passes: one with all modalities and one for each modality replaced.

        Args:
            inputs (list of tensors): List of modality inputs.

        Returns:
            List of outputs from the forward passes.
        """
        outputs = []
        if self.fusion_sparse:
            gate_losses = []

            output, gate_loss = self.forward(inputs)
            outputs.append(output)
            gate_losses.append(gate_loss)

            for i in range(len(inputs)):
                output, gate_loss = self.forward_with_replacement(
                    inputs, replace_index=i
                )
                outputs.append(output)
                gate_losses.append(gate_loss)

            return outputs, gate_losses
        else:
            outputs.append(self.forward(inputs))
        # print(len(outputs),outputs[0].shape)  # 1 torch.Size([64, 25])  两种模态的融合moe特征

        # Forward passes with each modality replaced
        for i in range(len(inputs)):
            outputs.append(self.forward_with_replacement(inputs, replace_index=i))  # 两种模态的融合moe特征 再加上 对两种模态中的一种替换随机向量再经过moe的特征 

        # print(len(outputs),outputs[0].shape)  # 3 torch.Size([64, 25]
        return outputs



class InteractionMoE(nn.Module):
    def __init__(
        self,
        num_modalities=3,
        fusion_model=None,
        fusion_sparse=True,
        hidden_dim=256,
        hidden_dim_rw=256,
        num_layer_rw=2,
        temperature_rw=1,
    ):
        super(InteractionMoE, self).__init__()
        num_branches = num_modalities + 1 + 1  # uni + syn + red
        self.num_modalities = num_modalities
        self.reweight = MLPReWeighting(
            num_modalities,
            num_branches,
            hidden_dim=hidden_dim,
            hidden_dim_rw=hidden_dim_rw,
            num_layers=num_layer_rw,
            temperature=temperature_rw,
        )
        self.interaction_experts = nn.ModuleList(
            [
                InteractionExpert(deepcopy(fusion_model), fusion_sparse)
                for _ in range(num_branches)
            ]
        )
        self.fusion_sparse = fusion_sparse

    def uniqueness_loss_single(self, anchor, pos, neg):
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        return triplet_loss(anchor, pos, neg)

    def synergy_loss(self, anchor, negatives):
        total_syn_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=1)
        for negative in negatives:
            negative_normalized = F.normalize(negative, p=2, dim=1)
            cosine_sim = torch.sum(anchor_normalized * negative_normalized, dim=1)
            total_syn_loss += torch.mean(cosine_sim)
        total_syn_loss = total_syn_loss / len(negatives)

        return total_syn_loss  # Synergy loss

    def redundancy_loss(self, anchor, positives):
        total_redundancy_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=1)
        for positive in positives:
            positive_normalized = F.normalize(positive, p=2, dim=1)
            cosine_sim = torch.sum(anchor_normalized * positive_normalized, dim=1)
            total_redundancy_loss += torch.mean(1 - cosine_sim)
        total_redundancy_loss = total_redundancy_loss / len(positives)
        return total_redundancy_loss  # Redundancy loss

    def forward(self, inputs):

        expert_outputs = []

        for expert in self.interaction_experts:
            expert_outputs.append(expert.forward_multiple(inputs))  # 应该是四个一样的3维，但参数不同  （3维：两种模态的融合moe特征 + 对两种模态中的一种替换随机向量再经过moe的特征 ）
        '''
        参数是否相同：
        代码里用的是 deepcopy(fusion_model)，所以每个专家都有一套独立参数。刚初始化时参数相同（深拷贝了同一初值），因此最初的输出可能很接近（若有 dropout 也可能略有差异）；
        训练一段时间后，因为每个分支承担的交互损失不同（唯一性/协同/冗余）且还有样本自适应的 reweighting 反向梯度，各专家会逐渐“分化/专精”，输出也会不同。

        如果没有 deepcopy 会怎样：
        如果误写成直接复用同一个 fusion_model 实例（不 deep copy），那各专家会共享同一套权重，输出就会高度一致，分支很难学到不同的交互功能。这正是这里用 deepcopy 的原因。
        '''
        
        # print(len(expert_outputs),len(expert_outputs[0]), expert_outputs[0][0].shape)  # 4 3 ([64, 25])

        ###### Define interaction losses ######
        # First n experts are uniqueness interaction expert
        uniqueness_losses = []
        for i in range(self.num_modalities):
            uniqueness_loss = 0
            outputs = expert_outputs[i]
            anchor = outputs[0]
            neg = outputs[i + 1]
            positives = outputs[1 : i + 1] + outputs[i + 2 :]
            for pos in positives:
                uniqueness_loss += self.uniqueness_loss_single(anchor, pos, neg)
            uniqueness_losses.append(uniqueness_loss / len(positives))

        # One Synergy Expert
        synergy_output = expert_outputs[-2]
        synergy_anchor = synergy_output[0]
        synergy_negatives = torch.stack(synergy_output[1:])
        synergy_loss = self.synergy_loss(synergy_anchor, synergy_negatives)

        # One Redundacy Expert
        redundancy_output = expert_outputs[-1]
        redundancy_anchor = redundancy_output[0]
        redundancy_positives = torch.stack(redundancy_output[1:])
        redundancy_loss = self.redundancy_loss(redundancy_anchor, redundancy_positives)

        interaction_losses = uniqueness_losses + [synergy_loss] + [redundancy_loss]

        all_logits = torch.stack([output[0] for output in expert_outputs], dim=1)

        ###### MLP reweighting the experts output ######
        interaction_weights = self.reweight(inputs)  # Get interaction weights
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_logits = (all_logits * weights_transposed).sum(dim=1)

        # print(weighted_logits.shape)  # [64, 25]
        return (
            expert_outputs,
            interaction_weights,
            weighted_logits,
            interaction_losses,
        )

    def inference(self, inputs):
        # Get outputs for each interaction type
        expert_outputs = []

        for expert in self.interaction_experts:
            expert_outputs.append(expert.forward(inputs))

        all_logits = torch.stack(expert_outputs, dim=1)

        interaction_weights = self.reweight(inputs)  # Get interaction weights
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_logits = (all_logits * weights_transposed).sum(dim=1)

        return expert_outputs, interaction_weights, weighted_logits
