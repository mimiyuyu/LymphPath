import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kan import *
import math
"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""

def pinv(A):  
    U, S, Vh = torch.svd(A) 
    S_inv = torch.zeros_like(S)  
    for i, s in enumerate(S):  
        if not torch.isclose(s, torch.tensor(0.0), atol=1e-10):  
            S_inv[i] = 1.0 / s  
    S_inv_diag = torch.diag(S_inv)  
    return Vh.t() @ S_inv_diag @ U.t()  
 
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  
        B = y.transpose(0, 1)  
        solution = torch.linalg.lstsq(
            A, B
        ).solution  
        result = solution.permute(
            2, 0, 1
        )  

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x) 
        splines = splines.permute(1, 0, 2)  
        orig_coeff = self.scaled_spline_weight 
        orig_coeff = orig_coeff.permute(1, 2, 0) 
        unreduced_spline_output = torch.bmm(splines, orig_coeff) 
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D), 
            nn.Tanh()]

        if dropout: 
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes)) 

        self.module = nn.Sequential(*self.module) 

    def forward(self, x):
        return self.module(x), x  


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))


        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  
        return A, x

class Attn_Net_Gated_kan(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated_kan, self).__init__()
        self.attention_a = [
            KANLinear(L,D),
            nn.Tanh()]

        self.attention_b = [
            KANLinear(L,D),
            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class LymphPath(nn.Module):
    def __init__(self, gate=True, ifkan=False, size_arg1="gigapath", size_arg2="uni", size_arg3="virchow2", dropout=True, k_sample=3, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), loss1=0.4, mergeloss=0.6, subtyping=False, **kwargs):
        super(LymphPath, self).__init__()
        self.size_dict = {'xs': [384, 256, 256], 's': [512, 256, 256], "small": [768, 512, 256], "uni": [1024, 512, 384], 'gigapath': [1536, 768, 512], 'large': [2048, 1024, 512], 'virchow2':[2560, 1280, 512]}
        size1 = self.size_dict[size_arg1]
        size2 = self.size_dict[size_arg2]
        size3 = self.size_dict[size_arg3]

        fc1 = [nn.Linear(size1[0], size1[1]), nn.ReLU()]
        fc2 = [nn.Linear(size2[0], size2[1]), nn.ReLU()]
        fc3 = [nn.Linear(size3[0], size3[1]), nn.ReLU()]

        if dropout:
            fc1.append(nn.Dropout(0.25))
            fc2.append(nn.Dropout(0.25))
            fc3.append(nn.Dropout(0.25))
        if gate:
            if ifkan:
                print("Using KAN")
                attention_net1 = Attn_Net_Gated_kan(L=size1[1], D=size1[2], dropout=dropout, n_classes=1) 
                attention_net2 = Attn_Net_Gated_kan(L=size2[1], D=size2[2], dropout=dropout, n_classes=1) 
                attention_net3 = Attn_Net_Gated_kan(L=size3[1], D=size3[2], dropout=dropout, n_classes=1) 
            else:
                attention_net1 = Attn_Net_Gated(L=size1[1], D=size1[2], dropout=dropout, n_classes=1) 
                attention_net2 = Attn_Net_Gated(L=size2[1], D=size2[2], dropout=dropout, n_classes=1) 
                attention_net3 = Attn_Net_Gated(L=size3[1], D=size3[2], dropout=dropout, n_classes=1) 
        else:
            attention_net1 = Attn_Net(L=size1[1], D=size1[2], dropout=dropout, n_classes=1)
            attention_net2 = Attn_Net(L=size2[1], D=size2[2], dropout=dropout, n_classes=1)
            attention_net3 = Attn_Net(L=size3[1], D=size3[2], dropout=dropout, n_classes=1)

        fc1.append(attention_net1)
        fc2.append(attention_net2)
        fc3.append(attention_net3)

        self.attention_net1 = nn.Sequential(*fc1)
        self.attention_net2 = nn.Sequential(*fc2)
        self.attention_net3 = nn.Sequential(*fc3)

        self.classifiers1 = nn.Linear(size1[1], n_classes)
        instance_classifiers1 = [nn.Linear(size1[1], 2) for i in range(n_classes)]
        self.instance_classifiers1 = nn.ModuleList(instance_classifiers1)  
        
        self.classifiers2 = nn.Linear(size2[1], n_classes)
        instance_classifiers2 = [nn.Linear(size2[1], 2) for i in range(n_classes)]
        self.instance_classifiers2 = nn.ModuleList(instance_classifiers2) 

        self.classifiers3 = nn.Linear(size3[1], n_classes)
        instance_classifiers3 = [nn.Linear(size3[1], 2) for i in range(n_classes)]
        self.instance_classifiers3 = nn.ModuleList(instance_classifiers3)  

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.loss1 = loss1
        self.mergeloss = mergeloss

        self.fusion_mlp = nn.Sequential(  
            nn.Linear(3 * self.n_classes, 512),  
            nn.ReLU(),  
            nn.Dropout(0.25),  
            nn.Linear(512, self.n_classes)  
        )  

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net1 = self.attention_net1.to(device)
        self.attention_net2 = self.attention_net2.to(device)
        self.attention_net3 = self.attention_net3.to(device)
        self.classifiers1 = self.classifiers1.to(device)
        self.classifiers2 = self.classifiers2.to(device)
        self.classifiers3 = self.classifiers3.to(device)
        self.instance_classifiers1 = self.instance_classifiers1.to(device)
        self.instance_classifiers2 = self.instance_classifiers2.to(device)
        self.instance_classifiers3 = self.instance_classifiers3.to(device)

    @staticmethod
    def create_positive_targets(length, device): 
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        if len(h) < 10 * self.k_sample:
            k = 1
        else:
            k = self.k_sample
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k, device)
        n_targets = self.create_negative_targets(k, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, feature1, feature2, feature3, label=None, instance_eval=False, return_features=False, attention_only=False):
        A1, feature1 = self.attention_net1(feature1)  
        A2, feature2 = self.attention_net2(feature2)  
        A3, feature3 = self.attention_net3(feature3) 
        A1_raw = A1
        A2_raw = A2
        A3_raw = A3
        A1 = torch.transpose(A1, 1, 0) 
        A2 = torch.transpose(A2, 1, 0)  
        A3 = torch.transpose(A3, 1, 0)  
        A1 = F.softmax(A1, dim=1)  
        A2 = F.softmax(A2, dim=1)  
        A3 = F.softmax(A3, dim=1)  

        if instance_eval:
            total_inst_loss1 = 0.0
            all_preds1 = []
            all_targets1 = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  
            for i in range(len(self.instance_classifiers1)):
                inst_label = inst_labels[i].item()
                classifier1 = self.instance_classifiers1[i]
                if inst_label == 1:  
                    instance_loss1, preds1, targets1 = self.inst_eval(A1, feature1, classifier1)
                    all_preds1.extend(preds1.cpu().numpy())
                    all_targets1.extend(targets1.cpu().numpy())
                else: 
                    if self.subtyping:
                        instance_loss1, preds1, targets1 = self.inst_eval_out(A1, feature1, classifier1)
                        all_preds1.extend(preds1.cpu().numpy())
                        all_targets1.extend(targets1.cpu().numpy())
                    else:
                        continue
                total_inst_loss1 += instance_loss1

            if self.subtyping:
                total_inst_loss1 /= len(self.instance_classifiers1)

        if instance_eval:
            total_inst_loss2 = 0.0
            all_preds2 = []
            all_targets2 = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  
            for i in range(len(self.instance_classifiers2)):
                inst_label = inst_labels[i].item()
                classifier2 = self.instance_classifiers2[i]
                if inst_label == 1:  
                    instance_loss2, preds2, targets2 = self.inst_eval(A2, feature2, classifier2)
                    all_preds2.extend(preds2.cpu().numpy())
                    all_targets2.extend(targets2.cpu().numpy())
                else:  
                    if self.subtyping:
                        instance_loss2, preds2, targets2 = self.inst_eval_out(A2, feature2, classifier2)
                        all_preds2.extend(preds2.cpu().numpy())
                        all_targets2.extend(targets2.cpu().numpy())
                    else:
                        continue
                total_inst_loss2 += instance_loss2

        if instance_eval:
            total_inst_loss3 = 0.0
            all_preds3 = []
            all_targets3 = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  
            for i in range(len(self.instance_classifiers3)):
                inst_label = inst_labels[i].item()
                classifier3 = self.instance_classifiers3[i]
                if inst_label == 1:  
                    instance_loss3, preds3, targets3 = self.inst_eval(A3, feature3, classifier3)
                    all_preds3.extend(preds3.cpu().numpy())
                    all_targets3.extend(targets3.cpu().numpy())
                else: 
                    if self.subtyping:
                        instance_loss3, preds3, targets3 = self.inst_eval_out(A3, feature3, classifier3)
                        all_preds3.extend(preds3.cpu().numpy())
                        all_targets3.extend(targets3.cpu().numpy())
                    else:
                        continue
                total_inst_loss3 += instance_loss3

            if self.subtyping:
                total_inst_loss3 /= len(self.instance_classifiers3)

            total_inst_loss = (total_inst_loss1 + total_inst_loss2 + total_inst_loss3) / 3

        M1 = torch.mm(A1, feature1)  
        M2 = torch.mm(A2, feature2)  
        M3 = torch.mm(A3, feature3)  

        logits1 = self.classifiers1(M1) 
        logits2 = self.classifiers2(M2)  
        logits3 = self.classifiers3(M3) 
        combined_logits = torch.cat([logits1, logits2, logits3], dim=1)  
        final_logits = self.fusion_mlp(combined_logits)  

        y_prob1 = torch.softmax(logits1, dim=-1)
        y_prob2 = torch.softmax(logits2, dim=-1)
        y_prob3 = torch.softmax(logits3, dim=-1)
        y_prob_merge = torch.softmax(final_logits, dim=-1)
        y_prob = self.loss1 * (y_prob1 + y_prob2 + y_prob3) / 3 + self.mergeloss * y_prob_merge

        result = {
            'logits1': logits1,
            'logits2': logits2,
            'logits3': logits3,
            'merge_logits': final_logits,
            'attention_raw1': A1_raw,
            'attention_raw2': A2_raw,
            'attention_raw3': A3_raw,
            'M1': M1,
            'M2': M2,
            'M3': M3,
            'y_prob': y_prob
        }

        if instance_eval:
            results_dict = {'instance_loss1': total_inst_loss, 'inst_labels1': np.array(all_targets1),
                            'inst_preds1': np.array(all_preds1)}
            result['inst_loss'] = total_inst_loss
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': final_logits})
        return result
