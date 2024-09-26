from typing import Optional, Tuple, Union
import chroma.utility.chroma
from chroma.data.protein import Protein
from chroma.data.xcs import validate_XC
from chroma.layers.structure import backbone, mvn, optimal_transport, symmetry
from chroma.layers.structure.backbone import expand_chain_map
from chroma.models import graph_classifier, procap
from chroma.models.graph_backbone import GraphBackbone
from chroma.models.graph_classifier import GraphClassifier
from chroma.models.graph_design import GraphDesign
from chroma.models.procap import ProteinCaption
from chroma.layers.structure.rmsd import BackboneRMSD
from chroma.layers.structure import conditioners
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MembraneConstraintor(conditioners.Conditioner):

    def __init__(
        self,
        protein: Protein,
        backbone_model: GraphBackbone,
        selection: str,
        debug = True,
        expect_rmsd = 6, 
        rg: bool = False,
        weight: float = 1.0,
        weight_max: float = 3.0,
        gamma: Optional[float] = None,
        center_init: Optional[bool] = True,
        loss_type: str = "mse",
    ):
        super().__init__()
        self.protein = protein
        self.debug = debug
        self.expect_rmsd = expect_rmsd
        self.backbone_model = backbone_model
        self.loss_type = loss_type
        X, C, S = protein.to_XCS()
        X = X[:, :, :4, :]
        if center_init:
            X = backbone.center_X(X, C)
        D = protein.get_mask(selection).bool()
        self.base_distribution = self.backbone_model.noise_perturb.base_gaussian
        self.noise_schedule = self.backbone_model.noise_perturb.noise_schedule
        D = D.unsqueeze(-1).unsqueeze(-1)
        X = torch.where( D, X, torch.zeros_like(X))
        self.weight = weight
        self.weight_max = weight_max
        self.rg = rg
        self.register_buffer("X", X)
        self.register_buffer("C", C)
        self.register_buffer("S", S)
        self.register_buffer("D", D)

        self.loss = []
        self.rmsd_loss =  BackboneRMSD().to(X.device)
    def _transform_gradient(self, grad, C, t):
        grad = torch.where(self.D,grad ,0)
        scale = (
            (self.weight * torch.tensor([1.0],device = C.device))
        )
        grad = (scale * grad).clamp(max=self.weight_max) / self.noise_schedule.sigma(t).to(C.device)
        if self.debug:
            print("substruct grad", grad.norm().item())
        return grad
    
    def general_chi_squ_corrected(self, mu, k, t, A=None, n_samples=10000, a=0.1):
        device = mu.device
        # 使用预计算的常数进行重参数化
        epsilon = torch.randn(n_samples, mu.size(1), mu.size(2), 3, device=device)
        C = torch.ones(n_samples, mu.size(1), device=device)

        # 如果可能，将常数移到循环外部预计算
        alpha = self.noise_schedule.alpha(t).view(-1, 1, 1, 1).to(device)
        sigma = self.noise_schedule.sigma(t).view(-1, 1, 1, 1).to(device)

        # 添加噪声并生成样本
        samples = mu + sigma * self.base_distribution._multiply_R(epsilon, C) / alpha

        # 如果没有提供A，则将A定义为单位矩阵
        if A is None:
            A = torch.eye(mu.size(-2) * mu.size(-1) * mu.size(-3), device=device)

        # 展平并计算二次型
        samples_flat = samples.view(n_samples, -1)
        quadratic_form_values = torch.sum((samples_flat @ A) * samples_flat, dim=1) / (mu.size(1) * mu.size(2))

        # 使用sigmoid计算满足条件的软条件
        soft_condition = torch.sigmoid(a * (k - quadratic_form_values))
        print("chi_count",torch.sum((k - quadratic_form_values) > 0).item())
        # 返回平均概率
        probabilities = soft_condition.mean()
        return probabilities

    def _rg_loss(self, X0, Xt, C ,t): #X0为通过Xt预测的0时刻的结构
        D2 = self.D.squeeze().repeat(1,1)
        mask = D2 == 1
        C = C[mask].repeat(1,1)
        D = self.D.repeat(1,1,4,3)
        mask = D == 1
        _X0 = X0[mask].view(X0.size()[0],-1,4,3)
        _Xt = Xt[mask].view(Xt.size()[0],-1,4,3)
        X_target = self.X[mask].view(self.X.size()[0],-1,4,3).to(X0.device)
        try:
            X_align , loss = self.rmsd_loss.align(X_target,_X0,C)
        except:
            torch.set_printoptions(profile="full")
            torch.set_printoptions(profile="default")
            return "a"
        #loss = (X_target - _X0)
        #loss = loss.square().sum() / (self.D == True).sum() /4
        if self.loss_type == "mse":
            if loss > self.expect_rmsd:
                loss = (loss - self.expect_rmsd)**2
            else:
                loss = (loss - loss)**2
        elif self.loss_type == "rg":
            alpha = self.noise_schedule.alpha(t)[:, None, None, None].to(Xt.device)
            mu = X_align - _Xt/alpha
            loss = self.general_chi_squ_corrected(mu = mu,n_samples= self.chi_num, a = self.chi_a, k = self.expect_rmsd**2,t = t)
            print("chi prob:",loss)
            loss = torch.log(loss)*(-1)
        return loss

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        loss = 0.0        
        # Reconstruction guidance
        if self.rg:
            X_input = X + 0.0
            X_input.register_hook(lambda _X: self._transform_gradient(_X, C, t))
            X_prior = self.base_distribution.sample(C)
            alpha = self.noise_schedule.alpha(t)[:, None, None, None].to(X.device)
            sigma = self.noise_schedule.sigma(t)[:, None, None, None].to(X.device)
            X0 = (X_input - sigma * X_prior) / alpha
            loss = self._rg_loss(X0, X_input, C, t)
            a = loss.clone()**0.5
            self.loss.append(a.cpu().detach().numpy())
        if self.debug:
            plt.plot(self.loss,label = 'loss')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
        U = U + loss
        return X, C, O, U, t
