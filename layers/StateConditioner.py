from chroma.models import Chroma, graph_design, graph_classifier,graph_backbone
from chroma import Chroma, Protein, conditioners, api
from chroma.layers.attention import AttentionChainPool
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from layers.ATPaseDataset import *
from layers.layers import *
from typing import Optional, Tuple, Union
from chroma.models.graph_classifier import GraphClassifier
from chroma.layers.structure.diffusion import GaussianNoiseSchedule
class ATPaseClassifier(nn.Module):      
     
    def __init__(self, dim_nodes = 512, n_head = 16, dropout = 0.1,device = 'cuda',MLP_dim = 512 * 2):
        super(ATPaseClassifier,self).__init__()
        self.device = device
        self.classifier = graph_classifier.load_model(weight_file = "named:public",device = self.device) 
        #self.classifier = graph_classifier.GraphClassifier()
        self.pool = AttentionChainPool(n_head, dim_nodes)
        self.head = nn.Sequential(
                    nn.Linear(dim_nodes, 64),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 64),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 4),
                )


    def forward(self, XCO, t):
        X ,C, O= XCO[0], XCO[1], XCO[2]
        
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.classifier.encode(
            X, C, None, t
        )
        node_h, c = self.pool(node_h,C)
        node_h = self.head(node_h)
        return node_h
class ATPaseConditioner(conditioners.Conditioner):
   
    def __init__(
        self,
        label: list = [1,1,1,1],
        models: Union[GraphClassifier, str] = None,
        weight: float = 5,
        max_norm: Optional[float] = 20,
        renormalize_grad: Optional[bool] = False,
        use_sequence: bool = True,
        device: Optional[str] = None,
        debug: bool = False,
        modelweight = [0.4,0.6],
    ) -> None:
        super().__init__()
        self.max_norm = max_norm
        self.renormalize_grad = renormalize_grad
        self.weight = weight
        self.use_sequence = use_sequence
        self.debug = debug
        self.schedule =  GaussianNoiseSchedule()
        self.modelweight = modelweight
        # Move Model to the indicated device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        self.label = torch.tensor(label,device = self.device)  #{a,b,c,1-a-b-c} 
        self.proclass_models = []
        for model in models:
            model.to(self.device)
            model.eval()
            self.proclass_models.append(model)
        
        self.logp = []
        self.states_acc = {"E1":[],"E1P":[],"E2P":[],"E2":[]}
        
    def _transform_gradient(self, grad, C, t):

        if grad.norm() > 1e-8:  # Don't rescale zero gradients!
            if self.max_norm is not None:
                if grad.norm() > self.max_norm:
                    grad = self.max_norm * (grad / grad.norm())
            # grad = clip_atomic_magnitudes_percentile(grad,percentile=0.95)
            if self.debug:
                print("conditioning grad norm:", grad.norm().item())
            if self.renormalize_grad:
                grad = self.weight * grad 
            else:
                grad = self.weight * grad / self.schedule.sigma(t).to(C.device)

        if self.debug:
            print("output_grad_norm", grad.norm().item())
        return grad
    def S2O( self, St):
    #St = torch.zeros(Ct.shape, device=Xt.device).long()
        Ot = F.one_hot(St, 20).float()
        return Ot

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
        
        X_input = X + 0.0
        X_input.register_hook(lambda _X: self._transform_gradient(_X, C, t))
        output = 0
        for i in range(len(self.proclass_models)):
            model = self.proclass_models[i]
            w = self.modelweight[i]
            score = model(
                [X_input, C, O],t
            )
            softmax = score.softmax(dim=-1) 
            output = output + w * softmax
        log = torch.log(output)
        neglogp = torch.sum(log * torch.tensor(self.label,device = self.device), dim = -1)*(-1)   
                
        if self.debug:
            acc = output.clone().cpu().detach().numpy()
            i= 0
            import matplotlib.pyplot as plt
            for key in self.states_acc:
                self.states_acc[key].append(acc[...,i].item())
                plt.plot(self.states_acc[key],label = key)
                i +=1
            plt.xlabel('step')
            plt.ylabel('p')
            plt.legend()
            plt.show()
            print("time", t.item(), "neglogp:", neglogp.item(),"U:" ,U)
        return X, C, O, neglogp + U, t