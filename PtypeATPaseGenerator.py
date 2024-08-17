import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pymol import cmd
from layers.StateConditioner import *
from layers.MembraneConstraintor import *
from layers.layers import *
from chroma.models import Chroma
from chroma.layers.structure import conditioners
from chroma import Chroma, Protein, conditioners, api

class PtypeATPaseGenerator():
    def __init__(
        self,   
        protein_path,
        state_label = [1,0,0,0],    #[E1,E1P,E2P,E2]
        check_protein = True,  # Check  orientation and atom.Only useful to PtypeATPase
        device = 'cuda:1',
        classifier_model = 'Model/model_weight.pt',
        classifier_weight = 1,
        classifier_max_norm = 25,
        seq_weight = 0.5,  
        membrane_constraintor = True,
        expect_rmsd = 4,
        membrane_weight = 0.1,  #recommend less than 0.25
        debug = False
     ):
        self.device = device
        self.check_protein = check_protein
        self.membrane_constraintor = membrane_constraintor
        #initial chroma
        print("Initial Chroma")
        _chroma = Chroma(device = device)   
        self.chroma = _chroma
        self.chroma.eval()
        seq_model = _chroma.design_network
        back_model = _chroma.backbone_network
        
        #Get and check protein
        print("Get and check protein")
        if check_protein:
            self.protein0 = self.CheckProtein(protein_path)
        else:
            self.protein0 = Protein(protein_path,device = device)
        xcs = self.protein0.to_XCS()
        min_z = torch.min(xcs[0][0,...,2]).item() 
        
        # load model
        print("Loading model")
        m = ATPaseClassifier(device = device)
        m.load_state_dict(torch.load(classifier_model))
        self.stateconditioner = ATPaseConditioner(
            modelweight = [1],
            models = [m],
            device = device,
            label = state_label,
            weight = classifier_weight,
            max_norm =classifier_max_norm,
            renormalize_grad = False,
            debug = debug)
        
        self.seqconditioner = conditioners.SubsequenceConditioner(protein = self.protein0 ,design_model=seq_model,weight = seq_weight) 
        
        if membrane_constraintor:
            self.constraintor = MembraneConstraintor(
                expect_rmsd = expect_rmsd,
                weight = membrane_weight,
                weight_max =3,
                protein = self.protein0, 
                backbone_model = back_model,
                rg=True,
                selection = f"z < {min_z + 45} and z > {min_z + 5}",
                debug = False
                )
    
    def CheckProtein(self,prot_path):
        dir_path = os.path.dirname(prot_path)
        if dir_path == "" :
            dir_path = "."
        base_name = os.path.basename(prot_path)
        base_name = "." + base_name
        new_prot_path = dir_path + '/' + base_name
        ref_path = 'Dataset/PtypeATPasePDBs_Enhanced/1iwo.pdb'
        cmd.load(prot_path,'prot')
        cmd.load(ref_path,'ref')
        cmd.remove("hetatm")
        cmd.align('prot','ref')
        cmd.save(new_prot_path,"prot")
        protein0 = Protein(new_prot_path,device = self.device)
        cmd.delete("ref")
        cmd.delete("prot")
        os.remove(new_prot_path)
        return protein0
    

    def SetStatelabel(self,label):
        self.stateconditioner.label = label
        
    def SetConditionerWeight(self,classifier_weight,seq_weight,membrane_weight):
        self.stateconditioner.weight =classifier_weight
        self.seqconditioner.weight = seq_weight
        self.constraintor.weight = membrane_weight
    
    def GenerateMultistate(
        self,
        t = 0.625 ,
        steps = 500,
        ):
        
        if self.membrane_constraintor:
            self.composeconditioner = conditioners.ComposedConditioner([self.stateconditioner,self.seqconditioner,self.constraintor])
        else:
            self.composeconditioner = conditioners.ComposedConditioner([self.stateconditioner,self.seqconditioner])
        self.chroma.eval()
        ATPase = self.chroma.sample( 
            full_output = False,
            initialize_noise=True, 
            protein_init = self.protein0, 
            conditioner = self.composeconditioner, 
            langevin_factor=2, 
            inverse_temperature=10, 
            tspan = [t,0.001], 
            sde_func = "ode", 
            design_method = None, 
            steps = steps)
        return ATPase
    
    def ClassifyMultistate(self,t=0,protein = None):
        if protein:
            p = Protein(protein,device = self.stateconditioner.device)
        else:
            p = self.protein0
        xcs = p.to_XCS()
        return F.softmax(self.stateconditioner.proclass_models[0](xcs,t),dim=-1)
