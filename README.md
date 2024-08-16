
# Generating Multi-state Conformations of P-type ATPases with a Diffusion Model

Understanding and predicting the various conformational states of membrane proteins is crucial for uncovering their biological functions. Although computational methods have advanced, accurately modeling these complex structural transitions remains challenging.  
In this study, we present a novel method for predicting diverse functional states of membrane protein conformations using a diffusion model. Our approach combines forward and backward diffusion processes and integrates state classifiers and additional conditioners to modulate the generation gradient of conformational states. We focus on P-type ATPases, a key class of membrane transporters, for which we have curated and expanded a comprehensive structural dataset. Leveraging a graph neural network with a custom membrane constraint, our model generates accurate structures for P-type ATPases across different functional states.  
This method marks a significant advancement in computational structural biology and offers promising potential for studying the dynamics of other membrane proteins.

# Installation

To get started, ensure all dependencies are installed by running:

```
pip install -r requirements.txt
```

Our model is built on Chroma, so you'll need to register for an access token at [Chroma Weights](https://chroma-weights.generatebiomedicines.com/).

# Note: Due to GitHub's file size limitations (<25MB), the code provided here is incomplete. We plan to share the full code on Zenodo.

# An example of usage

~~~python
# import package
from PtypeATPaseGenerator import *
from chroma import api
api.register_key("fdb2b9ae7e2744d1ad826cd622dc76dd") # put your token here

#Create a generator
generator = PtypeATPaseGenerator(protein_path="yourATPase.pdb",
                                 device = 'cuda',
                                 state_label = [1,0,0,0],    #Give a specific state for generation,  [E1,E1P,E2P,E2]
                                 classifier_weight = 0.75,  #The weight of classifier conditioner
                                 classifier_max_norm = 25,
                                 seq_weight = 0.5,  #The weight of sequence conditioner
                                 membrane_constraintor = True,
                                 expect_rmsd = 4,
                                 membrane_weight = 0.1  ##The weight of sequence conditioner, recommended less than 0.25
                                )
#Classify the state of Ptype-ATPase into [E1,E1P,E2P,E2]
generator.ClassifyMultistate()

# Reset the state label    
generator.SetStatelabel([E1,E1P,E2P,E2])

# Reset the weight of conditioners
generator.SetConditionerWeight(classifier_weight,seq_weight,membrane_weight)

#Generate the specific state for Ptype-ATPase
generator.GenerateMultistate(
                            t = 0.625 ,  #The noise scale belong to (0,1)
                            steps = 500,  #The number of denoising step
                            trajectory_length=500,
                            full_output = True,  
                            )

# Save ATPase
ATPase.to("generated.pdb")
