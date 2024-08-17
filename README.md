# Generating Multi-state Conformations of Ptype ATPases with a Diffusion Model

Understanding and predicting the diverse conformational states of membrane proteins is essential for elucidating their biological functions. Despite advancements in computational methods, accurately capturing these complex structural changes remains a significant challenge. 
In this study, we introduce a method for predicting diverse functional states of membrane protein conformations using a diffusion model. Our approach integrates forward and backward diffusion processes, incorporating state classifiers and additional conditioners to control the generation gradient of conformational states. We specifically target the P-type ATPases, a key membrane transporter, for which we curated and expanded a structural dataset. By employing a graph neural network with a custom membrane constraint, our model generates precise structures for P-type ATPases across different functional states. 
This approach represents a significant step forward in computational structural biology and holds great potential for studying the dynamics of other membrane proteins.

# Installation

First make sure you have all dependencies installed by running 

```
pip install -r requirements.txt
```
You also need to install pymol through conda due to no way to download it through pip.
```
conda install conda-forge::pymol-open-source
```

Our model is built on Chroma, so you'll need to register for an access token at Chroma Weights.

Note: Due to GitHub's file size limitations (<25MB), the code provided here is incomplete. We plan to share the full code on Zenodo.
# Usage

~~~python
# import package
from PtypeATPaseGenerator import *

# If it's your first time to use PtypeATPaseGenerator, then
# from chroma import api
# api.register_key("fdb2b9ae7e2744d1ad826cd622dc76dd") # put your token here

# Create a generator
generator = PtypeATPaseGenerator(protein_path="yourATPase.pdb",
                                 device = 'cuda',
                                 state_label = [1.0,0,0,0],    #Give a specific state for generation,  [E1,E1P,E2P,E2]
                                 classifier_weight = 0.75,  #The weight of classifier conditioner
                                 classifier_max_norm = 25,
                                 seq_weight = 0.5,  #The weight of sequence conditioner
                                 membrane_constraintor = True,
                                 expect_rmsd = 4,
                                 membrane_weight = 0.1  #The weight of sequence conditioner, recommended less than 0.25
                                )
#Classify the state of Ptype-ATPase into [E1,E1P,E2P,E2]
generator.ClassifyMultistate()

# Reset the state label    
generator.SetStatelabel([0,1.0,0,0]) #Give a specific state for generation,  [E1,E1P,E2P,E2]

# Reset the weight of conditioners
generator.SetConditionerWeight(1.25,0.75,0.1) #Give classifier_weight,seq_weight,membrane_weight respectively

#Generate the specific state for Ptype-ATPase
ATPase = generator.GenerateMultistate(
                            t = 0.635 ,  # The noise scale belong to (0,1)
                            steps = 500,  # The number of denoising step
                            )
# Save ATPase
ATPase.to("saved_path.pdb")
~~~

