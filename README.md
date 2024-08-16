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
