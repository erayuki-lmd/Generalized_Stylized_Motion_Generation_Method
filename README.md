# Generalized_Stylized_Motion_Generation_Method
Published in **IEEE Transactions on Multimedia**
paper:  
Movie: 
Webpage: 

# Abstruct
GeM2 aims to extend the applicability of stylized motion generation methods to be robust for large and diverse motions akin to those found in real-world data. Specifically, we introduce metadata-independent learning alongside style-focused learning, thereby enabling training from motions absent in motion-style datasets. In addition, we construct a novel motion dataset containing both various motions and stylized motions by unifying the multiple datasets to effectively train the model. Our novel learning method and dataset enable stylized motion generation methods to learn from both various motion knowledge and motion-style relations and improve their generalized performance.

# Directries
 loader : Directry of dataloader
 model  : Directry of model structure
 utils  : Directry of utility codes
 dataset: Directry of dataset (Created with reference to the following)

 # codes
 1_pretrain.py: Pre-training the Style Encoder
 2_train.py   : Training other networks
 3_Gen.py     : Generating motions 
 
 
