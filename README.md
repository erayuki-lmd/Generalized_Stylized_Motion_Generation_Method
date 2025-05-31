
![title](https://github.com/user-attachments/assets/2a0bf6af-5e67-43a3-92d7-6025b9ad61e5)


# Generalized_Stylized_Motion_Generation_Method
Published in **IEEE Transactions on Multimedia**
<br>
paper:
<br>
Movie:
<br>
Webpage: 

# Abstruct
![idea](https://github.com/user-attachments/assets/5faffa9b-0479-4e4a-a98f-edf7a3ccc553)
GeM2 aims to extend the applicability of stylized motion generation methods to be robust for large and diverse motions akin to those found in real-world data. Specifically, we introduce metadata-independent learning alongside style-focused learning, thereby enabling training from motions absent in motion-style datasets. In addition, we construct a novel motion dataset containing both various motions and stylized motions by unifying the multiple datasets to effectively train the model. Our novel learning method and dataset enable stylized motion generation methods to learn from both various motion knowledge and motion-style relations and improve their generalized performance.

# Demo
Comming Soon

# Method
Training
![training](https://github.com/user-attachments/assets/5af31282-0961-4d2c-bb56-d432c3bf82d4)
Genetration
![generation](https://github.com/user-attachments/assets/429fc5b3-e9e5-48bb-b8f0-62aa6798c551)


# Directries
 loader  : Directry of dataloader
 <br>
 model   : Directry of model structure
 <br>
 utils   : Directry of utility codes
 <br>
 dataset : Directry of dataset (Created with reference to the following)

 # Codes
 1_pretrain.py : Pre-training the Style Encoder
 <br>
 2_train.py    : Training other networks
 <br>
 3_Gen.py      : Generating motions 
 <br>
 GeM_conda.yml : Conda enviroments
 
 # Creating Dataset
1. Directry Construction
```bash
   dataset
   ├── train
   │    ├── style: Motions with style labels
   │    │    ├─ direction: Bone-directions (Size = T×54)
   │    │    ├─ position : Joint-positions (Size = T×54)
   │    │    ├─ velocity : Joint-velocities (Size = T×54)
   │    │    ├─ style    : Style-labels (Size = T×S)
   │    │    └─ content  : Content-labels (Size = T×C)
   │    │
   │    └── motion: Motions without style labels
   │         ├─ direction
   │         ├─ position
   │         ├─ velocity
   │         ├─ style
   │         └─ content
   │
   └── test (Same as train costruction)

**Parameter**
T: Number of flames
S: Number of styles
C: Number of contents (do not use)

**ATTENTION**
・The data of same motion are named the same.
  (ex.)
  direction: .../direction/motion_1.npy
  position : .../position/motion_1.npy
  velocity : .../velocity/motion_1.npy
  content  : .../content/motion_1.npy
  style    : .../style/motion_1.npy

・All contents are T×C zero array (although it is not necessary for the method, it is required to run the code).

・styles of motions without style label are T×S zero array (although it is not necessary for the method, it is required to run the code).

・All train data have same T.
```

2. Joints and Bones Template
![data_template](https://github.com/user-attachments/assets/5eaf81c9-467b-4e1e-ba58-763a3adff0d9)

```bash
(Joints)
[ 0] Left Hip
[ 1] Left Leg
[ 2] Left Foot
[ 3] Left Toe
[ 4] Right Hip
[ 5] Right Leg
[ 6] Right Foot
[ 7] Right Toe
[ 8] Spine
[ 9] Chest
[10] Neck
[11] Head
[12] Left Shoulder
[13] Left Arm
[14] Left Hand
[15] Right Shoulder
[16] Right Arm
[17] Right Hand

(Bones)
[ 0] Hip            -> Left Hip       (Length:2.405)
[ 1] Left Hip       -> Left Leg       (Length:7.158)
[ 2] Left Leg       -> Left Foot      (Length:7.491)
[ 3] Left Foot      -> Left Toe       (Length:2.368)
[ 4] Hip            -> Right Hip      (Length:2.373)
[ 5] Right Hip      -> Right Leg      (Length:7.434)
[ 6] Right Leg      -> Right Foot     (Length:7.509)
[ 7] Right Foot     -> Right Toe      (Length:2.412)
[ 8] Hip            -> Spine          (Length:2.045)
[ 9] Spine          -> Chest          (Length:2.050)
[10] Chest          -> Neck           (Length:1.756)
[11] Neck           -> Head           (Length:1.769)
[12] Chest          -> Left Shoulder  (Length:3.584)
[13] Left Shoullder -> Left Arm       (Length:4.983)
[14] Left Arm       -> Left Hand      (Length:3.484)
[15] Chest          -> Right Shoulder (Length:3.448)
[16] Right Shoulder -> Right Arm      (Length:5.242)
[17] Right Arm      -> Right Hand     (Length:3.444)
```

3. Creating Dataset
![dataset](https://github.com/user-attachments/assets/37aa74bd-cc5f-4ff4-8830-8b095dcdd750)
<ul>
  <li> Extracting each joint coordinates from original datasets</li>
  <li> Calculating direction (Direction vector of norm 1) -> direction</li>
  <li> Reconstructing joint position based on bone length -> position</li>
  <li> Calculating the amount of change in position of each joint -> velocity </li>
  <li> Labelling motion data</li>
</ul>
