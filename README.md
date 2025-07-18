##  GNN
This code implements a training algorithm for a GNN of GraspNet Graspability features for manipulators of object point cloud data

1. Download zShotGraspGNN.py
2. Download weights from edgeconv_weights.pth
3. Download the training data from: https://drive.google.com/drive/folders/12X2I8hkwpsmg_oIjCNRRNnyTBvg7o_Dz?usp=sharing
4. pip install -r requirements.txt
5. run program :)

### Minimal Training and Inference run: 

#### loading_weights isn't required if you'd like to train from scratch

#### train4Epochs isn't required if you'd like to just run inference
train4Epochs includes augmented data generation with minimal noise, translation, and rotation btw
<img width="1123" height="287" alt="image" src="https://github.com/user-attachments/assets/bbf256d6-8c3c-4f1a-a473-b7d6abed62ce" />

