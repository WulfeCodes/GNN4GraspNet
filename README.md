##  GNN
This code implements a training algorithm for a GNN of GraspNet Graspability features for manipulators of object point cloud data

# To Train and save: 
1.simply create a parent folder, download zShotGraspGNN.py,requirements.txt,and https://drive.google.com/file/d/1FCV6j2J2eQpVk_ddJXljJvjRT1KU3sJ6/view

2. pip install -r requirements.txt
3. run program :)

# For Inference
4. Once Weights are saved Call:
label_data=load("./PathToData.npz"),
positions,scores,offsets=GraphNet.sampleIndexes(label_data,label_data['collision'].shape,k),
GraphNet.makeGraph(positions,scores,offsets),
pred=GraphNet.forward()
