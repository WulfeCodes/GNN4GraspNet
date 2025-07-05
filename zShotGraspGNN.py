import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Sequential, Linear, ReLU
from numpy import load
import torch.optim as optim
from pathlib import Path
from torch_geometric.nn import MLP, knn_graph,EdgeConv
from torch_geometric.data import Data
import os
import numpy as np
from torch_geometric.transforms import RandomRotate, RandomScale, RandomTranslate
#TODO create augmented data, verify forward + train pass
#{EdgeConv} and graphnetConv
class EdgeConvNet(nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        nn1 = Sequential(
            Linear(2 * in_channels, 64),
            ReLU(),
            Linear(64, 64)
        )
        nn2 = Sequential(
            Linear(2 * 64, 64),
            ReLU(),
            Linear(64, 64)
        )

        self.conv1 = EdgeConv(nn1)
        self.edge_index=None
        self.data = None
        self.conv2 = EdgeConv(nn2)
        self.lin = Linear(64, 1)  # Output grasp quality scalar per node
        self.optimizer = optim.ASGD(self.parameters(),lr=1e-4)
    def forward(self,data):
          # x: (N, in_channels)
        x,edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        out = self.lin(x)  # (N, 1)

        return out.squeeze(-1)  # (N,)
    def train(self):
        total_loss = 0
        out=self.forward(self.data)
        self.optimizer.zero_grad()       
        loss=F.mse_loss(out,self.data.y)
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item() * self.data.num_nodes
        print("finished training:",total_loss)

    def save_weights(self, path="./edgeconv_weights.pth"):
        torch.save(self.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load_weights(self, path="./edgeconv_weights.pth", map_location=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No weights file found at {path}")
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        self.eval()
        print(f"Model weights loaded from {path}")

    def makeGraph(self,positions,scores,offsets):
        print(f"Dtype checks: {type(positions)},{type(scores)},{type(offsets)}")
        print(f"Lengths check: {positions.shape},{scores.shape},{offsets.shape}")
            # goal: make k,3 of pos
            #goal: make k,1 of scoress
            #goal make concated k,9 : [i]=pos,offset,score
            #         for b,indx in enumerate(bestScores):
        x = torch.cat([positions, offsets, scores], dim=1)
        
        positions = positions.float()
        offsets = offsets.float()
        scores = scores.float()
        x = x.float()
        
        self.edge_index = knn_graph(positions, k=10,loop=False)
        self.data = Data(x=x, pos=positions,edge_index=self.edge_index,y=x[:,-1])

    def sampleIndexes(self,label_data,DataShape,k):
        print(f"length check: {len(DataShape)},{DataShape}")
        shape = (len(DataShape),k)
        samplers = torch.empty(shape,dtype=torch.int32)

        for i, currShape in enumerate(DataShape):
            samplers[i]=torch.tensor(np.random.randint(0,currShape,size=k),dtype=torch.int32)

        sampledIndexes=samplers.T

        positions=torch.empty((k,3),dtype=float)
        scores = torch.empty((k,1),dtype=float)
        offsets = torch.empty((k,3),dtype=float)

        for i,row in enumerate(sampledIndexes):

            if(not label_data['collision'][row[0],row[1],row[2]].any()):
                positions[i]=torch.from_numpy((label_data['points'][row[0]]).T)
                scores[i]=float(label_data['scores'][row[0],row[1],row[2],row[3]])
                offsets[i]=torch.from_numpy(-(label_data['offsets'][row[0],row[1],row[2],row[3]]).T)
            else:
                collisions = True
                while(collisions==True):
                    idxList = []
                    for j, idxRange in enumerate(label_data['collision'].shape):
                        idxList.append(np.random.randint(0,idxRange))
                    if label_data['collision'][idxList[0],idxList[1],idxList[2],idxList[3]].any():
                        continue
                    else:
                        positions[i]=torch.from_numpy(label_data['points'][idxList[0]].T)
                        scores[i]=float(label_data['scores'][idxList[0],idxList[1],idxList[2],idxList[3]])
                        offsets[i]=torch.from_numpy(-(label_data['offsets'][idxList[0],idxList[1],idxList[2],idxList[3]]).T)
                        collisions=False
            print(i)       
        return positions,scores,offsets

def main():
    k = 100
    GraphNet = EdgeConvNet(in_channels=7)

    for i, npz_str in enumerate(os.listdir("./grasp_label")):
        label_data = load(os.path.join("./grasp_label",npz_str))

        positions,scores,offsets=GraphNet.sampleIndexes(label_data,label_data['collision'].shape,k)

        print("finished calculating best coords")
        GraphNet.makeGraph(positions,offsets,scores)
        GraphNet.train()
    GraphNet.save_weights()
  #   print(data[i][len(data[0])-1])

if __name__ == '__main__':
    main()