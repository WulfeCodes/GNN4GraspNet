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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d as o3d
#TODO verify augmented data
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
    
    def visualize(self,preds):

        positions = self.data.pos.cpu().numpy()   # shape: [N, 3]
        preds = preds.detach().cpu().numpy()               # shape: [N] or [N, 1]

        # Normalize predictions for color mapping
        norm_preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
        colors = cm.jet(norm_preds)[:, :3]  # get RGB from colormap

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])
        #pick whether meshing, texturing, meshing, smoothing?

    def augment(self,points,offsets):
        points,offsets=self.random_rotation(points,offsets)
        points,offsets = self.random_translation_torch(points,offsets)
        points,offsets = self.jitter_torch(points,offsets)
        return points, offsets
    def random_translation_torch(self,points, offsets, max_shift=0.05):

        shift = (torch.rand((1, 3), device=points.device) - 0.5) * 2 * max_shift
        translated_points = points + shift
        translated_offsets = offsets + shift
        return translated_points,translated_offsets

    def jitter_torch(self,points, offsets, sigma=0.005, clip=0.02):

        noise = torch.randn_like(points) * sigma
        noise = torch.clamp(noise, -clip, clip)
        offsets_noise = offsets + noise
        points_noise = points + noise
        return points_noise, offsets_noise

    def random_rotation(self,points, offsets,angle_range=15):
        if isinstance(angle_range, torch.Tensor):
            if angle_range.numel() == 1:
                angle_range = angle_range.item()  # Convert single-element tensor to scalar
            else:
                raise ValueError("angle_range must be a scalar or single-element tensor")
        
        angle_range = float(angle_range)
        angle = torch.deg2rad(torch.empty(1).uniform_(-angle_range, angle_range))
        axis = torch.randint(0, 3, (1,)).item()

        c, s = torch.cos(angle), torch.sin(angle)
        if axis == 0:  # x-axis
            R = torch.tensor([[1, 0, 0],
                            [0, c, -s],
                            [0, s,  c]], dtype=torch.float32)
        elif axis == 1:  # y-axis
            R = torch.tensor([[ c, 0, s],
                            [ 0, 1, 0],
                            [-s, 0, c]], dtype=torch.float32)
        else:  # z-axis
            R = torch.tensor([[c, -s, 0],
                            [s,  c, 0],
                            [0,  0, 1]], dtype=torch.float32)
        points_augmented = points @ R.T
        offsets_augmented = offsets @ R.T
        return points_augmented, offsets_augmented

    def forward(self):
          # x: (N, in_channels)
        x,edge_index = self.data.x, self.data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        out = self.lin(x)  # (N, 1)

        return out.squeeze(-1)  # (N,)
    def train(self):
        total_loss = 0
        out=self.forward()
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
        #self.eval()
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
        shape = (len(DataShape),k)
        samplers = torch.empty(shape,dtype=torch.int32)

        for i, currShape in enumerate(DataShape):
            samplers[i]=torch.tensor(np.random.randint(0,currShape,size=k),dtype=torch.int32)

        sampledIndexes=samplers.T

        positions=torch.empty((k,3),dtype=torch.float32)
        scores = torch.empty((k,1),dtype=torch.float32)
        offsets = torch.empty((k,3),dtype=torch.float32)

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
        return positions,scores,offsets

    def train4Epochs(self,epochs,k,grasp_label_path="./grasp_label"):
        #train w original label data
        for z in range(epochs):
            for i, npz_str in enumerate(os.listdir(grasp_label_path)):
                label_data = load(os.path.join(grasp_label_path,npz_str))

                positions,scores,offsets=self.sampleIndexes(label_data,label_data['collision'].shape,k)

                self.makeGraph(positions,offsets,scores)
                self.train()
            self.save_weights()

        #train w augmented label data
            for i, npz_str in enumerate(os.listdir(grasp_label_path)):
                label_data = load(os.path.join(grasp_label_path,npz_str))

                positions,scores,offsets=self.sampleIndexes(label_data,label_data['collision'].shape,k)

                positions,offsets=self.augment(positions,offsets)
                self.makeGraph(positions,offsets,scores)
                self.train()
            self.save_weights()
        
    def inferenceWithVisualization(self,path):
        label_data = load(path)
        positions,scores,offsets=self.sampleIndexes(label_data,label_data['collision'].shape,5000)
        self.makeGraph(positions,offsets,scores)    
        pos=self.forward()
        self.visualize(pos)


def main():
    #in channels is fixed, dont change pls :)
    GraphNet = EdgeConvNet(in_channels=7)
    GraphNet.load_weights(path="./edgeconv_weights.pth")
    #epochs: number of training cycles, 
    # k: number of points sampled
    GraphNet.train4Epochs(epochs=2,k=3500)

    GraphNet.inferenceWithVisualization(path="C:/Project/tmp/grasp_label/008_labels.npz")


if __name__ == '__main__':
    main()