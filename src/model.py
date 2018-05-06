# -*- coding: latin-1 -*-
import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
class FaceModel(nn.Module):
    def __init__(self,embedding_size,num_classes,pretrained=False):
        super(FaceModel, self).__init__()

        self.model = resnet18(pretrained)

        self.embedding_size = embedding_size

        self.model.fc = nn.Linear(512*3*3, self.embedding_size)

        self.model.classifier = nn.Linear(self.embedding_size, num_classes)


    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=10
        self.features = self.features*alpha

        #x = self.model.classifier(self.features)
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res


from torch.nn.parameter import Parameter

class FaceModelSoftmax(nn.Module):
    def __init__(self,embedding_size,num_classes,pretrained=False,checkpoint = None):
        super(FaceModelSoftmax, self).__init__()

        self.model = resnet18(pretrained)

        self.embedding_size = embedding_size

        self.model.fc = nn.Linear(512*3*3, self.embedding_size)

        self.model.classifier = nn.Linear(self.embedding_size, num_classes)
        if checkpoint is not None:
            # Check if there are the same number of classes
            if list(checkpoint['state_dict'].values())[-1].size(0) == num_classes:
                self.load_state_dict(checkpoint['state_dict'])
            else:
                own_state = self.state_dict()
                for name, param in checkpoint['state_dict'].items():
                    if "classifier" not in name:
                        if isinstance(param, Parameter):
                            # backwards compatibility for serialized parameters
                            param = param.data
                        own_state[name].copy_(param)
                        
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=10
        self.features = self.features*alpha

        #x = self.model.classifier(self.features)
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res


class FaceModelCenter(nn.Module):
    def __init__(self,embedding_size,num_classes, checkpoint=None):
        super(FaceModelCenter, self).__init__()
        self.model = resnet18()
        self.model.avgpool = None
        self.model.fc1 = nn.Linear(512*3*3, 512)
        self.model.fc2 = nn.Linear(512, embedding_size)
        self.model.classifier = nn.Linear(embedding_size, num_classes)
        self.centers = torch.zeros(num_classes, embedding_size).type(torch.FloatTensor)
        self.num_classes = num_classes

        self.apply(self.weights_init)

        if checkpoint is not None:
            # Check if there are the same number of classes
            if list(checkpoint['state_dict'].values())[-1].size(0) == num_classes:
                self.load_state_dict(checkpoint['state_dict'])
                self.centers = checkpoint['centers']
            else:
                own_state = self.state_dict()
                for name, param in checkpoint['state_dict'].items():
                    if "classifier" not in name:
                        if isinstance(param, Parameter):
                            # backwards compatibility for serialized parameters
                            param = param.data
                        own_state[name].copy_(param)

    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


    def get_center_loss(self,target, alpha):
        batch_size = target.size(0)
        features_dim = self.features.size(1)

        target_expand = target.view(batch_size,1).expand(batch_size,features_dim)

        centers_var = Variable(self.centers)
        centers_batch = centers_var.gather(0,target_expand).cuda()

        criterion = nn.MSELoss()
        center_loss = criterion(self.features,  centers_batch)

        diff = centers_batch - self.features

        unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)

        appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))

        appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)

        diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)

        diff_cpu = alpha * diff_cpu

        for i in range(batch_size):
            #Update the parameters c_j for each j by c^(t+1)_j = c^t_j − α · ∆c^t_j
            self.centers[target.data[i]] -= diff_cpu[i].type(self.centers.type())

        return center_loss, self.centers

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc1(x)
        #feature for center loss
        x = self.model.fc2(x)
        self.features = x
        self.features_norm = self.l2_norm(x)
        return self.features_norm

    def forward_classifier(self,x):
        features_norm = self.forward(x)
        x = self.model.classifier(features_norm)
        return F.log_softmax(x)
class FaceModelMargin(nn.Module):
    def __init__(self,embedding_size,num_classes,batch_k,pretrained=False):
        super(FaceModelMargin, self).__init__()

        self.model = resnet18(pretrained)

        self.embedding_size = embedding_size

        self.model.fc = nn.Linear(512*3*3, self.embedding_size)

        self.model.classifier = nn.Linear(self.embedding_size, num_classes)
        
        self.batch_k = batch_k


    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def get_distance(self,x):
        """Helper function for margin-based loss. Return a distance matrix given a matrix."""
        n = x.shape[0]
        square = np.sum(x ** 2,1,keepdims =True)
        distance_square = square + square.T - (2.0 * np.dot(x, x.T))
        # Adding identity to make sqrt work.
        return np.sqrt(distance_square + np.identity(n))

    def DistanceWeightedSampling(self,x,batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, **kwargs):
        k = batch_k
        n, d = x.data.shape
        distance = self.get_distance(x.cpu().data.numpy())
        # Cut off to avoid high variance.
        distance = np.clip(distance,cutoff,None)
        #distance[torch.lt(distance, cutoff)] = cutoff

        # Subtract max(log(distance)) for stability.
        log_weights = ((2.0 - float(d)) * np.log(distance)
                       - (float(d - 3) / 2) * np.log(1.0 - 0.25 * (distance ** 2.0)))
        weights = np.exp(log_weights - np.max(log_weights))

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = np.ones(weights.shape)
        for i in range(0, n, k):
            mask[i:i+k, i:i+k] = 0

        weights = weights * np.array(mask) * (distance < nonzero_loss_cutoff)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        a_indices = []
        p_indices = []
        n_indices = []

        for i in range(n):
            block_idx = i // k
            try:
                n_indices += np.random.choice(n, k-1, p=weights[i]).tolist()
            except:
                n_indices += np.random.choice(n, k-1).tolist()
            for j in range(block_idx * k, (block_idx + 1) * k):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)
    #     GPU
        a_indices = Variable(torch.LongTensor(a_indices).cuda())
        p_indices = Variable(torch.LongTensor(p_indices).cuda())
        n_indices = Variable(torch.LongTensor(n_indices).cuda())
    #     CPU
    #     a_indices = Variable(torch.LongTensor(a_indices))
    #     p_indices = Variable(torch.LongTensor(p_indices))
    #     n_indices = Variable(torch.LongTensor(n_indices))

        return a_indices, x.index_select(0,a_indices), x.index_select(0,p_indices), x.index_select(0,n_indices)
    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        a_indices, x_a, x_p, x_n = self.DistanceWeightedSampling(self.features,self.batch_k)
        alpha=10
        self.features = self.features*alpha       

        #x = self.model.classifier(self.features)
        return self.features,a_indices, x_a, x_p, x_n

    def forward_classifier(self, x):
        features = (self.forward(x))[0]
        res = self.model.classifier(features)
        return res
    # Distance Weighted Sampling