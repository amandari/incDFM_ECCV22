from torch.utils.data import Dataset
import numpy as np

def combine_for_replay_all(list_paths):
    combined_paths = []
    num_tasks = len(list_paths)
    for t in range(num_tasks):
        combined_paths.append([])
        for i, l in enumerate(list_paths):
            if i<=t:
                combined_paths[-1].append(l)
            else:
                break
    return combined_paths
        

class DSET_wrapper_Replay(Dataset):
    """
    Wrapper for coreset replay data
    """
    def __init__(self, features, labels, latents={}, transform=None, target_transform=None):
        
        self.x = features
        self.y = labels
        self.latents=latents

        self.transform = transform 
        self.target_transform = target_transform

        self.indices_task = np.arange(self.y.shape[0])

    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
           
        feat, target = self.x[index,...], self.y[index]

        if self.transform is not None: ##input the desired tranform 

            feat = self.transform(feat)

        if self.target_transform is not None:
            
            target = self.target_transform(target)

        if len(self.latents)>0:
            activities={}
            for j, (key, val) in enumerate(self.latents.items()):
                activities[key] = val[index,...]
            return feat, target, activities
            
        return feat, target


# class DSET_wrapper_Replay(Dataset):
#     """
#     Wrapper for coreset replay data
#     """
#     def __init__(self, features, labels, latents={}, transform=None, target_transform=None):
        
#         self.features = features
#         self.labels = labels
#         self.latents=latents

#         self.transform = transform 
#         self.target_transform = target_transform

#     def __len__(self):
#         return self.labels.shape[0]
    
#     def __getitem__(self, index):
           
#         feat, target = self.features[index,...], self.labels[index]

#         if self.transform is not None: ##input the desired tranform 

#             feat = self.transform(feat)

#         if self.target_transform is not None:
            
#             target = self.target_transform(target)

#         if len(self.latents)>0:
#             activities={}
#             for j, (key, val) in enumerate(self.latents.items()):
#                 activities[key] = val[index,...]
#             return feat, target, activities
            
#         return feat, target


class DSET_wrapper_images(Dataset):
    """
    Make subset of data a dataset object
    images, labels: tensor
    """
    def __init__(self, images, labels, labels_homog, transform=None, target_transform=None):
        
        self.images = images
        self.labels = labels
        self.labels_homog = labels_homog

        self.transform = transform 
        self.target_transform = target_transform

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
           
        im, target, homog = self.images[index,...], self.labels[index], self.labels_homog[index]

        if self.transform is not None: ##input the desired tranform 

            im = self.transform(im)

        if self.target_transform is not None:
            
            target = self.target_transform(target)
        
        return im, homog, target
