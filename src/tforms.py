import torchvision.transforms as TF
import torchvision.transforms.functional as F
import torch


class core50_normalize():
    def __init__(self):
        self.tf = TF.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

    def __call__(self, img):
        return self.tf(img)

class cifar_normalize():
    def __init__(self):
        self.tf = TF.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __call__(self, img):
        return self.tf(img)

class svhn_normalize():
    def __init__(self):
        self.tf = TF.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __call__(self, img):
        return self.tf(img)


class emnist_normalize():
    def __init__(self):
        self.tf = TF.Normalize((0.1722, 0.1722, 0.1722), (0.3309, 0.3309, 0.3309))

    def __call__(self, img):
        return self.tf(img)


class core50_train():
    """Pre-processing for Core50 training.
    """
    # TODO maybe the tranform topilimage is messing up the normalization!
    def __init__(self):
        self.tf = TF.Compose([TF.ToPILImage(), TF.Resize(size=(224,224)), TF.RandomHorizontalFlip(), TF.ToTensor(), core50_normalize()])
        # self.tf = TF.Compose([core50_normalize()])

    def __call__(self, img):
        return self.tf(img)


class core50_test():
    """Pre-processing for Core50 test.
    """
    def __init__(self):
        self.tf = TF.Compose([TF.ToPILImage(), TF.Resize(size=(224,224)), TF.ToTensor(), core50_normalize()])
        # self.tf = TF.Compose([core50_normalize()])
        
    def __call__(self, img):
        return self.tf(img)






class cifar_train():
    '''Pr-processing for cifar datasets both train and test'''
    def __init__(self):

        self.tf = TF.Compose(
            [TF.ToPILImage(), TF.Resize(size = 224), TF.ToTensor(), cifar_normalize()])

    def __call__(self, img):
        return self.tf(img)

class cifar_test():
    '''Pr-processing for cifar datasets both train and test'''
    def __init__(self):

        self.tf = TF.Compose(
            [TF.ToPILImage(), TF.Resize(size = 224), TF.ToTensor(),cifar_normalize()])

    def __call__(self, img):
        return self.tf(img)





class svhn_train():
    '''Pr-processing for cifar datasets both train and test'''
    def __init__(self):

        self.tf = TF.Compose(
            [TF.ToPILImage(), TF.Resize(size = 224), TF.ToTensor(), svhn_normalize()])

    def __call__(self, img):
        return self.tf(img)

class svhn_test():
    '''Pr-processing for cifar datasets both train and test'''
    def __init__(self):

        # self.tf = TF.Compose(
        #     [TF.ToPILImage(), TF.Resize(size = 224), TF.CenterCrop(224), TF.ToTensor(), svhn_normalize()])
        self.tf = TF.Compose(
            [TF.ToPILImage(), TF.Resize(size = 224), TF.ToTensor(), svhn_normalize()])

    def __call__(self, img):
        return self.tf(img)





class emnist_train():
    '''Pr-processing for cifar datasets both train and test'''
    def __init__(self):

        self.tf = TF.Compose(
            [TF.ToPILImage(), TF.Resize(size = 224), TF.ToTensor(), emnist_normalize()])

            
        # self.tf = TF.Compose(
        #     [TF.ToPILImage(), TF.Resize(size = 224), TF.ToTensor()])


    def __call__(self, img):
        return self.tf(img)

class emnist_test():
    '''Pr-processing for cifar datasets both train and test'''
    def __init__(self):

        self.tf = TF.Compose(
            [TF.ToPILImage(), TF.Resize(size = 224),  TF.ToTensor(), emnist_normalize()])

    def __call__(self, img):
        return self.tf(img)









class tf_simple():
    '''Pr-processing for cifar datasets both train and test'''
    def __init__(self):

        normalize = cifar_normalize()

        self.tf = TF.Compose(
            [TF.ToPILImage(), TF.ToTensor(), normalize])

    def __call__(self, img):
        return self.tf(img)




class inaturalist_normalize():
    def __init__(self):
        self.tf = TF.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __call__(self, img):
        return self.tf(img)







class inaturalist_train():
    """Pre-processing for inaturalist training.
    """
    def __init__(self):
        self.tf = TF.Compose([TF.Resize(256), TF.CenterCrop(224), TF.RandomHorizontalFlip(), TF.ToTensor(), 
                              inaturalist_normalize(),
                              ])
        # self.tf = TF.Compose([core50_normalize()])

    def __call__(self, img):
        return self.tf(img)


    
    
class inaturalist_test():
    """Pre-processing for inaturalist training.
    """
    def __init__(self):
        self.tf = TF.Compose([TF.Resize(256), TF.CenterCrop(224), TF.ToTensor(), inaturalist_normalize()])

    def __call__(self, img):
        return self.tf(img)
    





class inaturalist_preload():
    """Pre-processing for inaturalist training.
    """
    # TODO maybe the tranform topilimage is messing up the normalization!
    def __init__(self):
        self.tf = TF.Compose([TF.Resize(256), TF.CenterCrop(224), TF.ToTensor(), inaturalist_normalize()])
        # self.tf = TF.Compose([core50_normalize()])

    def __call__(self, img):
        return self.tf(img)
    

class inaturalist_train_afterload():
    """Pre-processing for inaturalist training.
    """
    # has tensor coming in 
    def __init__(self):
        self.tf = TF.Compose([TF.ToPILImage(), TF.RandomHorizontalFlip(), TF.ToTensor()])
        # self.tf = TF.Compose([core50_normalize()])

    def __call__(self, img):
        return self.tf(img)
    
class inaturalist_test_afterload():
    """Pre-processing for inaturalist training.
    """
    # has tensor coming in 
    def __init__(self):
        self.tf = TF.Compose([TF.ToPILImage(), TF.ToTensor()])
        # self.tf = TF.Compose([core50_normalize()])

    def __call__(self, img):
        return self.tf(img)
    






class eightdset_normalize():
    def __init__(self):
        self.tf = TF.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __call__(self, img):
        return self.tf(img)


class eightdset_train():
    """Pre-processing for inaturalist training.
    """
    # TODO maybe the tranform topilimage is messing up the normalization!
    def __init__(self):
        self.tf = TF.Compose([TF.Resize(256), TF.CenterCrop(224), TF.RandomHorizontalFlip(), TF.ToTensor(), 
                              eightdset_normalize(),
                              ])
    def __call__(self, img):
        return self.tf(img)



class eightdset_test():
    """Pre-processing for inaturalist training.
    """
    # TODO maybe the tranform topilimage is messing up the normalization!
    def __init__(self):
        self.tf = TF.Compose([TF.Resize(256), TF.CenterCrop(224), TF.ToTensor(), 
                              eightdset_normalize(),
                              ])
    def __call__(self, img):
        return self.tf(img)





class eightdset_train_svhn():
    """Pre-processing for inaturalist training.
    """
    # TODO maybe the tranform topilimage is messing up the normalization!
    def __init__(self):
        self.tf = TF.Compose([TF.ToPILImage(), TF.Resize(224), TF.CenterCrop(224), TF.RandomHorizontalFlip(), TF.ToTensor(), 
                              eightdset_normalize(),
                              ])
    def __call__(self, img):
        return self.tf(img)



class eightdset_test_svhn():
    """Pre-processing for inaturalist training.
    """
    # TODO maybe the tranform topilimage is messing up the normalization!
    def __init__(self):
        self.tf = TF.Compose([TF.ToPILImage(), TF.Resize(224), TF.CenterCrop(224), TF.ToTensor(), 
                              eightdset_normalize(),
                              ])
    def __call__(self, img):
        return self.tf(img)




# class tf_simple():
#     '''Pr-processing for cifar datasets both train and test'''
#     def __init__(self):
#         self.tf = TF.ToTensor()

#     def __call__(self, img):
#         return self.tf(img)



class tf_additional():
    def __init__(self, dataset):
        if dataset=='cifar10' or dataset=='cifar100':
            normalize = cifar_normalize()
        elif dataset=='core50':
            normalize = core50_normalize()

        self.tf = TF.Compose(
            [TF.ToPILImage(), TF.Resize(size = 224), TF.CenterCrop(224), TF.ToTensor(), normalize])

    def __call__(self, img):
        return self.tf(img)



class TFBatch():
    def __init__(self, transform):
        self.tf = transform
    def apply(self, batch):
        batch_tf = torch.stack([self.tf(batch[i].squeeze()) for i in range(batch.shape[0])])
        return batch_tf