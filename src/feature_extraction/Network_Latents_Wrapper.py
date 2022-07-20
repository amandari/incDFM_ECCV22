import torch.nn.functional as F
import torch.nn as nn
import re


class NetworkLatents():
    def __init__(self, model: nn.Module, layer_names, pool_factors=None):
        self.layer_names = layer_names
        self.pool_factors = pool_factors
        if isinstance(model, nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.activations = dict()
        if pool_factors is None:
            pool_factors = {layer_name: 1 for layer_name in layer_names}

        d = dict(self.model.named_modules())
        print('Will fetch activations from:')
        for layer_name in layer_names:
            if layer_name in d:
                layer = self.getLayer(layer_name)
                pool_factor = pool_factors[layer_name]
                # layer_rep = re.match('.+($|\n)', layer.__repr__())
                print('{}, average pooled by {}'.format(layer_name, pool_factor))
                layer.register_forward_hook(self.getActivation(layer_name, pool_factor))
            else:
                print("Warning: Layer {} not found".format(layer_name))

    def __repr__(self):
        out = 'Layers {}\n'.format(self.layer_names)
        if self.pool_factors:
            out = '{}Pool factors {}\n'.format(out, list(self.pool_factors.values()))
        out = '{}'.format(self.model.__repr__())
        return out


    def getActivation(self, name, pool):
        def hook(_, __, output):
            layer_out = output.detach()

            if layer_out.dim() == 4 and pool > 1:
                layer_out_pool = F.avg_pool2d(layer_out, pool)
            elif layer_out.dim() == 4 and pool == -1:
                layer_out_pool = F.avg_pool2d(layer_out, layer_out.size()[-1])
            else:
                layer_out_pool = layer_out
            # print(layer_out_pool.shape)
            if len(layer_out_pool.shape)>2:
                self.activations[name] = layer_out_pool.view(output.size(0), -1)
            else:
                self.activations[name] = layer_out_pool
        return hook

    def __call__(self, data, task_num=None, base_apply=True):
        # self.activations.clear()
        if task_num is not None:
            out = self.model(data, task_num, base_apply=base_apply)
        else:
            out = self.model(data, base_apply=base_apply)
        return out, self.activations

    def getLayer(self, layer_name):
        m = self.model
        sep = '.'
        attrs = layer_name.split(sep)
        for a in attrs:
            try:
                i = int(a)
                m = m[i]
            except ValueError:
                m = m.__getattr__(a)
        return m

