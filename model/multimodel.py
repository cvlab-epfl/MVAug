import torch

from torch.nn import functional as F
from torchvision import models

from misc.log_utils import log
from misc.geometry import project_to_ground_plane_pytorch 


class MultiNet(torch.nn.Module):
    def __init__(self, hm_size, homography_input_size, homography_output_size, nb_ch_out, model_image_pred=False, nb_view=3):
        super(MultiNet, self).__init__()
        
        self.nb_view = nb_view
        frontend_feature_outsize = 512
        multiview_features = 128
        backend_feat = [512,256,128,64]
        output_layer = 64

        self.homography_input_size = homography_input_size
        self.homography_output_size = homography_output_size

        self.hm_size = hm_size

        resnet = models.resnet34(pretrained=True)
        self.frontend = torch.nn.Sequential(*list(resnet.children())[:-4])
        frontend_feature_outsize = 128
        backend_feat = [128,128,64,32]
        output_layer = 32

        self.img_classifier = torch.nn.Sequential(
                torch.nn.ReLU(), #since the last layer of pretrained model r34 is convolution
                torch.nn.Conv2d(backend_feat[0], backend_feat[1], 1), 
                torch.nn.ReLU(),                                       
                torch.nn.Conv2d(backend_feat[1], 1, 1, bias=False),
                torch.nn.Sigmoid()
              )

        self.multiview_layer_front = torch.nn.Conv2d(frontend_feature_outsize, multiview_features, kernel_size=1)

        self.multiview_layer_back = torch.nn.Sequential(*[torch.nn.Conv2d(multiview_features*self.nb_view, multiview_features*2, kernel_size=5, padding=2),#torch.nn.Dropout(p=0.5),
        torch.nn.BatchNorm2d(multiview_features*2),
        torch.nn.ReLU(inplace=True),
        MultiScaleBackend(multiview_features*2, backend_feat, out_features=output_layer) #torch.nn.Conv2d(1024, 64, kernel_size=1)#None#dla34up(64)
        ])

        self.output_layer = torch.nn.Conv2d(output_layer, nb_ch_out, kernel_size=3 ,padding=1)

        self.logit_f = torch.nn.Sigmoid()

        self.image_pred = model_image_pred



    def forward(self, x, h_groundplane, roi):
        # x = project_to_ground_plane_pytorch(x, h_groundplane, self.homography_input_size, self.homography_output_size, self.homography_output_size)

        B, V, C, H, W = x.shape
        x = x.view(B*V,C,H,W)

        x = self.frontend(x)

        if self.image_pred:
            img_pred = self.img_classifier(x).reshape(B,V,*x.shape[-2:])
        else:
            img_pred = None

        #Reshape to have batchdimension before grounplane projection
        x = x.view(B,V,x.shape[-3],x.shape[-2],x.shape[-1]) 
        #project
        x = project_to_ground_plane_pytorch(x, h_groundplane,  self.homography_input_size, self.homography_output_size, self.hm_size)
        #merge batch and view dimension for fronted of multiview aggreagation
        x = x.view(B*V,x.shape[-3],x.shape[-2],x.shape[-1]) 
        

        x = self.multiview_layer_front(x)
        x = x.view(B,V*x.shape[1],*x.shape[2:])
        x = self.multiview_layer_back(x)

        x = self.output_layer(x)
        x = self.logit_f(x)

            
        return x, img_pred



def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
            in_channels = v
    return layers


class MultiScaleBackend(torch.nn.Module):
    def __init__(self, features, backend_feat, out_features=64, nb_sizes=4):
        super(MultiScaleBackend, self).__init__()
        self.sizes = [2**i for i in list(range(8))[::-1][:nb_sizes]]
        self.backend_feat  = backend_feat #[512,256,128,64]
        
        self.scales = torch.nn.ModuleList([self._make_scale(features, self.backend_feat, size) for size in self.sizes])
        self.bottleneck = torch.nn.Conv2d(self.backend_feat[-1] * nb_sizes, out_features, kernel_size=1)
        self.relu = torch.nn.ReLU()

    def __make_weight(self,feature,scale_feature):
        weight_feature = feature - scale_feature
        return F.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, backend_feat, size):
        layers = []
        
        layers += [torch.nn.AdaptiveAvgPool2d(output_size=(size, size))]
        layers += make_layers(backend_feat, in_channels=features, batch_norm=True, dilation=True)
        #torch.nn.Conv2d(features, features, kernel_size=1, bias=False)
        return torch.nn.Sequential(*layers)
    
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.interpolate(stage(feats), size=(h, w), mode='bilinear') for stage in self.scales]
        bottle = self.bottleneck(torch.cat(multi_scales, 1))
        
        return self.relu(bottle)