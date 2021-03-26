import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.bbreg as bbmodels
from ltr import model_constructor
import torch

class ATOMnet(nn.Module):
    """ ATOM network module"""
    def __init__(self, feature_extractor, bb_regressor, bb_regressor_layer, extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(ATOMnet, self).__init__()

        self.feature_extractor = feature_extractor
        self.bb_regressor = bb_regressor
        self.bb_regressor_layer = bb_regressor_layer

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        train_feat_iou = [feat for feat in train_feat.values()]
        test_feat_iou = [feat for feat in test_feat.values()]

        # Obtain iou prediction
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou,
                                     train_bb.reshape(num_train_images, num_sequences, 4),
                                     test_proposals.reshape(num_train_images, num_sequences, -1, 4))
        return iou_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)

class ATOMnet_DeT(nn.Module):
    """ ATOM network module"""
    def __init__(self, feature_extractor, feature_extractor_depth, bb_regressor, bb_regressor_layer, merge_type='mean', extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(ATOMnet_DeT, self).__init__()

        self.feature_extractor = feature_extractor
        self.feature_extractor_depth = feature_extractor_depth
        self.bb_regressor = bb_regressor
        self.bb_regressor_layer = bb_regressor_layer

        self.merge_type = merge_type
        if self.merge_type == 'conv':
            self.merge_layer2 = nn.Conv2d(256, 128, (1,1))
            self.merge_layer3 = nn.Conv2d(512, 256, (1,1))

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        train_feat_iou = [feat for feat in train_feat.values()]
        test_feat_iou = [feat for feat in test_feat.values()]

        # Obtain iou prediction
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou,
                                     train_bb.reshape(num_train_images, num_sequences, 4),
                                     test_proposals.reshape(num_train_images, num_sequences, -1, 4))
        return iou_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer

        dims = im.shape
        if dims[1] == 6:
            color_feat = self.feature_extractor(im[:, :3, :, :], layers)
            depth_feat = self.feature_extractor_depth(im[:, 3:, :, :], layers)
            return self.merge(color_feat, depth_feat)
        else:
            return self.feature_extractor(im, layers)

        # return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        dims = im.shape
        # return self.feature_extractor(im, layers)
        if dims[1] == 6:
            color_feat = self.feature_extractor(im[:, :3, :, :], layers)
            depth_feat = self.feature_extractor_depth(im[:, 3:, :, :], layers)
            return self.merge(color_feat, depth_feat)
        else:
            return self.feature_extractor(im, layers)

    def merge(self, color_feat, depth_feat):

        feat = {}

        if self.merge_type == 'conv':
            feat['layer2'] = self.merge_layer2(torch.cat((color_feat['layer2'], depth_feat['layer2']), 1))
            feat['layer3'] = self.merge_layer3(torch.cat((color_feat['layer3'], depth_feat['layer3']), 1))

        elif self.merge_type == 'max':
            # feat['layer2'] = torch.maximum(color_feat['layer2'], depth_feat['layer2'])
            # feat['layer3'] = torch.maximum(color_feat['layer3'], depth_feat['layer3'])

            # for Torch 1.4.0
            feat['layer2'] = torch.max(color_feat['layer2'], depth_feat['layer2'])
            feat['layer3'] = torch.max(color_feat['layer3'], depth_feat['layer3'])

        elif self.merge_type == 'mul':
            feat['layer2'] = torch.mul(color_feat['layer2'], depth_feat['layer2'])
            feat['layer3'] = torch.mul(color_feat['layer3'], depth_feat['layer3'])

        elif self.merge_type == 'mean':
            feat['layer2'] = 0.5 * color_feat['layer2'] + 0.5 * depth_feat['layer2']
            feat['layer3'] = 0.5 * color_feat['layer3'] + 0.5 * depth_feat['layer3']

        elif self.merge_type == 'weightedSum':
            feat['layer2'] = self.W_rgb * color_feat['layer2'] + self.W_depth * depth_feat['layer2']
            feat['layer3'] = self.W_rgb * color_feat['layer3'] + self.W_depth * depth_feat['layer3']

        return feat


@model_constructor
def atom_resnet18_DeT(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True, merge_type='mean'):
    # backbones
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)
    backbone_net_depth = backbones.resnet18(pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = ATOMnet_DeT(feature_extractor=backbone_net, feature_extractor_depth=backbone_net_depth, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                      extractor_grad=False, merge_type=merge_type)

    return net

@model_constructor
def atom_resnet18(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False)

    return net

@model_constructor
def atom_resnet50(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False)

    return net
