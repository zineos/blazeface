# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.box_utils import match, log_sum_exp
from utils.box_utils import match_gious, bbox_overlaps_giou, decode


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=1.5, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = torch.ones(class_num, 1) * alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        # print(self.gamma,self.alpha)

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim= 1)
        class_mask = inputs.data.new(N, C).fill_(0)
        #class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class GiouLoss(nn.Module):
    """
        This criterion is a implemenation of Giou Loss, which is proposed in 
        Generalized Intersection over Union Loss for: A Metric and A Loss for Bounding Box Regression.
            Loss(loc_p, loc_t) = 1-GIoU
        The losses are summed across observations for each minibatch.
        Args:
            size_sum(bool): By default, the losses are summed over observations for each minibatch.
                                However, if the field size_sum is set to False, the losses are
                                instead averaged for each minibatch.
            predmodel(Corner,Center): By default, the loc_p is the Corner shape like (x1,y1,x2,y2)
            The shape is [num_prior,4],and it's (x_1,y_1,x_2,y_2)
            loc_p: the predict of loc
            loc_t: the truth of boxes, it's (x_1,y_1,x_2,y_2)
            
    """
    def __init__(self,pred_mode = 'Center',size_sum=True,variances=None):
        super(GiouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances

    def forward(self, loc_p, loc_t, prior_data):
        num = loc_p.shape[0] 
        
        if self.pred_mode == 'Center':
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        #loss = torch.tensor([1.0])
        gious =1.0 - bbox_overlaps_giou(decoded_boxes,loc_t)
        
        loss = torch.sum(gious)
     
        if self.size_sum:
            loss = loss
        else:
            loss = loss/num
        return loss


class WingLoss(nn.Module):
    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()

        # https://arxiv.org/pdf/1711.06753v4.pdf   Figure 5
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, weight, sigma=1):
        diff = weight * (x - t)
        abs_diff = diff.abs()

        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y.sum()


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = True
        self.num_classes = cfg['num_classes']
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.loc_weight = cfg['loc_weight']
        self.cls_weight = cfg['cls_weight']
        self.landm_weight = cfg['landm_weight']
        self.focalloss = FocalLoss(self.num_classes, alpha=0.25, gamma=1.5, size_average=False)
        self.gious = GiouLoss(pred_mode = 'Center', size_sum=True, variances=self.variance)
        # self.wingloss = WingLoss(w=2)


    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, landm_data = predictions
        
        priors = priors
        num = loc_data.size(0)
        # priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            match_gious(self.threshold, truths, defaults, self.variance, 
                        labels, landms, loc_t, conf_t, landm_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()


        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        loss_landm *= self.landm_weight


        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Giou)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        num_pos = pos.long().sum(1, keepdim=True)
        giou_priors = priors.data.unsqueeze(0).expand_as(loc_data)
        loss_l = self.gious(loc_p, loc_t, giou_priors[pos_idx].view(-1, 4))
        loss_l *= self.loc_weight 


        # Focal loss
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = self.focalloss(batch_conf,conf_t)
        
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos_landm.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm