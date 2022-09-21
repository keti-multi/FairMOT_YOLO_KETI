from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class MotAttLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotAttLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID

        self.nAtt1=opt.num_att1
        self.nAtt2=opt.num_att2
        self.nAtt3=opt.num_att3
        self.nAtt4=opt.num_att4
        self.nAtt5=opt.num_att5
        self.nAtt6=opt.num_att6

        self.classifier = nn.Linear(self.emb_dim, self.nID)



        if opt.id_loss == 'focal':
            torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.classifier.bias, bias_value)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)

        self.att1loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.att2loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.att3loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.att4loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.att5loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.att6loss = nn.CrossEntropyLoss(ignore_index=-1)

        # Hair_style
        self.classifier_att1 = nn.Linear(self.emb_dim, opt.num_att1)
        # Hair_color
        self.classifier_att2 = nn.Linear(self.emb_dim, opt.num_att2)
        # Top_style
        self.classifier_att3 = nn.Linear(self.emb_dim, opt.num_att3)
        # Top_Color
        self.classifier_att4 = nn.Linear(self.emb_dim, opt.num_att4)
        # Bottom_style
        self.classifier_att5 = nn.Linear(self.emb_dim, opt.num_att5)
        # Bottom_Color
        self.classifier_att6 = nn.Linear(self.emb_dim, opt.num_att6)

        print("self.nID : ",self.nID)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)

        self.emb_scale1 = math.sqrt(2) * math.log(opt.num_att1 - 1)
        self.emb_scale2 = math.sqrt(2) * math.log(opt.num_att2 - 1)
        self.emb_scale3 = math.sqrt(2) * math.log(opt.num_att3 - 1)
        self.emb_scale4 = math.sqrt(2) * math.log(opt.num_att4 - 1)
        self.emb_scale5 = math.sqrt(2) * math.log(opt.num_att5 - 1)
        self.emb_scale6 = math.sqrt(2) * math.log(opt.num_att6 - 1)





        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        # opt.heads = {'hm': opt.num_classes,
        #              'wh': 2 if not opt.ltrb else 4,
        #              'id': opt.reid_dim,
        #              'att': opt.num_att}
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss,att1_loss,att2_loss,att3_loss,att4_loss,att5_loss,att6_loss = 0, 0, 0, 0, 0,0,0,0,0,0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)


                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                if self.opt.id_loss == 'focal':
                    id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(1,
                                                                                                  id_target.long().view(
                                                                                                      -1, 1), 1)
                    id_loss += sigmoid_focal_loss_jit(id_output, id_target_one_hot,
                                                      alpha=0.25, gamma=2.0, reduction="sum"
                                                      ) / id_output.size(0)
                else:
                    id_loss += self.IDLoss(id_output, id_target)
                    # if len(id_target)>0 and len(id_output)>0:
                    #     id_loss += self.IDLoss(id_output, id_target)
                    #     print("\nid_output.shape : ", id_output.shape )
                    #     print("id_target : ", id_target.shape )
                    #     print("\n\n\nself.IDLoss(id_output, id_target) : ", self.IDLoss(id_output, id_target))
                    # else :
                    #     print ("\n######\n\nself.IDLoss(id_output, id_target) : ",self.IDLoss(id_output, id_target))
                    #     print("id_output : ", id_output.shape )
                    #     print("id_target : ", id_target.shape )
                    #     id_loss += torch.Tensor(0.0001).cuda()
            if opt.att_weight > 0:
                att1_head = _tranpose_and_gather_feat(output['att1'], batch['ind'])
                att2_head = _tranpose_and_gather_feat(output['att2'], batch['ind'])
                att3_head = _tranpose_and_gather_feat(output['att3'], batch['ind'])
                att4_head = _tranpose_and_gather_feat(output['att4'], batch['ind'])
                att5_head = _tranpose_and_gather_feat(output['att5'], batch['ind'])
                att6_head = _tranpose_and_gather_feat(output['att6'], batch['ind'])

                att1_head = att1_head[batch['reg_mask'] > 0].contiguous()
                att2_head = att2_head[batch['reg_mask'] > 0].contiguous()
                att3_head = att3_head[batch['reg_mask'] > 0].contiguous()
                att4_head = att4_head[batch['reg_mask'] > 0].contiguous()
                att5_head = att5_head[batch['reg_mask'] > 0].contiguous()
                att6_head = att6_head[batch['reg_mask'] > 0].contiguous()

                att1_head = self.emb_scale1 * F.normalize(att1_head)
                att2_head = self.emb_scale2 * F.normalize(att2_head)
                att3_head = self.emb_scale3 * F.normalize(att3_head)
                att4_head = self.emb_scale4 * F.normalize(att4_head)
                att5_head = self.emb_scale5 * F.normalize(att5_head)
                att6_head = self.emb_scale6 * F.normalize(att6_head)

                att1_target = batch['att1'][batch['reg_mask'] > 0]
                att2_target = batch['att2'][batch['reg_mask'] > 0]
                att3_target = batch['att3'][batch['reg_mask'] > 0]
                att4_target = batch['att4'][batch['reg_mask'] > 0]
                att5_target = batch['att5'][batch['reg_mask'] > 0]
                att6_target = batch['att6'][batch['reg_mask'] > 0]
                # print("att1_head shape : ", att1_head.shape)
                att1_output = self.classifier_att1(att1_head).contiguous()
                att2_output = self.classifier_att2(att2_head).contiguous()
                att3_output = self.classifier_att3(att3_head).contiguous()
                att4_output = self.classifier_att4(att4_head).contiguous()
                att5_output = self.classifier_att5(att5_head).contiguous()
                att6_output = self.classifier_att6(att6_head).contiguous()




                if self.opt.id_loss == 'focal':
                    att1_target_one_hot = att1_output.new_zeros((att1_head.size(0), self.nAtt1)).scatter_(1,
                                                                                                  att1_target.long().view(
                                                                                                      -1, 1), 1)
                    att1_loss += sigmoid_focal_loss_jit(att1_output, att1_target_one_hot,
                                                      alpha=0.25, gamma=2.0, reduction="sum"
                                                      ) / att1_output.size(0)
                else:
                    att1_loss += self.att1loss(att1_output, att1_target)
                    att2_loss += self.att2loss(att2_output, att2_target)
                    att3_loss += self.att3loss(att3_output, att3_target)
                    att4_loss += self.att4loss(att4_output, att4_target)
                    att5_loss += self.att5loss(att5_output, att5_target)
                    att6_loss += self.att6loss(att6_output, att6_target)


        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        att_loss = att1_loss + att2_loss + att3_loss + att4_loss + att5_loss + att6_loss
        if opt.multi_loss == 'uncertainty':
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss *= 0.5
        elif opt.att_weight > 0: ##ToDo att_loss hyperparameter tuning
            loss = det_loss + 0.1 * id_loss + 0.1 * att_loss
        else:
            loss = det_loss + 0.1 * id_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss, 'att1_loss': att1_loss,'att2_loss': att2_loss,'att3_loss': att3_loss,'att4_loss': att4_loss,'att5_loss': att5_loss,'att6_loss': att6_loss}
        return loss, loss_stats

class MotAttTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotAttTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss', 'att1_loss', 'att2_loss', 'att3_loss', 'att4_loss', 'att5_loss', 'att6_loss']
        loss = MotAttLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]