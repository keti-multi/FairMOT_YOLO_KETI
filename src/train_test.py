from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel


# for test
from datasets.dataset.jde import DetDataset, collate_fn
from models.decode import mot_decode
from utils.post_process import ctdet_post_process
from utils.utils import xywh2xyxy, ap_per_class, bbox_iou
import numpy as np
import time



from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def post_process(opt, dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], opt.num_classes)
    for j in range(1, opt.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]


def merge_outputs(opt, detections):
    results = {}
    for j in range(1, opt.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, opt.num_classes + 1)])
    if len(scores) > 128:
        kth = len(scores) - 128
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def test_det(
        opt=None,
        batch_size=12,
        img_size=(1088, 608),
        iou_thres=0.5,
        print_interval=40,
        model=None,
        dataloader=None
):
    #data_cfg = opt.data_cfg
    #f = open(data_cfg)
    #data_cfg_dict = json.load(f)
    #f.close()
    nC = 1
    #test_path = data_cfg_dict['test']
    #dataset_root = data_cfg_dict['root']
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    #model = create_model(opt.arch, opt.heads, opt.head_conv)
    #model = load_model(model, opt.load_model)
    #model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    model.eval()

    # Get dataloader
    #transforms = T.Compose([T.ToTensor()])
    #dataset = DetDataset(dataset_root, test_path, img_size, augment=False, transforms=transforms)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
    #                                         num_workers=8, drop_last=False, collate_fn=collate_fn)
    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    for batch_i, (imgs, targets, paths, shapes, targets_len) in enumerate(dataloader):
        t = time.time()
        #seen += batch_size

        output = model(imgs.cuda())[-1]
        origin_shape = shapes[0]
        width = origin_shape[1]
        height = origin_shape[0]
        inp_height = img_size[1]
        inp_width = img_size[0]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // opt.down_ratio,
                'out_width': inp_width // opt.down_ratio}
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg'] if opt.reg_offset else None
        opt.K = 200
        detections, inds = mot_decode(hm, wh, reg=reg, ltrb=opt.ltrb, K=opt.K)
        # Compute average precision for each sample
        targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
        for si, labels in enumerate(targets):
            seen += 1
            #path = paths[si]
            #img0 = cv2.imread(path)
            dets = detections[si]
            dets = dets.unsqueeze(0)
            dets = post_process(opt, dets, meta)
            dets = merge_outputs(opt, [dets])[1]

            #remain_inds = dets[:, 4] > opt.det_thres
            #dets = dets[remain_inds]
            if dets is None:
                # If there are labels but no detections mark as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # If no labels add number of detections as incorrect
            correct = []
            if labels.size(0) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mAPs.append(0), mR.append(0), mP.append(0)
                continue
            else:
                target_cls = labels[:, 0]

                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = xywh2xyxy(labels[:, 2:6])
                target_boxes[:, 0] *= width
                target_boxes[:, 2] *= width
                target_boxes[:, 1] *= height
                target_boxes[:, 3] *= height

                '''
                path = paths[si]
                img0 = cv2.imread(path)
                img1 = cv2.imread(path)
                for t in range(len(target_boxes)):
                    x1 = target_boxes[t, 0]
                    y1 = target_boxes[t, 1]
                    x2 = target_boxes[t, 2]
                    y2 = target_boxes[t, 3]
                    cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.imwrite('gt.jpg', img0)
                for t in range(len(dets)):
                    x1 = dets[t, 0]
                    y1 = dets[t, 1]
                    x2 = dets[t, 2]
                    y2 = dets[t, 3]
                    cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.imwrite('pred.jpg', img1)
                abc = ace
                '''

                detected = []
                for *pred_bbox, conf in dets:
                    obj_pred = 0
                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)

            # Compute Average Precision (AP) per class
            AP, AP_class, R, P = ap_per_class(tp=correct,
                                              conf=dets[:, 4],
                                              pred_cls=np.zeros_like(dets[:, 4]),  # detections[:, 6]
                                              target_cls=target_cls)

            # Accumulate AP per class
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            # Compute mean AP across all classes in this image, and append to image list
            mAPs.append(AP.mean())
            mR.append(R.mean())
            mP.append(P.mean())

            # Means of all images
            mean_mAP = np.sum(mAPs) / (AP_accum_count + 1E-16)
            mean_R = np.sum(mR) / (AP_accum_count + 1E-16)
            mean_P = np.sum(mP) / (AP_accum_count + 1E-16)

        if batch_i % print_interval == 0:
            # Print image mAP and running mean mAP
            print(('%11s%11s' + '%11.3g' * 4 + 's') %
                  (seen, dataloader.dataset.nF, mean_P, mean_R, mean_mAP, time.time() - t))
    # Print mAP per class
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))

    print('AP: %-.4f\n\n' % (AP_accum[0] / (AP_accum_count[0] + 1E-16)))

    # Return mAP
    return mean_mAP, mean_R, mean_P

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    trainset_root = data_config['root']
    dataset_root = data_config['data']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, trainset_root, (opt.image_width, opt.image_height), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    ## 220723_testloader
    data_cfg_test = opt.data_cfg
    f_test = open(data_cfg_test)
    data_cfg_dict_test = json.load(f_test)
    f_test.close()
    img_size = (opt.image_width,opt.image_height)
    test_path = data_cfg_dict_test['test']
    dataset_root = data_cfg_dict_test['root']
    test_transforms = T.Compose([T.ToTensor()])
    dataset = DetDataset(dataset_root, test_path, img_size, augment=False, transforms=test_transforms)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False,
                                             num_workers=8, drop_last=False, collate_fn=collate_fn)

    # test_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)




    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)
    best_map = 0
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)

        ## test code

        map,_,_=test_det(opt=opt,batch_size=4,model = model,img_size=img_size, dataloader=test_loader)
        #print(map)
        #raise(KeyboardInterrupt)
        logger.write('epoch: {} |'.format(epoch))
        logger.scalar_summary('test_mAP', map[0], epoch)
        logger.write('mAP {:8f} | '.format(map[0]))
        if map>best_map:
            print("NEW BEST mAP : ",map)
            save_model(os.path.join(opt.save_dir, 'model_best_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            best_map = map

        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr                                
        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
