import torch
from torch.autograd import Variable
import time
import os
from tqdm import tqdm
import sys
from quadriplet_loss import batch_hard_quadriplet_loss
from utils import AverageMeter, calculate_accuracy, calculate_accuracy_top_2, calculate_accuracy_top_5


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, log_step=100, experiment=None):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    accuracies_2 = AverageMeter()
    accuracies_5 = AverageMeter()

    end_time = time.time()

    for i, (inputs, targets, scene_targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(device=opt.cuda_id, non_blocking=True)
        inputs = Variable(inputs)
        targets = Variable(targets)

        if opt.use_quadriplet:
            embs, outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_hard_loss = 0.5 * batch_hard_quadriplet_loss(targets, scene_targets, embs)
            loss += batch_hard_loss
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        acc = calculate_accuracy(outputs, targets)
        acc_2 = calculate_accuracy_top_2(outputs, targets)
        acc_5 = calculate_accuracy_top_5(outputs, targets)
        accuracies.update(acc, inputs.size(0))
        accuracies_2.update(acc_2, inputs.size(0))
        accuracies_5.update(acc_5, inputs.size(0))

        experiment.log_metric('TRAIN Acc epoch', accuracies.val.cpu())
        experiment.log_metric('TRAIN Acc_2 epoch', accuracies_2.val.cpu())
        experiment.log_metric('TRAIN Acc_5 epoch', accuracies_5.val.cpu())

        losses.update(loss.data, inputs.size(0))
        experiment.log_metric('TRAIN Loss batch', losses.val.cpu())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    if experiment:
        experiment.log_metric('TRAIN Loss epoch', losses.avg.cpu())
        experiment.log_metric('TRAIN Acc epoch', accuracies.avg.cpu())
        experiment.log_metric('TRAIN Acc_2 epoch', accuracies_2.avg.cpu())
        experiment.log_metric('TRAIN Acc_5 epoch', accuracies_5.avg.cpu())

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
