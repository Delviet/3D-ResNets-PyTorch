import torch
from torch.autograd import Variable
import time
import os
from tqdm import tqdm
import sys
from quadriplet_loss import batch_hard_quadriplet_loss
from utils import AverageMeter, calculate_accuracy, calculate_accuracy_top_2


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, experiment=None):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    accuracies_2 = AverageMeter()

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

        losses.update(loss.data, inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        accuracies_2.update(acc_2, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'acc_2': accuracies_2.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        if experiment:
            experiment.log_metric('TRAIN Loss batch', losses.val.cpu())
            experiment.log_metric('TRAIN Acc batch', accuracies.val.cpu())
            experiment.log_metric('TRAIN Acc 2 batch', accuracies_2.val.cpu())

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f}) \t '
              'Acc 2 {acc_2.val:.3f} ({acc_2.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies,
                  acc_2=accuracies_2))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    if experiment:
        experiment.log_metric('TRAIN Loss epoch', losses.avg.cpu())
        experiment.log_metric('TRAIN Acc epoch', accuracies.avg.cpu())
        experiment.log_metric('TRAIN Acc_2 epoch', accuracies_2.avg.cpu())
        experiment.log_metric('TRAIN LR', optimizer.param_groups[0]['lr'])

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
