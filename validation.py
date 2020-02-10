import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy, calculate_accuracy_top_2


def val_epoch(epoch, data_loader, model, criterion, opt, logger, experiment=None):
    print('validation at epoch {}'.format(epoch))

    model.eval()
    print('starting eval')
    with torch.no_grad():
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        accuracies_2 = AverageMeter()

        end_time = time.time()
        print(len(data_loader))
        for i, (inputs, targets, scene_targets) in enumerate(data_loader):
            print(i)
            data_time.update(time.time() - end_time)

            if not opt.no_cuda:
                targets = targets.cuda(device=opt.cuda_id, non_blocking=True)

            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)
            acc_2 = calculate_accuracy_top_2(outputs, targets)

            losses.update(loss.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            accuracies_2.update(acc_2, inputs.size(0))

            if experiment:
                experiment.log_metric('VAL Loss batch', losses.val.cpu())
                experiment.log_metric('VAL Acc batch', accuracies.val.cpu())
                experiment.log_metric('VAL Acc 2 batch', accuracies_2.val.cpu())

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f}) \t'
                  'Acc 2 {acc_2.val:.3f} ({acc_2.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies,
                      acc_2=accuracies_2))

        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        if experiment:
            experiment.log_metric('VAL Loss epoch', losses.avg.cpu())
            experiment.log_metric('VAL Acc epoch', accuracies.avg.cpu())
            experiment.log_metric('VAL Acc_2 epoch', accuracies_2.avg.cpu())

    return losses.avg
