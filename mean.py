import torch
def get_mean(norm_value=255, dataset='activitynet'):
    assert dataset in ['activitynet', 'kinetics', 'gta']

    if dataset == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]
    elif dataset == 'gta':
        return [
            75.582 / norm_value,  72.9776 / norm_value,
            72.3873 / norm_value
        ]
    # elif dataset == 'gta':
    #     mean, std = online_mean_and_sd(loa)



def get_std(norm_value=255, dataset='kinetics'):
    # Kinetics (10 videos for each class)
    if dataset == 'kinetics':
        return [
            38.7568578 / norm_value, 37.88248729 / norm_value,
            40.02898126 / norm_value
        ]
    elif dataset == 'gta':
        return [
            54.3873 / norm_value, 51.6349 / norm_value,
            48.5021 / norm_value
        ]



def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:

        b, c, d, h, w = images.shape
        # exit()
        nb_pixels = b * h * w * d
        sum_ = torch.sum(images, dim=[0, 2, 3, 4])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3, 4])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)