import csv
import numpy as np
import torch
import time
import cv2
import math

class Timer(object):
	"""
	docstring for Timer
	"""
	def __init__(self):
		super(Timer, self).__init__()
		self.total_time = 0.0
		self.calls = 0
		self.start_time = 0.0
		self.diff = 0.0
		self.average_time = 0.0

	def tic(self):
		self.start_time = time.time()

	def toc(self, average = False):
		self.diff = time.time() - self.start_time
		self.calls += 1
		self.total_time += self.diff
		self.average_time = self.total_time / self.calls
		if average:
			return self.average_time
		else:
			return self.diff

	def format(self, time):
		m,s = divmod(time, 60)
		h,m = divmod(m, 60)
		d,h = divmod(h, 24)
		return ("{}d:{}h:{}m:{}s".format(int(d), int(h), int(m), int(s)))

	def end_time(self, extra_time):
		"""
		calculate the end time for training, show local time
		"""
		localtime= time.asctime(time.localtime(time.time() + extra_time))
		return localtime


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size


class MixUp(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def mixup_data(self, x, y, use_cuda=True):
        """
        return mixed inputs. pairs of targets 
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        # print(lam)
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class TemporalMixup(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def mixup_data(self, x):
        """
        return mixed inputs. pairs of targets
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        b, c, t, h, w = x.size()
        from numpy import random
        # skip = random.randint()
        skip = 16
        mixed_x = x
        for i in range(b):
            for j in range(t):
                mixed_x[i, :, j, :, :] = lam * x[i, :, j, :, :] + (1 - lam) * x[i, :, (j+skip) % t, :, :]
        return mixed_x

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class SpatialMixup(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def mixup_data(self, x, y=None):
        """
        return mixed inputs. pairs of targets
        """
        # ================version 1: random select sample and fusion with stable frame (all video)===================
        b, c, t, h, w = x.size()
        from numpy import random
        loss_prob = random.random() * 0.3
        mixed_x = x
        for i in range(b):
            tmp = random.randint(b)
            img_index = random.randint(t)
            for j in range(t):
                mixed_x[i, :, j, :, :] = (1-loss_prob) * x[i, :, j, :, :] + loss_prob * x[tmp, :, img_index, :, :]
                # cv2.imshow("", mixed_x[i,:,j,:,:])
        return mixed_x
        # # ================version 2: random select one same video sample and fusion with stable frame=================
        # b, c, t, h, w = x.size()
        # from numpy import random
        # loss_prob = random.random() * 0.3
        # mixed_x = x
        # for i in range(b):
        #     img_index = random.randint(t)
        #     for j in range(t):
        #         mixed_x[i, :, j, :, :] = (1-loss_prob) * x[i, :, j, :, :] + loss_prob * x[i, :, img_index, :, :]
        # return mixed_x
        # # version 3: x and y all change
        # b, c, t, h, w = x.size()
        # from numpy import random
        # loss_prob = random.random() * 0.3
        # gama = 3  # control the importance of spatial information
        # mixed_x = x
        # index = torch.randperm(b)
        # for i in range(b):
        #     img_index = random.randint(t)
        #     for j in range(t):
        #         mixed_x[i, :, j, :, :] = (1-loss_prob) * x[i, :, j, :, :] + loss_prob * x[index[i], :, img_index, :, :]
        # return mixed_x, y, y[index], loss_prob/gama

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return (1-lam) * criterion(pred, y_a) + lam * criterion(pred, y_b)


class Cut(object):
    def __init__(self, beta, cut_prob):
        self.beta = beta
        self.cut_prob = cut_prob

    def rand_bbox(self, size, lam):
        T = size[2]
        W = size[3]
        H = size[4]
        cut_rat = np.sqrt(1. - lam)
        cut_t = np.int(T * cut_rat)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        ct = np.random.randint(T)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbt1 = np.clip(ct - cut_t // 2, 0, T)
        bbt2 = np.clip(ct + cut_t // 2, 0, T)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbt1, bbt2, bbx1, bby1, bbx2, bby2

    def mixup_data(self, input):
        lam = np.random.beta(self.beta, self.beta)
        if lam > (1 - self.cut_prob):
            bbt1, bbt2, bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
            input[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = 0
        return input


class PartMix(object):
    """
    crop part of a and part of b, contact them
    """
    def __init__(self, beta, cutmix_prob):
        self.beta = beta
        self.cutmix_prob = cutmix_prob

    def rand_bbox(self, size, lam):
        T = size[2]
        W = size[3]
        H = size[4]

        bbt1 = 0
        bbt2 = T
        bbx1 = 0
        bbx2 = int(lam*W)
        bby1 = 0
        bby2 = H

        return bbt1, bbt2, bbx1, bby1, bbx2, bby2

    def mixup_data(self, input, target):
        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbt1, bbt2, bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
        input[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbt2 - bbt1) / (input.size()[-1] * input.size()[-2] * input.size()[-3]))
        return input, target_a, target_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


class CutMix(object):
    def __init__(self, beta, cutmix_prob):
        self.beta = beta
        self.cutmix_prob = cutmix_prob

    def rand_bbox(self, size, lam):
        T = size[2]
        W = size[3]
        H = size[4]
        cut_rat = np.sqrt(1. - lam)
        cut_t = np.int(T * cut_rat)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        ct = np.random.randint(T)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbt1 = np.clip(ct - cut_t // 2, 0, T)
        bbt2 = np.clip(ct + cut_t // 2, 0, T)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbt1, bbt2, bbx1, bby1, bbx2, bby2

    def mixup_data(self, input, target):
        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbt1, bbt2, bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
        input[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbt2 - bbt1) / (input.size()[-1] * input.size()[-2] * input.size()[-3]))
        return input, target_a, target_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


class CombineMix(object):
    """
    crop part of a and part of b, contact them
    """
    def __init__(self, beta):
        self.beta = beta

    def rand_bbox(self, size, lam):
        T = size[2]
        bbt1 = 0
        bbt2 = math.ceil(lam*T)

        return bbt1, bbt2

    def mixup_data(self, input, target):
        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbt1, bbt2 = self.rand_bbox(input.size(), lam)
        input[:, :, bbt1:bbt2, :, :] = input[rand_index, :, bbt1:bbt2, :, :]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - (bbt2 - bbt1)/input.size()[-3]
        return input, target_a, target_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


class GridMix(object):
    def __init__(self, beta):
        self.beta = beta

    def rand_bbox(self, size, lam):
        W = size[3]
        H = size[4]
        # cut_rat = np.sqrt(1. - lam)
        cut_rat = lam
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        bbx1 = cut_w
        bby1 = cut_h
        bbx2 = W
        bby2 = H
        return bbx1, bby1, bbx2, bby2

    def imgs_resize(self, input, f_size):
        """
        resize input (spatial) into fixed size
        :param input:
        :param size:
        :return:
        """
        b, c, t, h, w = input.size()
        resize_imgs = torch.nn.functional.interpolate(input, size=(t, f_size[0], f_size[1]), scale_factor=None,
                                                      mode='trilinear', align_corners=True)
        return resize_imgs

    def mixup_data(self, input, target):
        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
        output = torch.zeros_like(input)
        if bbx1 > 1 and bby1 > 1:
            output[:, :, :, :bbx1, :bby1] = self.imgs_resize(input[:, :, :, :, :], (bbx1, bby1))
            output[:, :, :, bbx1:bbx2, bby1:bby2] = self.imgs_resize(input[rand_index, :, :, bbx1:bbx2, bby1:bby2],
                                                                    (bbx2 - bbx1, bby2 - bby1))
        else:
            output = input
        return output, target_a, target_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


class VideoShake(object):
    def __init__(self, beta=1.0, range=3):
        self.beta = beta
        self.range = range

    def image_shake(self, img):
        lam = np.random.beta(self.beta, self.beta)
        motion = int(lam*self.range)
        import random
        direct = random.random() * 4
        c, h, w = img.size()
        new_img = img
        if motion > 0:
            if direct < 1: # up
                new_img[:, :h-motion, :] = img[:, motion:, :]
                new_img[:, 0:motion, :] = img[:, h-motion:h, :]
            elif direct < 2: # down
                new_img[:, motion:, :] = img[:, :h-motion, :]
                new_img[:, h-motion:h, :] = img[:, 0:motion, :]
            elif direct < 3: # left
                new_img[:, :, :w-motion] = img[:, :, motion:]
                new_img[:, :, 0:motion] = img[:, :, w-motion:w]
            else: # right
                new_img[:, :, motion:] = img[:, :, :w-motion]
                new_img[:, :, w-motion:w] = img[:, :, 0:motion]

        return new_img

    def mixup_data(self, input):
        b, c, t, h, w = input.size()
        mixed_x = input
        for i in range(b):
            for j in range(t):
                mixed_x[i, :, j, :, :] = self.image_shake(input[i, :, j, :, :])
        return input


class TrainingHelper(object):
    def __init__(self, image):
        self.image = image

    @staticmethod
    def congratulation(self):
        """
        if finish training success, print congratulation information
        """
        for i in range(40):
            print('*')*i
            print('finish training')


def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))