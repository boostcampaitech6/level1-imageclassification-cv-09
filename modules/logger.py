# Misc
class MetricAverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.acc_val = 0
    self.acc_avg = 0
    self.acc_sum = 0
    self.count = 0
    self.f1_val = 0
    self.f1_avg = 0
    self.f1_sum = 0

  def update(self, acc_val, f1_val, n=1):
    self.acc_val = acc_val
    self.acc_sum += acc_val * n
    self.count += n
    self.acc_avg = self.acc_sum / self.count
    
    self.f1_val = f1_val
    self.f1_sum += f1_val * n
    self.f1_avg = self.f1_sum / self.count
    
class loss_AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.acc_val = 0
    self.acc_avg = 0
    self.acc_sum = 0
    self.count = 0
    self.f1_val = 0
    self.f1_avg = 0
    self.f1_sum = 0

  def update(self, acc_val, f1_val, n=1):
    self.acc_val = acc_val
    self.acc_sum += acc_val * n
    self.count += n
    self.acc_avg = self.acc_sum / self.count
    
    self.f1_val = f1_val
    self.f1_sum += f1_val * n
    self.f1_avg = self.f1_sum / self.count
    
# Misc
class LossAverageMeter(object):
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