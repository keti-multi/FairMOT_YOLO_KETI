from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mot import MotTrainer
from .mot_att import MotAttTrainer

train_factory = {
  'mot': MotTrainer,
  'mot_att': MotAttTrainer
}
