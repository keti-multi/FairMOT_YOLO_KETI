from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mot import MotTrainer
from .mot_att import MotAttTrainer
from .mot import ModTrainer

train_factory = {
  'mot': MotTrainer,
  'mod': ModTrainer,
  'mot_att': MotAttTrainer
}
