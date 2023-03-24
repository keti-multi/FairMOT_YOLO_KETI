from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset
from .dataset.jde_attribute import AttJointDataset as AttJointDataset


def get_dataset(dataset, task):
  if task == 'mot':
    return JointDataset
  if task == 'mod':
    return JointDataset
  if task == 'mot_att':
    return AttJointDataset
  else:
    return None
  
