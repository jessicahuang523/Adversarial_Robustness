# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluates a PyTorch checkpoint on CIFAR-10/100 or MNIST."""

from absl import app
from absl import flags
import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import tqdm

import model_zoo

from PIL import Image
import subprocess
import os

_CKPT = flags.DEFINE_string(
    'ckpt', None, 'Path to checkpoint.')
_DATASET = flags.DEFINE_enum(
    'dataset', 'cifar10', ['cifar10', 'cifar100', 'mnist'],
    'Dataset on which the checkpoint is evaluated.')
_WIDTH = flags.DEFINE_integer(
    'width', 16, 'Width of WideResNet (if set to zero uses a PreActResNet).')
_DEPTH = flags.DEFINE_integer(
    'depth', 70, 'Depth of WideResNet or PreActResNet.')
_USE_CUDA = flags.DEFINE_boolean(
    'use_cuda', True, 'Whether to use CUDA.')
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 100, 'Batch size.')
_NUM_BATCHES = flags.DEFINE_integer(
    'num_batches', 0,
    'Number of batches to evaluate (zero means the whole dataset).')

class CustomDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, filename) 
                    for filename in os.listdir(root_dir) 
                    if os.path.isfile(os.path.join(root_dir, filename))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

def main(unused_argv):
  print(f'Loading "{_CKPT.value}"')

  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  # Create model and dataset.
  if _WIDTH.value == 0:
    print(f'Using a PreActResNet with depth {_DEPTH.value}.')
    model_ctor = model_zoo.PreActResNet
  else:
    print(f'Using a WideResNet with depth {_DEPTH.value} and width '
          f'{_WIDTH.value}.')
    model_ctor = model_zoo.WideResNet
  if _DATASET.value == 'mnist':
    model = model_ctor(
        num_classes=10, depth=_DEPTH.value, width=_WIDTH.value,
        activation_fn=model_zoo.Swish, mean=.5, std=.5, padding=2,
        num_input_channels=1)
    dataset_fn = datasets.MNIST
  elif _DATASET.value == 'cifar10':
    model = model_ctor(
        num_classes=10, depth=_DEPTH.value, width=_WIDTH.value,
        activation_fn=model_zoo.Swish, mean=model_zoo.CIFAR10_MEAN,
        std=model_zoo.CIFAR10_STD)
    dataset_fn = datasets.CIFAR10
  else:
    assert _DATASET.value == 'cifar100'
    model = model_ctor(
        num_classes=100, depth=_DEPTH.value, width=_WIDTH.value,
        activation_fn=model_zoo.Swish, mean=model_zoo.CIFAR100_MEAN,
        std=model_zoo.CIFAR100_STD)
    dataset_fn = datasets.CIFAR100

  # Load model.
  if _CKPT.value != 'dummy':
    params = torch.load(_CKPT.value)
    model.load_state_dict(params)
  if _USE_CUDA.value:
    model.cuda()
  model.to(device)
  model.eval()
  print('Successfully loaded.')

  # Load dataset.
  # transform_chain = transforms.Compose([transforms.Lambda(lambda x: run_runner_script(x))])
  # ds = dataset_fn(root='/tmp/data', train=False, transform=transform_chain,
  #                 download=True)
  # test_loader = data.DataLoader(ds, batch_size=_BATCH_SIZE.value, shuffle=False,
  #                               num_workers=0)
  transform_chain = transforms.Compose([transforms.ToTensor()])
  custom_images_dir = '/home/cwtang/deepmind-research/adversarial_robustness/pytorch/cifar10_after_runner'
  custom_dataset = CustomDataset(root_dir=custom_images_dir, transform=transform_chain)
  test_loader = data.DataLoader(custom_dataset, batch_size=_BATCH_SIZE.value, shuffle=False, num_workers=0)

  # Evaluation.
  correct = 0
  total = 0
  batch_count = 0
  total_batches = min((10_000 - 1) // _BATCH_SIZE.value + 1, _NUM_BATCHES.value)
  with torch.no_grad():
    for images, labels in tqdm.tqdm(test_loader, total=total_batches):
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      batch_count += 1
      if _NUM_BATCHES.value > 0 and batch_count >= _NUM_BATCHES.value:
        break
  print(f'Accuracy on the {total} test images: {100 * correct / total:.2f}%')

count = 1

def run_runner_script(image):
    global count
    base_path = "/home/cwtang/deepmind-research/adversarial_robustness/pytorch"
    tmp_cifar10_path = os.path.join(base_path, "tmp_cifar10")
    cifar10_after_runner_path = os.path.join(base_path, "cifar10_after_runner")
    
    image = image.resize((128, 128))
    image.save(os.path.join(tmp_cifar10_path, "image{}.jpg".format(count)))
    subprocess.run(['python', 'runner.py', str(count)], cwd='../../../Attack-LDM')
    
    # Load and return the processed image from runner.py
    processed_image = Image.open(os.path.join(cifar10_after_runner_path, "image{}.jpg".format(count)))
    processed_image = processed_image.resize((32, 32))
    print("|- Loaded runner result")
    count += 1
    return transforms.ToTensor()(processed_image)

if __name__ == '__main__':
  flags.mark_flag_as_required('ckpt')
  app.run(main)
