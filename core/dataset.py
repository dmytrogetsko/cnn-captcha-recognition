import torch
from PIL import Image
from torch.utils.data import Dataset
import os

class CaptchaDataset(Dataset):
    def __init__(self, root_dir, captcha_length, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.char_set = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.char_to_label = {char: idx for idx, char in enumerate(self.char_set)}

        self.image_files = []
        for f in os.listdir(root_dir):
          if not f.endswith('.png'):
              continue
          name = os.path.splitext(f)[0]
          if len(name) != captcha_length:
              print(f"Skip (wrong filename length): {f}")
              continue
          if any(c not in self.char_set for c in name):
              print(f"Skip (undefined symbols): {f}")
              continue
          self.image_files.append(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_str = os.path.splitext(img_name)[0]
        label = torch.tensor([self.char_to_label[char] for char in label_str], dtype=torch.long)
        image = Image.open(os.path.join(self.root_dir, img_name)).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label