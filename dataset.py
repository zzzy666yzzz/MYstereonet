import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F  # 确保引用了这个
from PIL import Image

# ImageNet 统计量
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class StereoDataset(data.Dataset):
    def __init__(self, data_root, train=True, train_ratio=0.9, seed=42,
                 crop_size=(320, 640), val_target_size=(704, 1280)):
        super(StereoDataset, self).__init__()
        self.root = data_root
        self.train = train
        self.crop_size = crop_size
        self.val_target_size = val_target_size

        left_dir = os.path.join(data_root, 'left')
        right_dir = os.path.join(data_root, 'right')
        disp_dir = os.path.join(data_root, 'labels')

        self.samples = []
        if os.path.exists(left_dir):
            left_imgs = sorted(os.listdir(left_dir))
            for lname in left_imgs:
                if not (lname.endswith('.jpg') or lname.endswith('.png')):
                    continue

                basename = os.path.splitext(lname)[0]
                rname = lname
                dname = basename + '.png'

                if not os.path.exists(os.path.join(disp_dir, dname)):
                    if os.path.exists(os.path.join(disp_dir, basename + '.pfm')):
                        dname = basename + '.pfm'

                if os.path.exists(os.path.join(right_dir, rname)) and \
                        os.path.exists(os.path.join(disp_dir, dname)):
                    self.samples.append({
                        'left': os.path.join(left_dir, lname),
                        'right': os.path.join(right_dir, rname),
                        'disp': os.path.join(disp_dir, dname)
                    })

        if len(self.samples) > 0:
            random.seed(seed)
            random.shuffle(self.samples)
            split_idx = int(len(self.samples) * train_ratio)
            if self.train:
                self.samples = self.samples[:split_idx]
            else:
                self.samples = self.samples[split_idx:]

        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __getitem__(self, index):
        sample = self.samples[index]

        imgL = Image.open(sample['left']).convert('RGB')
        imgR = Image.open(sample['right']).convert('RGB')

        disp_path = sample['disp']
        if disp_path.endswith('.pfm'):
            disp_np = self._read_pfm(disp_path)
        else:
            disp_img = Image.open(disp_path)
            disp_np = np.ascontiguousarray(disp_img, dtype=np.float32)

            # 强制转单通道
            if disp_np.ndim == 3:
                disp_np = disp_np[:, :, 0]

            if disp_np.max() > 1000:
                disp_np = disp_np / 256.0

        disp = torch.from_numpy(disp_np).unsqueeze(0)

        if self.train:

            # 训练增强
            b_factor = random.uniform(0.6, 1.4)
            c_factor = random.uniform(0.6, 1.4)
            s_factor = random.uniform(0.6, 1.4)
            h_factor = random.uniform(-0.1, 0.1)

            transforms_list = [
                lambda x: TF.adjust_brightness(x, b_factor),
                lambda x: TF.adjust_contrast(x, c_factor),
                lambda x: TF.adjust_saturation(x, s_factor),
                lambda x: TF.adjust_hue(x, h_factor)
            ]
            random.shuffle(transforms_list)

            for t in transforms_list:
                imgL = t(imgL)
                imgR = t(imgR)

            imgL = TF.to_tensor(imgL)
            imgR = TF.to_tensor(imgR)
            imgL, imgR, disp = self.random_crop(imgL, imgR, disp, self.crop_size)

        else:
            # 验证 Padding
            imgL = TF.to_tensor(imgL)
            imgR = TF.to_tensor(imgR)
            # 这里的调用现在是安全的
            imgL, imgR, disp = self.validate_padding(imgL, imgR, disp, self.val_target_size)

        imgL = self.normalize(imgL)
        imgR = self.normalize(imgR)

        return imgL, imgR, disp

    def random_crop(self, imgL, imgR, disp, crop_size):
        h, w = imgL.shape[1], imgL.shape[2]
        th, tw = crop_size

        if w < tw or h < th:
            pad_h = max(th - h, 0)
            pad_w = max(tw - w, 0)
            # 使用 F.pad (torch.nn.functional) 更加稳健
            imgL = F.pad(imgL, (0, pad_w, 0, pad_h))
            imgR = F.pad(imgR, (0, pad_w, 0, pad_h))
            disp = F.pad(disp, (0, pad_w, 0, pad_h))
            h, w = th, tw

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        imgL = imgL[:, y1:y1 + th, x1:x1 + tw]
        imgR = imgR[:, y1:y1 + th, x1:x1 + tw]
        disp = disp[:, y1:y1 + th, x1:x1 + tw]
        return imgL, imgR, disp

    def validate_padding(self, imgL, imgR, disp, target_size):
        if target_size is None: return imgL, imgR, disp
        th, tw = target_size
        h, w = imgL.shape[1], imgL.shape[2]
        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)

        # F.pad 的 padding 参数顺序是 (Left, Right, Top, Bottom)
        # 且它接受 Tensor 作为输入，这是最正确的用法
        padding = (0, pad_w, 0, pad_h)

        imgL = F.pad(imgL, padding, mode='constant', value=0)
        imgR = F.pad(imgR, padding, mode='constant', value=0)
        disp = F.pad(disp, padding, mode='constant', value=0)

        return imgL, imgR, disp

    def _read_pfm(self, file):
        with open(file, 'rb') as f:
            header = f.readline().rstrip()
            color = (header.decode('ascii') == 'PF')
            dim_match = f.readline().rstrip()
            scale = float(f.readline().rstrip())
            dims = list(map(int, dim_match.split()))
            data = np.fromfile(f, '<f')
            shape = (dims[1], dims[0], 3) if color else (dims[1], dims[0])
            data = np.reshape(data, shape)
            data = np.flipud(data)
        return data

    def __len__(self):
        return len(self.samples)