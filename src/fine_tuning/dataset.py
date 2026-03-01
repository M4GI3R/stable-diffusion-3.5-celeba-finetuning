import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CelebADataset(Dataset):
    def __init__(self, csv_path, img_dir, tokenizer_one=None, tokenizer_two=None, tokenizer_three=None, size=1024):
        self.data = []
        self.img_dir = img_dir
        self.size = size
        
        # Read CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)
        
        # Transforms (Resize + Normalize to [-1, 1])
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.tokenizer_three = tokenizer_three

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        filename = item['filename']
        prompt = item['prompt']
        
        # Load Image
        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transforms(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy or skip (simplified: raise for now)
            raise e

        return {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "filename": filename
        }
