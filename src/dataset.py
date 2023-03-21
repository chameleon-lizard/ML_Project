from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Colored3DMNIST(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.float32))
            x = self.transform(x)
        
        return x
    
    def __len__(self):
        return len(self.data)