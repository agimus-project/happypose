from pathlib import Path

import pandas as pd

from happypose.pose_estimators.cosypose.cosypose.config import MEMORY


class TextureDataset:
    def __init__(self, ds_dir):
        ds_dir = Path(ds_dir)
        self.parse_image_dir = MEMORY.cache(self.parse_image_dir)
        self.index = self.parse_image_dir(ds_dir)

    @staticmethod
    def parse_image_dir(ds_dir):
        ds_dir = Path(ds_dir)
        index = []
        for im_path in ds_dir.glob("*"):
            if im_path.suffix in {".png", ".jpg"}:
                index.append({"texture_path": im_path})
        index = pd.DataFrame(index)
        return index

    def __getitem__(self, idx):
        return self.index.iloc[idx]

    def __len__(self):
        return len(self.index)
