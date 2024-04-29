import json
from pathlib import Path

import pandas as pd


class UrdfDataset:
    def __init__(self, urdf_ds_dir, label_filename="objname2label.json"):
        urdf_ds_dir = Path(urdf_ds_dir)
        label_path = urdf_ds_dir / label_filename
        if label_path.exists():
            with label_path.open() as fp:
                objname2label = json.load(fp)
        else:
            objname2label = None
        index = []
        for obj_dir in urdf_ds_dir.iterdir():
            urdf_paths = list(obj_dir.glob("*.urdf"))
            if len(urdf_paths) == 1:
                urdf_path = urdf_paths[0]
                if objname2label is None:
                    label = obj_dir.name
                else:
                    label = objname2label[obj_dir.name]
                infos = {
                    "label": label,
                    "urdf_path": urdf_path.as_posix(),
                    "scale": 1.0,
                }
                index.append(infos)
        self.index = pd.DataFrame(index)

    def __getitem__(self, idx):
        return self.index.iloc[idx]

    def __len__(self):
        return len(self.index)


class OneUrdfDataset:
    def __init__(self, urdf_path, label, scale=1.0):
        index = [
            {"urdf_path": urdf_path, "label": label, "scale": scale},
        ]
        self.index = pd.DataFrame(index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.index.iloc[idx]
