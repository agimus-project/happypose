from pathlib import Path

import pandas as pd


class UrdfDataset:
    def __init__(self, urdf_ds_dir):
        urdf_ds_dir = Path(urdf_ds_dir)
        index = []
        for obj_dir in urdf_ds_dir.iterdir():
            urdf_paths = list(obj_dir.glob("*.urdf"))
            if len(urdf_paths) == 1:
                urdf_path = urdf_paths[0]
                # HACK for ycbv
                label = 'ycbv-'+obj_dir.name
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
