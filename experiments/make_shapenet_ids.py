import json
import os
import pathlib as p
import typing as tp
from collections import deque
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ShapeNetSynset:
    id: str
    name: str
    parents: List[str]
    children: List[str]


@dataclass
class ModelInfo:
    obj_id: int
    shapenet_synset_id: str
    shapenet_source_id: str
    shapenet_names: str


def read_models(shapenet_dir):
    # TODO: This probably has issues / is poorly implemented and very slow
    taxonomy = json.load(open(shapenet_dir / "taxonomy.json"))

    id_to_synset: Dict[int, ShapeNetSynset] = {}

    for synset in taxonomy:
        synset_id = synset["synsetId"]
        id_to_synset[synset_id] = ShapeNetSynset(
            id=synset_id,
            name=synset["name"],
            children=synset["children"],
            parents=[],
        )

    for synset in taxonomy:
        for children in synset["children"]:
            id_to_synset[children].parents.append(synset["synsetId"])

    def get_names(synset_id, id_to_synset):
        nodes = deque()
        synset = id_to_synset[synset_id]
        nodes.append(synset)

        names = []
        while nodes:
            node = nodes.pop()
            names.append(node.name)
            for parent in node.parents:
                nodes.append(id_to_synset[parent])
        return names

    models_path = shapenet_dir.glob("**/**/models/model_normalized.obj")
    models: List[Dict[str, tp.Union[int, str]]] = []
    for n, model_path in enumerate(models_path):
        source_id = model_path.parent.parent.name
        synset_id = model_path.parent.parent.parent.name
        names = get_names(synset_id, id_to_synset)
        names = ",".join(names)
        models.append(
            {
                "obj_id": n,
                "shapenet_synset_id": synset_id,
                "shapenet_source_id": source_id,
                "shapenet_name": names,
            },
        )
    return models


if __name__ == "__main__":
    hp_data_dir = os.environ["HP_DATA_DIR"]
    shapenet_dir = p.Path(hp_data_dir) / "shapenetcorev2" / "models_orig"
    models = read_models(shapenet_dir)
    infos_path = p.Path(hp_data_dir) / "dataset-infos" / "shapenet_models.json"
    infos_path.write_text(json.dumps(models, indent=2))
