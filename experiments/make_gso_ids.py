import os
import json
import pathlib as p


if __name__ == "__main__":
    hp_data_dir = os.environ["HP_DATA_DIR"]
    gso_dir = p.Path(hp_data_dir) / "google_scanned_objects" / "models_orig"
    models = []
    for n, model_path in enumerate(gso_dir.glob("**/meshes/model.obj")):
        models.append(dict(obj_id=n, gso_id=model_path.parent.parent.name))
    infos_path = p.Path(hp_data_dir) / "dataset-infos" / "gso_models.json"
    infos_path.write_text(json.dumps(models, indent=2))
