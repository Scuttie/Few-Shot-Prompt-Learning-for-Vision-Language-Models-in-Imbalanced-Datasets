import os
import pickle
import random
import math

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


@DATASET_REGISTRY.register()
class Food101(DatasetBase):

    dataset_dir = "food-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # (1) train/val/test
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            # DTD.read_and_split_data => 101개 클래스
            train, val, test = DTD.read_and_split_data(self.image_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # (2) few-shot
        num_shots = cfg.DATASET.NUM_SHOTS
        per_class_shots = cfg.DATASET.PER_CLASS_SHOTS
        seed = cfg.SEED
        random.seed(seed)

        if num_shots > 0:
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )
            if os.path.exists(preprocessed):
                print(f"Loading few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val   = self.generate_fewshot_dataset(val, num_shots=min(num_shots,4))
                data = {"train": train, "val": val}
                print(f"Saving few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif num_shots < 0:
            if len(per_class_shots) > 0:
                val_shots = [min(s, 4) for s in per_class_shots]
                preprocessed = os.path.join(
                    self.split_fewshot_dir, f"per_class_shots-seed_{seed}.pkl"
                )
                train = self.generate_per_class_fewshot_dataset(train, per_class_shots)
                val   = self.generate_per_class_fewshot_dataset(val, val_shots)
                data = {"train": train, "val": val}
                print(f"Saving per-class few-shot to {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print("[Food101] num_shots<0 but no per_class_shots => skip few-shot")
        else:
            pass

        # (3) subsample
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def generate_fewshot_dataset(dataset, num_shots=1):
        from collections import defaultdict
        tracker = defaultdict(list)
        for i, item in enumerate(dataset):
            tracker[item.label].append(i)

        new_data = []
        for cls_label, idxs in tracker.items():
            random.shuffle(idxs)
            selected = idxs[:num_shots]
            for s in selected:
                new_data.append(dataset[s])
        return new_data

    @staticmethod
    def generate_per_class_fewshot_dataset(dataset, shots_per_class):
        from collections import defaultdict
        tracker = defaultdict(list)
        for i, item in enumerate(dataset):
            tracker[item.label].append(i)

        new_data = []
        for cls_label, idxs in tracker.items():
            n_shots = shots_per_class[cls_label]
            random.shuffle(idxs)
            selected = idxs[:n_shots]
            for s in selected:
                new_data.append(dataset[s])
        return new_data
