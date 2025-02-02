import os
import pickle
import random
import math
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
from .oxford_pets import OxfordPets  # subsample_classes 용

@DATASET_REGISTRY.register()
class FGVCAircraft(DatasetBase):
    dataset_dir = "fgvc_aircraft"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # -------------------------
        # 1) 전체 클래스 불러오기
        # -------------------------
        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())

        # cname2lab: 원본 클래스명 -> 원본 라벨
        cname2lab = {c: i for i, c in enumerate(classnames)}

        # lab2cname_full: 원본 라벨 -> 클래스명
        self.lab2cname_full = {i: c for i, c in enumerate(classnames)}

        # -------------------------
        # 2) train/val/test 읽기
        # -------------------------
        train = self.read_data(cname2lab, "images_variant_train.txt")
        val = self.read_data(cname2lab, "images_variant_val.txt")
        test = self.read_data(cname2lab, "images_variant_test.txt")

        num_shots = cfg.DATASET.NUM_SHOTS
        per_class_shots = cfg.DATASET.PER_CLASS_SHOTS
        seed = cfg.SEED
        random.seed(seed)

        # -------------------------
        # 3) Few-shot 처리
        # -------------------------
        if num_shots > 0:
            # uniform few-shot
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )
            if os.path.exists(preprocessed):
                print(f"[FGVCAircraft] Loading few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"[FGVCAircraft] Saving few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        elif num_shots < 0:
            # per-class few-shot
            if not per_class_shots:
                print("[FGVCAircraft] num_shots < 0 이지만 per_class_shots가 비어있어 few-shot 적용안함")
            else:
                preprocessed = os.path.join(
                    self.split_fewshot_dir, f"per_class_shots-seed_{seed}.pkl"
                )
                # if os.path.exists(preprocessed):
                #     print(f"[FGVCAircraft] Loading per-class few-shot data from {preprocessed}")
                #     with open(preprocessed, "rb") as file:
                #         data = pickle.load(file)
                #         train, val = data["train"], data["val"]
                # else:
                val_shots_list = [min(s, 4) for s in per_class_shots]
                train = self.generate_per_class_fewshot_dataset(train, per_class_shots)
                val = self.generate_per_class_fewshot_dataset(val, val_shots_list)
                data = {"train": train, "val": val}
                print(f"[FGVCAircraft] Saving per-class few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # num_shots == 0 => 원본 그대로
            pass

        # -------------------------
        # 4) 서브샘플링(base/new/all)
        # -------------------------
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []
        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]

                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)
        return items

    @staticmethod
    def generate_per_class_fewshot_dataset(dataset, shots_per_class):
        from collections import defaultdict
        tracker = defaultdict(list)
        for idx, item in enumerate(dataset):
            tracker[item.label].append(idx)

        new_dataset = []
        for cls_label, idxs in tracker.items():
            n_shots = shots_per_class[cls_label]
            random.shuffle(idxs)
            selected_idxs = idxs[:n_shots]
            for idx in selected_idxs:
                new_dataset.append(dataset[idx])
        return new_dataset

    @staticmethod
    def generate_fewshot_dataset(dataset, num_shots=1):
        from collections import defaultdict
        tracker = defaultdict(list)
        for idx, item in enumerate(dataset):
            tracker[item.label].append(idx)

        new_dataset = []
        for cls_label, idxs in tracker.items():
            random.shuffle(idxs)
            selected_idxs = idxs[:num_shots]
            for idx in selected_idxs:
                new_dataset.append(dataset[idx])
        return new_dataset
