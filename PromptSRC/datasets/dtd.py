import os
import pickle
import random
import math

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class DescribableTextures(DatasetBase):

    dataset_dir = "dtd"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_DescribableTextures.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # (1) train/val/test 읽기
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # (2) Few-shot 관련 처리
        num_shots = cfg.DATASET.NUM_SHOTS           # int
        per_class_shots = cfg.DATASET.PER_CLASS_SHOTS  # list
        seed = cfg.SEED
        random.seed(seed)

        if num_shots > 0:
            # uniform few-shot
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        elif num_shots < 0:
            # per-class few-shot
            if len(per_class_shots) > 0:
                val_shots = [min(s, 4) for s in per_class_shots]
                preprocessed = os.path.join(
                    self.split_fewshot_dir, f"per_class_shots-seed_{seed}.pkl"
                )
                # 매번 생성 (필요시 캐싱 가능)
                train = self.generate_per_class_fewshot_dataset(train, per_class_shots)
                val = self.generate_per_class_fewshot_dataset(val, val_shots)

                data = {"train": train, "val": val}
                print(f"Saving per-class few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print("[DTD] num_shots<0 이지만 per_class_shots가 비어있음 => 미적용")

        else:
            # num_shots == 0 => 원본 데이터 그대로
            pass

        # (3) 클래스 서브샘플링
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        # (4) DatasetBase 초기화
        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(
        image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None
    ):
        """원본 데이터를 (50% train, 20% val, 30% test)로 분할"""
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            c = category
            if new_cnames is not None and category in new_cnames:
                c = new_cnames[category]

            train.extend(_collate(images[:n_train], label, c))
            val.extend(_collate(images[n_train : n_train + n_val], label, c))
            test.extend(_collate(images[n_train + n_val :], label, c))

        return train, val, test

    @staticmethod
    def generate_fewshot_dataset(dataset, num_shots=1):
        """uniform few-shot"""
        from collections import defaultdict
        tracker = defaultdict(list)
        for idx, item in enumerate(dataset):
            tracker[item.label].append(idx)

        new_dataset = []
        for cls_label, idxs in tracker.items():
            random.shuffle(idxs)
            selected = idxs[:num_shots]
            for s in selected:
                new_dataset.append(dataset[s])
        return new_dataset

    @staticmethod
    def generate_per_class_fewshot_dataset(dataset, shots_per_class):
        """클래스별 shots_per_class[label]개씩 샘플링"""
        from collections import defaultdict
        tracker = defaultdict(list)
        for idx, item in enumerate(dataset):
            tracker[item.label].append(idx)

        new_dataset = []
        for cls_label, idxs in tracker.items():
            n_shots = shots_per_class[cls_label]
            random.shuffle(idxs)
            selected = idxs[:n_shots]
            for s in selected:
                new_dataset.append(dataset[s])
        return new_dataset
