import os
import pickle
import random
import math

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # ---------------------------
        # (1) train/val/test 불러오기
        # ---------------------------
        if os.path.exists(self.split_path):
            # OxfordPets.read_split을 재사용(안에 Datum 변환 로직 있음)
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            # DTD.read_and_split_data를 재사용(내부에서 train/val/test 생성)
            train, val, test = DTD.read_and_split_data(
                self.image_dir,
                ignored=IGNORED,
                new_cnames=NEW_CNAMES
            )
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # ---------------------------
        # (2) Few-shot 관련 설정
        # ---------------------------
        num_shots = cfg.DATASET.NUM_SHOTS           # int (양수/음수/0)
        per_class_shots = cfg.DATASET.PER_CLASS_SHOTS  # list, default=[]
        seed = cfg.SEED
        random.seed(seed)

        # ---------------------------
        # (3) Few-shot 처리 로직
        # ---------------------------
        if num_shots > 0:
            # (3-1) uniform few-shot
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
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        elif num_shots < 0:
            # (3-2) per-class few-shot
            if len(per_class_shots) > 0:
                val_shots_list = [min(s, 4) for s in per_class_shots]
                preprocessed = os.path.join(
                    self.split_fewshot_dir, f"per_class_shots-seed_{seed}.pkl"
                )
                # 여기서는 매번 생성하도록 예시
                train = self.generate_per_class_fewshot_dataset(train, per_class_shots)
                val = self.generate_per_class_fewshot_dataset(val, val_shots_list)

                data = {"train": train, "val": val}
                print(f"[Caltech101] Saving per-class few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print("[Caltech101] num_shots<0 이지만 per_class_shots가 비어있으므로 few-shot 미적용")
                # 아무 작업 없음

        else:
            # (3-3) num_shots == 0 => few-shot 미적용
            pass

        # ---------------------------
        # (4) 클래스 서브샘플링 (base/new/all)
        # ---------------------------
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        # ---------------------------
        # (5) DatasetBase 초기화
        # ---------------------------
        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def generate_fewshot_dataset(dataset, num_shots=1):
        """uniform few-shot, 클래스마다 동일한 num_shots개"""
        from collections import defaultdict
        tracker = defaultdict(list)
        for idx, item in enumerate(dataset):
            tracker[item.label].append(idx)

        new_dataset = []
        for cls_label, idxs in tracker.items():
            random.shuffle(idxs)
            selected_idxs = idxs[:num_shots]
            for idx_ in selected_idxs:
                new_dataset.append(dataset[idx_])
        return new_dataset

    @staticmethod
    def generate_per_class_fewshot_dataset(dataset, shots_per_class):
        """클래스별 shots_per_class[label]개만큼 샘플링"""
        from collections import defaultdict
        tracker = defaultdict(list)
        for idx, item in enumerate(dataset):
            tracker[item.label].append(idx)

        new_dataset = []
        for cls_label, idxs in tracker.items():
            n_shots = shots_per_class[cls_label]
            random.shuffle(idxs)
            selected_idxs = idxs[:n_shots]
            for idx_ in selected_idxs:
                new_dataset.append(dataset[idx_])
        return new_dataset
