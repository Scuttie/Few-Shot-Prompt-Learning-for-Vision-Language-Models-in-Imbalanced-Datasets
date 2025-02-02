import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):

    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # ---------------------------
        # (1) train/val/test 불러오기
        # ---------------------------
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        # -----------------------------------------------------
        # (2) Few-shot 관련 설정
        # -----------------------------------------------------
        num_shots = cfg.DATASET.NUM_SHOTS         # int (양수면 uniform few-shot, 음수면 per-class)
        per_class_shots = cfg.DATASET.PER_CLASS_SHOTS  # list (디폴트: [])
        seed = cfg.SEED
        random.seed(seed)

        # -------------------------------------------------------------
        # (3) Few-shot 처리 로직
        # -------------------------------------------------------------
        #  - num_shots > 0  => uniform few-shot
        #  - num_shots < 0  => per-class few-shot(단, per_class_shots가 비어있지 않아야)
        #  - num_shots == 0 => few-shot 적용 안 함
        # -------------------------------------------------------------
        if num_shots > 0:
            # (3-1) 단일 정수 shot
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )
            if os.path.exists(preprocessed):
                print(f"[OxfordPets] Loading few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                # validation set은 너무 크지 않게 min(num_shots, 4)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"[OxfordPets] Saving few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        elif num_shots < 0:
            # (3-2) per-class few-shot
            #      num_shots<0 임을 'flag' 로 보고, 실제 샷 수는 per_class_shots에서 가져옴
            #      per_class_shots가 비어있다면 적용 X
            if len(per_class_shots) > 0:
                preprocessed = os.path.join(
                    self.split_fewshot_dir, f"per_class_shots-seed_{seed}.pkl"
                )
                #if os.path.exists(preprocessed):
                #     print(f"[OxfordPets] Loading per-class few-shot data from {preprocessed}")
                #     with open(preprocessed, "rb") as file:
                #         data = pickle.load(file)
                #         train, val = data["train"], data["val"]
                #else:
                # val에는 min(기존값, 4) 적용
                val_shots_list = [min(s, 4) for s in per_class_shots]
                train = self.generate_per_class_fewshot_dataset(train, per_class_shots)
                val = self.generate_per_class_fewshot_dataset(val, val_shots_list)

                data = {"train": train, "val": val}
                print(f"[OxfordPets] Saving per-class few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

                
            else:
                print("[OxfordPets] num_shots<0 이지만 per_class_shots가 비어있으므로 few-shot 미적용")
                # 아무 작업 안 함

        else:
            # (3-3) num_shots == 0 => few-shot 적용 안 함
            pass

        # ---------------------------
        # (4) 클래스 서브샘플링 (base/new/all)
        # ---------------------------
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        # ---------------------------
        # (5) DatasetBase 초기화
        # ---------------------------
        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []
        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)
        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        """단순히 전체 trainval을 일부 비율로 나누어 train/val로 분리"""
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            tracker[item.label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            # label당 val이 0이 되지 않도록
            assert n_val > 0
            random.shuffle(idxs)
            for i, idx in enumerate(idxs):
                if i < n_val:
                    val.append(trainval[idx])
                else:
                    train.append(trainval[idx])

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                # 절대경로->상대경로
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train_info = _extract(train)
        val_info   = _extract(val)
        test_info  = _extract(test)

        split = {"train": train_info, "val": val_info, "test": test_info}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        """저장해둔 JSON 스플릿을 읽어서 Datum 리스트로 복원"""
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath_full = os.path.join(path_prefix, impath)
                item = Datum(impath=impath_full, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])
        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups (base/new), or use all classes."""
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()

        n = len(labels)
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # 앞쪽 절반
        else:
            selected = labels[m:]  # 뒤쪽 절반

        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for ds in args:
            ds_new = []
            for item in ds:
                if item.label not in selected:
                    continue
                ds_new.append(
                    Datum(
                        impath=item.impath,
                        label=relabeler[item.label],
                        classname=item.classname
                    )
                )
            output.append(ds_new)
        return output

    @staticmethod
    def generate_per_class_fewshot_dataset(dataset, shots_per_class):
        """클래스별로 지정된 수(shots_per_class[cls])만큼 샘플링"""
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

    @staticmethod
    def generate_fewshot_dataset(dataset, num_shots=1):
        """각 클래스마다 num_shots개씩 균일하게 샘플링"""
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
