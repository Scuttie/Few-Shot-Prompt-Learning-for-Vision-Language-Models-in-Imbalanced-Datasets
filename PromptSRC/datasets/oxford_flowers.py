import os
import pickle
import random
import math
from scipy.io import loadmat
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing, write_json

@DATASET_REGISTRY.register()
class OxfordFlowers(DatasetBase):

    dataset_dir = "oxford_flowers"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # ---------------------------
        # (1) train/val/test 불러오기
        # ---------------------------
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_data()
            self.save_split(train, val, test, self.split_path, self.image_dir)

        # ---------------------------
        # (2) Few-shot 관련 설정
        # ---------------------------
        num_shots = cfg.DATASET.NUM_SHOTS                # 정수 (양수/음수/0)
        per_class_shots = cfg.DATASET.PER_CLASS_SHOTS     # 리스트 (기본=[]), 음수면 여기서 사용
        seed = cfg.SEED
        random.seed(seed)

        # ---------------------------
        # (3) Few-shot 처리 로직
        # ---------------------------
        #  - num_shots > 0  => uniform few-shot
        #  - num_shots < 0  => per-class few-shot (per_class_shots 사용)
        #  - num_shots == 0 => few-shot 미적용
        # ---------------------------
        if num_shots > 0:
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
            # per-class few-shot
            if len(per_class_shots) > 0:
                # per_class_shots를 그대로 train에 적용
                # val은 너무 크지 않도록 각 클래스별 min(해당샷,4) 적용
                val_shots_list = [min(s, 4) for s in per_class_shots]

                preprocessed = os.path.join(
                    self.split_fewshot_dir, f"per_class_shots-seed_{seed}.pkl"
                )
                # 필요하다면 캐시 로딩/저장 가능
                # 여기서는 매번 생성하도록 예시

                train = self.generate_per_class_fewshot_dataset(train, per_class_shots)
                val = self.generate_per_class_fewshot_dataset(val, val_shots_list)

                data = {"train": train, "val": val}
                print(f"[OxfordFlowers] Saving per-class few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print("[OxfordFlowers] num_shots<0 이지만 per_class_shots가 비어있으므로 few-shot 미적용")
                # 아무 작업 안 함

        else:
            # num_shots == 0 => few-shot 미적용
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

    def read_data(self):
        """기존 OxfordFlowers 데이터 전체를 50% train, 20% val, 30% test로 쪼개는 함수"""
        from dassl.utils import read_json

        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)  # 1~102
            tracker[label].append(impath)

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                # y-1로 해서 0-based label
                item = Datum(impath=im, label=y - 1, classname=c)
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)  # { "1":"rose", "2":"tulip", ... }
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0, f"Too few samples in class {label}"
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train : n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val :], label, cname))

        return train, val, test

    def save_split(self, train, val, test, filepath, path_prefix):
        """train/val/test 리스트를 json에 저장"""
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                # 절대경로 -> 상대경로
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train_info = _extract(train)
        val_info = _extract(val)
        test_info = _extract(test)
        split = {"train": train_info, "val": val_info, "test": test_info}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    def read_split(self, filepath, path_prefix):
        """저장된 split json을 읽어 Datum 형태로 복원"""
        from dassl.utils import read_json
        print(f"Reading split from {filepath}")
        split = read_json(filepath)

        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath_full = os.path.join(path_prefix, impath)
                item = Datum(impath=impath_full, label=int(label), classname=classname)
                out.append(item)
            return out

        train = _convert(split["train"])
        val   = _convert(split["val"])
        test  = _convert(split["test"])
        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """base/new/all 중 선택. base=> 앞쪽 절반, new=> 뒤쪽 절반, all=>전체"""
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = sorted(list(labels))
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
    def generate_fewshot_dataset(dataset, num_shots=1):
        """uniform few-shot, 클래스마다 동일한 num_shots"""
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
        """shots_per_class[label] 만큼 해당 label에서 샘플링"""
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
