import os
import pickle
import random
from collections import OrderedDict
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets  # subsample_classes 재사용


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # 사전 캐싱된 preprocessed.pkl (train/val 목록) 불러오기
        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # ImageNet 전통적으로 val 디렉토리를 test 용으로 사용
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        # =====================================================================
        # Few-shot 로직
        # =====================================================================
        num_shots = cfg.DATASET.NUM_SHOTS
        per_class_shots = cfg.DATASET.PER_CLASS_SHOTS
        seed = cfg.SEED
        random.seed(seed)

        if num_shots > 0:
            # ----------------------------------------------------------
            # (1) uniform few-shot
            # ----------------------------------------------------------
            preprocessed_fs = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )
            if os.path.exists(preprocessed_fs):
                print(f"[ImageNet] Loading few-shot data from {preprocessed_fs}")
                with open(preprocessed_fs, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
                    test = data["test"]
            else:
                print(f"[ImageNet] Generating uniform few-shot (train={num_shots}, val=min({num_shots}, 4))")
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                # test를 val 용도로도 쓰니, 너무 커지지 않도록 min(num_shots, 4)
                test = self.generate_fewshot_dataset(test, num_shots=min(num_shots, 4))

                data = {"train": train, "test": test}
                print(f"[ImageNet] Saving few-shot data to {preprocessed_fs}")
                with open(preprocessed_fs, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        elif num_shots < 0:
            # ----------------------------------------------------------
            # (2) per-class few-shot
            #     per_class_shots에 클래스별 샷 수가 지정
            # ----------------------------------------------------------
            if len(per_class_shots) > 0:
                preprocessed_fs = os.path.join(
                    self.split_fewshot_dir, f"per_class_shots-seed_{seed}.pkl"
                )
                # if os.path.exists(preprocessed_fs):
                #     print(f"[ImageNet] Loading per-class few-shot data from {preprocessed_fs}")
                #     with open(preprocessed_fs, "rb") as file:
                #         data = pickle.load(file)
                #         train = data["train"]
                #         test = data["test"]
                # else:
                # validation/test는 너무 작아지지 않게 min(., 4) 적용
                test_shots_list = [min(s, 4) for s in per_class_shots]

                print("[ImageNet] Generating per-class few-shot dataset...")
                train = self.generate_per_class_fewshot_dataset(train, per_class_shots)
                test = self.generate_per_class_fewshot_dataset(test, test_shots_list)

                data = {"train": train, "test": test}
                print(f"[ImageNet] Saving per-class few-shot data to {preprocessed_fs}")
                with open(preprocessed_fs, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # num_shots < 0인데 per_class_shots가 비어있으면 few-shot 미적용
                print("[ImageNet] num_shots<0 이지만 PER_CLASS_SHOTS가 비어 있어 few-shot 미적용")

        else:
            # ----------------------------------------------------------
            # (3) num_shots == 0 => 그대로 사용
            # ----------------------------------------------------------
            pass

        # =====================================================================
        # 클래스 서브샘플링 (base/new/all)
        # =====================================================================
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>."""
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        # 폴더(클래스) 목록
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    @staticmethod
    def generate_fewshot_dataset(dataset, num_shots=1):
        """각 클래스마다 num_shots 개씩 균일하게 샘플링"""
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
        """클래스별로 지정된 shots_per_class[label] 만큼 샘플링"""
        from collections import defaultdict
        tracker = defaultdict(list)
        for idx, item in enumerate(dataset):
            tracker[item.label].append(idx)

        new_dataset = []
        for cls_label, idxs in tracker.items():
            # 혹시 shots_per_class가 클래스 개수보다 적게 들어왔거나
            # label 순서가 다른 경우를 대비하여 index 범위 체크
            if cls_label < len(shots_per_class):
                n_shots = shots_per_class[cls_label]
            else:
                # 만약 shots_per_class 길이가 부족하면 일단 0으로 처리
                n_shots = 0

            random.shuffle(idxs)
            selected_idxs = idxs[:n_shots]
            for idx_ in selected_idxs:
                new_dataset.append(dataset[idx_])
        return new_dataset
