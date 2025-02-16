import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.promptsrc
import trainers.linear_probe

from trainers.simclr_utils import SimCLRDataset, simclr_transform, simclr_collate_fn
from torch.utils.data import DataLoader


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed:
        cfg.SEED = args.seed
    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

def extend_cfg(cfg):
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp16"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.COOP.USE_FOCAL_LOSS = False
    cfg.TRAINER.COOP.LOSS_TYPE = "ce"

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""
    cfg.TRAINER.COCOOP.PREC = "fp16"
    cfg.TRAINER.COCOOP.USE_FOCAL_LOSS = False

    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.MAPLE.PREC = "fp16"
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.TRAINER.MAPLE.USE_FOCAL_LOSS = False

    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1
    cfg.TRAINER.PROMPTSRC.LABEL_SCOPE = "default"
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.TRAINER.PROMPTSRC.LOSS_TYPE = "ce"
    cfg.TRAINER.PROMPTSRC.SIMCLR_ALPHA = 0.0

    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"
    cfg.TRAINER.IVLP.PREC = "fp16"
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.TRAINER.IVLP.USE_FOCAL_LOSS = False
    cfg.TRAINER.IVLP.SIMCLR_ALPHA = 0.0
    cfg.TRAINER.IVLP.USE_MIXUP = False
    cfg.TRAINER.IVLP.MIXUP_ALPHA = 1.0
    cfg.TRAINER.IVLP.USE_KD = False
    cfg.TRAINER.IVLP.KD_TEACHER_MODEL = "resnet50"
    cfg.TRAINER.IVLP.KD_ALPHA = 1.0
    cfg.TRAINER.IVLP.KD_T = 4.0

    cfg.TRAINER.LINEAR_PROBE = CN()
    cfg.TRAINER.LINEAR_PROBE.LOSS_TYPE = "ce"
    cfg.TRAINER.LINEAR_PROBE.USE_BIAS = True

    if not hasattr(cfg.DATASET, "PER_CLASS_SHOTS"):
        cfg.DATASET.PER_CLASS_SHOTS = []

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    # GPU
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    # ---------------------------
    # (A) Alias 매핑
    # ---------------------------
    dataset_alias = {
        "DescribableTextures": "dtd",
        "OxfordPets": "oxford_pets",
        "OxfordFlowers": "oxford_flowers",
        "FGVCAircraft": "fgvc_aircraft",
        "Caltech101": "caltech101",
        "EuroSAT": "eurosat",
        "Food101": "food101",
        "UCF101": "ucf101",
        "StanfordCars": "stanford_cars",
        "SUN397": "sun397",
        "ImageNet": "imagenet"
        # 필요에 따라 추가
    }

    # ---------------------------
    # (B) base_label_count 매핑
    # ---------------------------
    dataset_name_to_basecount = {
        "dtd": 24,
        "oxford_pets": 19,
        "oxford_flowers": 51,
        "fgvc_aircraft": 50,
        "caltech101": 51,
        "food101": 51,
        "ucf101": 51,
        "stanford_cars": 98,
        "sun397": 199,
        "eurosat": 5,
        "imagenet": 500
    }

    # Dassl 프레임워크에서 불러온 config상 dataset name
    dataset_name = getattr(cfg.DATASET, "NAME", None)  # e.g. "DescribableTextures"

    # Alias 변환
    if dataset_name in dataset_alias:
        alias_name = dataset_alias[dataset_name]
        print(f"[INFO] Convert dataset_name '{dataset_name}' -> alias '{alias_name}'")
    else:
        alias_name = dataset_name

    # base_label_count 결정
    if alias_name in dataset_name_to_basecount:
        base_label_count = dataset_name_to_basecount[alias_name]
        print(f"[INFO] base_label_count={base_label_count}")
    else:
        base_label_count = 0
        print(f"[WARNING] alias_name '{alias_name}' not in dict; base_label_count set to 0")

    # 트레이너 생성
    from dassl.engine import build_trainer
    trainer = build_trainer(cfg)

    # SIMCLR 교체(필요 시)
    if cfg.TRAINER.PROMPTSRC.SIMCLR_ALPHA > 0:
        print(">> SIMCLR_ALPHA > 0 => Overriding train_loader_x with a SimCLR DataLoader!")
        base_data_list = trainer.dm.dataset.train_x
        from torchvision.datasets.folder import default_loader

        class MyBaseDataset(torch.utils.data.Dataset):
            def __init__(self, datum_list):
                self.datum_list = datum_list
                self.loader = default_loader
            def __len__(self):
                return len(self.datum_list)
            def __getitem__(self, idx):
                d = self.datum_list[idx]
                img = self.loader(d.impath)
                return img, d.label

        base_dataset = MyBaseDataset(base_data_list)
        simclr_ds = SimCLRDataset(base_dataset, transform=simclr_transform)
        simclr_loader = DataLoader(
            simclr_ds,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=True,
            collate_fn=simclr_collate_fn
        )
        trainer.dm.train_loader_x = simclr_loader

    # ---------------------------
    # eval-only
    # ---------------------------
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        y_true, y_pred = trainer.test(return_pred=True)

        from sklearn.metrics import classification_report
        print("\n===========================")
        print("Classification Report")
        print("===========================")
        print(classification_report(y_true, y_pred))

        if base_label_count > 0:
            y_true_t = torch.tensor(y_true)
            y_pred_t = torch.tensor(y_pred)
            base_mask = (y_true_t < base_label_count)
            new_mask = (y_true_t >= base_label_count)
            base_correct = (y_pred_t[base_mask] == y_true_t[base_mask]).sum().item()
            new_correct = (y_pred_t[new_mask] == y_true_t[new_mask]).sum().item()
            base_total = base_mask.sum().item()
            new_total = new_mask.sum().item()
            base_acc = 100.0 * base_correct / base_total if base_total else 0.0
            new_acc = 100.0 * new_correct / new_total if new_total else 0.0
            print(f"Base class accuracy: {base_acc:.2f}% ({base_correct}/{base_total})")
            print(f"New  class accuracy: {new_acc:.2f}% ({new_correct}/{new_total})")
        return

    # ---------------------------
    # Train
    # ---------------------------
    if not args.no_train:
        trainer.train()
        print(">>> Evaluating on the test set right after training...")
        y_true, y_pred = trainer.test(return_pred=True)

        from sklearn.metrics import classification_report
        print("\n===========================")
        print("Classification Report")
        print("===========================")
        print(classification_report(y_true, y_pred))

        if base_label_count > 0:
            y_true_t = torch.tensor(y_true)
            y_pred_t = torch.tensor(y_pred)
            base_mask = (y_true_t < base_label_count)
            new_mask = (y_true_t >= base_label_count)
            base_correct = (y_pred_t[base_mask] == y_true_t[base_mask]).sum().item()
            new_correct = (y_pred_t[new_mask] == y_true_t[new_mask]).sum().item()
            base_total = base_mask.sum().item()
            new_total = new_mask.sum().item()
            base_acc = 100.0 * base_correct / base_total if base_total else 0.0
            new_acc = 100.0 * new_correct / new_total if new_total else 0.0
            print(f"Base class accuracy: {base_acc:.2f}% ({base_correct}/{base_total})")
            print(f"New  class accuracy: {new_acc:.2f}% ({new_correct}/{new_total})")

    # 디버그
    first_batch = next(iter(trainer.dm.train_loader_x))
    print("DEBUG batch keys:", first_batch.keys())
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory")
    parser.add_argument("--seed", type=int, default=-1, help="only positive => fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="", help="dataset config")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="dir for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="which epoch to load weight")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    args = parser.parse_args()
    main(args)
