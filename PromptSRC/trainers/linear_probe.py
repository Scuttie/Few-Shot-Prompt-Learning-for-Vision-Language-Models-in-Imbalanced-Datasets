import torch
import torch.nn as nn
import torch.nn.functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from dassl.utils import load_checkpoint

from clip import clip
from clip.model import convert_weights


class MultiClassFocalLoss(nn.Module):
    """
    다중 클래스 Focal Loss
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        elif alpha is None:
            self.alpha = None
        else:
            raise TypeError('alpha must be None, list, or torch.Tensor')

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha = self.alpha[targets]
        else:
            alpha = 1.0

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def load_clip_to_cpu(cfg):
    """CLIP 모델을 CPU로 로드 후, build_model() 수행 (LinearProbe에서는 text encoder는 사용 안 함)."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": 'CoOp',
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


@TRAINER_REGISTRY.register()
class LinearProbeCLIP(TrainerX):
    """
    CLIP image encoder 위에 linear layer만 얹어서 분류하는 Linear Probe 예시.
    Focal Loss & CrossEntropyLoss 모두 지원하며, Focal Loss 시 클래스별 alpha 자동 계산 가능.
    """
    def build_model(self):
        cfg = self.cfg
        
        # ---------------------------------------------------------
        # 1) 고정된 100 클래스 대신, DatasetManager로부터 클래스 수를 받아옴
        #    (Dassl에서 보통 self.dm.num_classes 사용)
        # ---------------------------------------------------------
        num_classes = self.dm.num_classes
        print(f"[LinearProbeCLIP] Detected num_classes: {num_classes}")

        print(f"[LinearProbeCLIP] Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # CLIP 전체 파라미터 freeze
        for param in clip_model.parameters():
            param.requires_grad_(False)

        # CLIP을 GPU로 옮기고, FP32 변환
        clip_model.to(self.device)
        clip_model.float()

        # Image Encoder만 사용
        self.image_encoder = clip_model.visual
        embed_dim = clip_model.visual.output_dim

        # (추가) bias 사용 옵션
        use_bias = getattr(cfg.TRAINER.LINEAR_PROBE, "USE_BIAS", True)
        print(f"[LinearProbeCLIP] USE_BIAS = {use_bias}")

        # 학습할 Linear Layer 생성
        self.linear_head = nn.Linear(embed_dim, num_classes, bias=use_bias).to(self.device)
        self.linear_head.requires_grad_(True)

        # 손실 함수 결정
        loss_type = getattr(cfg.TRAINER.LINEAR_PROBE, "LOSS_TYPE", "ce")
        print(f"[LinearProbeCLIP] LOSS_TYPE = {loss_type}")

        if loss_type.lower() == "focal":
            print("[LinearProbeCLIP] -> Use Focal Loss")

            per_class = getattr(cfg.DATASET, "PER_CLASS_SHOTS", None)
            alpha = None
            if per_class is not None:
                if isinstance(per_class, str):
                    # 문자열 형태라면 리스트로 변환
                    per_class = list(map(int, per_class.strip("[]").split(",")))
                total_samples = sum(per_class)
                # alpha = total_samples / (num_classes * per_class[i])
                alpha = [
                    total_samples / (num_classes * cnt) if cnt > 0 else 0 
                    for cnt in per_class
                ]

            self.criterion = MultiClassFocalLoss(
                alpha=alpha,
                gamma=2,
                reduction='mean'
            )
        else:
            print("[LinearProbeCLIP] -> Use Cross Entropy Loss")
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer, Scheduler
        self.optim = build_optimizer(self.linear_head, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("linear_head", self.linear_head, self.optim, self.sched)

        self.clip_model = clip_model

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        # CLIP Image Encoder는 freeze되므로 grad 필요 X
        with torch.no_grad():
            feat = self.image_encoder(image)
            # feat = feat / feat.norm(dim=-1, keepdim=True)  # 필요시 정규화

        # Linear Classifier
        logits = self.linear_head(feat)
        loss = self.criterion(logits, label)

        self.model_backward_and_update(loss)

        acc = compute_accuracy(logits, label)[0].item()
        loss_summary = {"loss": loss.item(), "acc": acc}

        # epoch 마지막 배치 시점에서 LR 업데이트
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, image):
        """Dassl 엔진의 default test()에서 사용 -> self.model_inference()."""
        with torch.no_grad():
            feat = self.image_encoder(image)
            # feat = feat / feat.norm(dim=-1, keepdim=True)
            logits = self.linear_head(feat)
            probs = torch.softmax(logits, dim=1)
        return probs

    def parse_batch_train(self, batch):
        x = batch["img"].to(self.device)
        y = batch["label"].to(self.device)
        return x, y

    def load_model(self, directory, epoch=None):
        """저장된 linear_head만 불러오기."""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = f"model.pth.tar-{epoch}"

        for name in names:
            model_path = f"{directory}/{name}/{model_file}"
            ckpt = load_checkpoint(model_path)
            state_dict = ckpt["state_dict"]
            ep = ckpt["epoch"]
            print(f"Loading weights to {name} from {model_path} (epoch = {ep})")
            self._models[name].load_state_dict(state_dict, strict=False)
