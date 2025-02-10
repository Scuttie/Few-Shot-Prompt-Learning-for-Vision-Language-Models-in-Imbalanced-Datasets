import os.path as osp
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import timm  # 추가 (Teacher 모델 로드용)

_tokenizer = _Tokenizer()

##########################
# Mixup Loss Helper
##########################
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    y_a, y_b는 라벨 텐서
    lam은 mixup 파라미터
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

##########################
# Knowledge Distillation
##########################
def distillation_loss(
    student_logits, 
    teacher_logits, 
    labels, 
    criterion_ce, 
    T=4.0, 
    alpha=0.5, 
    lam=None, 
    y_a=None, 
    y_b=None
):
    """
    student_logits: 학생 모델 출력 로짓
    teacher_logits: 교사 모델 출력 로짓
    labels: GT 라벨 (hard label)
    criterion_ce: CE Loss (혹은 Focal Loss)
    T: temperature
    alpha: hard-vs-soft 비중(하드 라벨 loss 비중)
    lam: mixup 파라미터 (없으면 None)
    y_a, y_b: mixup용 라벨
    """
    # hard label loss
    if lam is not None and y_a is not None and y_b is not None:
        # mixup이 적용된 상황
        hard_loss = mixup_criterion(criterion_ce, student_logits, y_a, y_b, lam)
    else:
        # 일반 CE(Focal) loss
        hard_loss = criterion_ce(student_logits, labels)

    # soft label (teacher) => KL Div
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T**2)

    return alpha * hard_loss + (1 - alpha) * kd_loss


##########################
# 기존 IVLP Loss 등
##########################
class ImageNTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        N2 = z.shape[0]  # 2N
        sim = torch.matmul(z, z.t()) / self.temperature

        # exclude diagonal
        mask = ~torch.eye(N2, dtype=bool, device=z.device)
        sim_masked = sim[mask].view(N2, -1)

        # positive pairs: z1[i] <-> z2[i]
        row_idx = torch.arange(N2, device=z.device)
        pos_idx = torch.arange(N2, device=z.device)
        N = N2 // 2
        pos_idx[:N] += N
        pos_idx[N:] -= N

        pos_vals = sim[row_idx, pos_idx].unsqueeze(1)

        # negatives
        full_c = torch.arange(N2, device=z.device).unsqueeze(0)
        row_c = row_idx.unsqueeze(1)
        pos_c = pos_idx.unsqueeze(1)
        row_mask = (full_c != row_c) & (full_c != pos_c)

        neg_list = []
        for i in range(N2):
            row_neg = sim[i][row_mask[i]]
            neg_list.append(row_neg.unsqueeze(0))
        neg_vals = torch.cat(neg_list, dim=0)

        out = torch.cat([pos_vals, neg_vals], dim=1)

        labels = torch.zeros(N2, dtype=torch.long, device=z.device)
        loss = self.ce(out, labels)
        return loss


class MultiClassFocalLoss(nn.Module):
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
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": 'IVLP',
        "vision_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION,
        "language_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT,
        "vision_ctx": cfg.TRAINER.IVLP.N_CTX_VISION,
        "language_ctx": cfg.TRAINER.IVLP.N_CTX_TEXT
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = x @ self.text_projection
        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        assert cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT >= 1

        n_ctx = cfg.TRAINER.IVLP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init and (n_ctx <= 4):
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.IVLP.N_CTX_VISION}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.use_focal_loss = cfg.TRAINER.IVLP.get("USE_FOCAL_LOSS", False)
        print(f">> USE_FOCAL_LOSS = {self.use_focal_loss}")

        per_class = getattr(cfg.DATASET, "PER_CLASS_SHOTS", None)
        n_cls = len(classnames)

        if self.use_focal_loss:
            print(">> Use Focal Loss!")
            alpha = None
            if per_class is not None:
                if isinstance(per_class, str):
                    per_class = list(map(int, per_class.strip("[]").split(",")))
                total_samples = sum(per_class)
                alpha = [
                    total_samples / (n_cls * count) if count > 0 else 0.0 
                    for count in per_class
                ]
            self.criterion_ce = MultiClassFocalLoss(alpha=alpha, gamma=2, reduction='mean')
        else:
            print(">> Use Cross Entropy Loss!")
            self.criterion_ce = nn.CrossEntropyLoss()

        # SimCLR
        self.simclr_alpha = getattr(cfg.TRAINER.IVLP, "SIMCLR_ALPHA", 0.0)
        print(f">> SimCLR alpha = {self.simclr_alpha}")
        self.criterion_simclr = ImageNTXentLoss(temperature=0.07) if self.simclr_alpha > 0 else None

    def forward(self, image1, label=None, image2=None):
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features1 = self.image_encoder(image1.type(self.dtype))
        image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features1 @ text_features.t()

        loss_ce = None
        if self.training and (label is not None):
            loss_ce = self.criterion_ce(logits, label)

        loss_simclr = None
        if (self.training) and (image2 is not None) and (self.simclr_alpha > 0.0):
            image_features2 = self.image_encoder(image2.type(self.dtype))
            image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)
            loss_simclr = self.criterion_simclr(image_features1, image_features2)

        if self.training:
            total_loss = 0.0 if loss_ce is None else loss_ce
            if loss_simclr is not None:
                total_loss = total_loss + self.simclr_alpha * loss_simclr
            return total_loss

        return logits


@TRAINER_REGISTRY.register()
class IVLP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.IVLP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.IVLP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP (IVLP) ...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        ###################################
        # (추가) Knowledge Distillation 옵션
        ###################################
        self.use_kd = getattr(cfg.TRAINER.IVLP, "USE_KD", False)
        if self.use_kd:
            print(">> [KD] USE_KD=True => Load teacher model for distillation!")

            teacher_name = getattr(cfg.TRAINER.IVLP, "KD_TEACHER_MODEL", "resnet50")
            print(f"   Teacher Model: {teacher_name}")

            # 1) 먼저 user가 지정한 KD_NUM_CLASSES가 있는지 확인
            kd_num_classes = getattr(cfg.TRAINER.IVLP, "KD_NUM_CLASSES", None)
            if kd_num_classes is not None:
                # 사용자가 직접 KD_NUM_CLASSES를 지정한 경우
                print(f"   [KD] Using user-specified KD_NUM_CLASSES = {kd_num_classes}")
                num_classes_for_teacher = kd_num_classes
            else:
                # 없으면 dataset에서 자동 추론
                # self.dm.dataset.num_classes 또는 self.dm.num_classes (둘 중 편한 것 사용)
                num_classes_for_teacher = self.dm.num_classes  
                print(f"   [KD] No KD_NUM_CLASSES specified. Using dataset.num_classes = {num_classes_for_teacher}")

            self.teacher = timm.create_model(
                teacher_name, 
                pretrained=True, 
                num_classes=num_classes_for_teacher
            )
            self.teacher.eval().to(self.device)

            self.kd_alpha = getattr(cfg.TRAINER.IVLP, "KD_ALPHA", 0.5)
            self.kd_T = getattr(cfg.TRAINER.IVLP, "KD_T", 4.0)
        else:
            self.teacher = None


        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def parse_batch_train(self, batch):
        """
        - SimCLR 배치 => {"img1","img2","label"}
        - Mixup 배치 => {"img","y_a","y_b","lam"} (그리고 label=None)
        - 일반 배치 => {"img","label"}
        """
        # (1) Mixup 또는 일반 배치
        if "y_a" in batch and "y_b" in batch and "lam" in batch:
            # Mixup 모드
            x = batch["img"].to(self.device)
            y_a = batch["y_a"].to(self.device)
            y_b = batch["y_b"].to(self.device)
            lam = batch["lam"]
            return x, None, None, (y_a, y_b, lam)  # label=None
        # (2) SimCLR 배치
        elif "img1" in batch and "img2" in batch:
            x1 = batch["img1"].to(self.device)
            x2 = batch["img2"].to(self.device)
            label = batch["label"].to(self.device)
            return x1, label, x2, None
        # (3) 일반 CE/Focal
        else:
            x = batch["img"].to(self.device)
            label = batch["label"].to(self.device)
            return x, label, None, None

    def forward_backward(self, batch):
        """
        forward + backward
        """
        image1, label, image2, mixup_args = self.parse_batch_train(batch)
        # mixup_args: (y_a, y_b, lam) 또는 None

        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.IVLP.PREC

        if self.use_kd:
            # KD = alpha * HardLoss + (1-alpha)*KDLoss
            # => hard_loss는 mixup/CE/focal 중 하나
            # => teacher forward => teacher_logits
            with torch.no_grad():
                # teacher가 mixup된 이미지를 입력받아 logits 추론
                teacher_logits = self.teacher(image1)  # [bs, #class]
            # student forward
            student_logits = None
            if isinstance(model, nn.DataParallel):
                # DataParallel이면 model.module.image_encoder(...) 등 따로 처리 가능
                # 여기서는 CustomCLIP의 forward를 간접 호출해야 하므로 아래처럼:
                with autocast(enabled=(prec=="amp")):
                    # text prompts
                    prompts = model.module.prompt_learner()
                    text_features = model.module.text_encoder(prompts, model.module.tokenized_prompts)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    image_features = model.module.image_encoder(image1.type(model.module.dtype))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logit_scale = model.module.logit_scale.exp()
                    student_logits = logit_scale * (image_features @ text_features.t())
            else:
                with autocast(enabled=(prec=="amp")):
                    # CustomCLIP에서는 forward를 직접 부르면 SimCLR loss 계산도 해버리므로,
                    # 그냥 내부 로직을 비슷하게 가져온다:
                    prompts = model.prompt_learner()
                    text_features = model.text_encoder(prompts, model.tokenized_prompts)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    image_features = model.image_encoder(image1.type(model.dtype))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logit_scale = model.logit_scale.exp()
                    student_logits = logit_scale * image_features @ text_features.t()

            if prec == "amp":
                optim.zero_grad()
                with autocast():
                    # mixup인 경우 hard_loss 계산 다르게, 일반 라벨인 경우 label로 CE
                    if mixup_args is not None:
                        y_a, y_b, lam = mixup_args
                        loss_kd = distillation_loss(
                            student_logits, 
                            teacher_logits, 
                            label,   # label은 None이지만, mixup에서는 쓰지 않음
                            model.criterion_ce,
                            T=self.kd_T,
                            alpha=self.kd_alpha,
                            lam=lam,
                            y_a=y_a,
                            y_b=y_b
                        )
                    else:
                        # 일반 KD
                        loss_kd = distillation_loss(
                            student_logits,
                            teacher_logits,
                            label,
                            model.criterion_ce,
                            T=self.kd_T,
                            alpha=self.kd_alpha
                        )
                scaler.scale(loss_kd).backward()
                scaler.step(optim)
                scaler.update()
                loss = loss_kd
            else:
                optim.zero_grad()
                if mixup_args is not None:
                    y_a, y_b, lam = mixup_args
                    loss_kd = distillation_loss(
                        student_logits, 
                        teacher_logits, 
                        label, 
                        model.criterion_ce,
                        T=self.kd_T,
                        alpha=self.kd_alpha,
                        lam=lam,
                        y_a=y_a,
                        y_b=y_b
                    )
                else:
                    loss_kd = distillation_loss(
                        student_logits,
                        teacher_logits,
                        label,
                        model.criterion_ce,
                        T=self.kd_T,
                        alpha=self.kd_alpha
                    )
                loss_kd.backward()
                optim.step()
                loss = loss_kd

        else:
            # 기존 IVLP 로직 (SimCLR or CE/Focal)
            if prec == "amp":
                with autocast():
                    loss = model(image1, label, image2)
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss = model(image1, label, image2)
                optim.zero_grad()
                loss.backward()
                optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = f"model.pth.tar-{epoch}"

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            e = checkpoint["epoch"]

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {e})')
            self._models[name].load_state_dict(state_dict, strict=False)
