import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    """
    CoCoOp용 CLIP을 CPU로 로드 후, build_model().
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": 'CoCoOp',
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0
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
        x = x.permute(1, 0, 2)  # (N, L, D) -> (L, N, D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (L, N, D) -> (N, L, D)
        x = self.ln_final(x).type(self.dtype)

        # eot_token 임베딩만 추출
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = x @ self.text_projection
        return x

class MultiClassFocalLoss(nn.Module):
    """
    다중 클래스 Focal Loss
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
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

class PromptLearner(nn.Module):
    """
    CoCoOp에서, 이미지 feature에 따라 context token을 조정하는 learner
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        assert cfg_imsize == clip_imsize, f"cfg_imsize({cfg_imsize}) != clip_imsize({clip_imsize})"

        # (1) context token 초기화
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        # (2) meta_net: 이미지 feature에 기반하여 context token을 살짝 shift
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        # fp16인 경우 meta_net에도 half()
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        # 클래스 이름 관련
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # prefix/suffix는 클래스별로 고정된 부분
        self.register_buffer("token_prefix", embedding[:, :1, :])    # (n_cls, 1, dim)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # (n_cls, *, dim)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        # (dim0, 1, dim) + (dim0, n_ctx, dim) + (dim0, *, dim)
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, im_features):
        """
        im_features: (batch, vis_dim)
        """
        prefix = self.token_prefix       # (n_cls, 1, dim)
        suffix = self.token_suffix       # (n_cls, *, dim)
        ctx = self.ctx                   # (n_ctx, dim)

        # meta_net: (batch, dim)
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)         # (batch, 1, dim)
        ctx = ctx.unsqueeze(0)          # (1, n_ctx, dim)
        ctx_shifted = ctx + bias        # (batch, n_ctx, dim)

        # 각 이미지별 instance-conditioned prompt
        prompts = []
        for ctx_i in ctx_shifted:  # (n_ctx, dim)
            # (n_cls, n_ctx, dim)
            ctx_i_expanded = ctx_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            # (n_cls, n_tkn, dim)
            prompt_i = self.construct_prompts(ctx_i_expanded, prefix, suffix)
            prompts.append(prompt_i)

        # (batch, n_cls, n_tkn, dim)
        prompts = torch.stack(prompts, dim=0)
        return prompts

class CustomCLIP(nn.Module):
    """
    CoCoOp 모델 + Focal Loss 옵션화
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Focal Loss 사용 여부
        self.use_focal_loss = cfg.TRAINER.COCOOP.get("USE_FOCAL_LOSS", False)
        print(f">> USE_FOCAL_LOSS = {self.use_focal_loss}")

        # per_class 샘플 정보로 alpha 계산
        per_class = getattr(cfg.DATASET, "PER_CLASS_SHOTS", None)
        n_cls = len(classnames)

        if self.use_focal_loss:
            print(">> Use Focal Loss!")
            alpha = None
            if per_class is not None and isinstance(per_class, (list, str)):
                if isinstance(per_class, str):
                    per_class = list(map(int, per_class.strip("[]").split(",")))
                total_samples = sum(per_class)
                alpha = [total_samples / (n_cls * cnt) for cnt in per_class]
            self.criterion = MultiClassFocalLoss(alpha=alpha, gamma=2, reduction='mean')
        else:
            print(">> Use Cross Entropy Loss!")
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, image, label=None):
        # 1) 이미지 특징
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 2) instance-conditioned prompts
        prompts = self.prompt_learner(image_features)  # (batch, n_cls, n_tkn, dim)

        # 3) 로짓 계산
        logit_scale = self.logit_scale.exp()
        logits_list = []

        for ctx_prompts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(ctx_prompts_i, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_i = logit_scale * imf_i @ text_features.t()  # (n_cls)
            logits_list.append(logits_i)

        # (batch, n_cls)
        logits = torch.stack(logits_list, dim=0)

        # 4) 학습 시 loss 계산
        if self.training and label is not None:
            return self.criterion(logits, label)
        else:
            return logits

@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    """
    CoCoOp Trainer 통합
    -> Focal Loss on/off 옵션 (TRAINER.COCOOP.USE_FOCAL_LOSS)
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        prompt_module = "prompt_learner"
        for name, param in self.model.named_parameters():
            if prompt_module not in name:
                param.requires_grad_(False)

        # 체크
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.COCOOP.PREC

        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        x = batch["img"].to(self.device)
        y = batch["label"].to(self.device)
        return x, y

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

            # token_prefix, token_suffix는 고정된 토큰
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {e})')
            self._models[name].load_state_dict(state_dict, strict=False)
