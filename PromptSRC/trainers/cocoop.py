import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
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

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
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
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        # meta_net: 이미지 특징(im_features)에 기반해서 context token을 조금씩 shift
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # 아래 prefix, suffix는 고정 (CLIP 소스 토큰)
        self.register_buffer("token_prefix", embedding[:, :1, :])    # (n_cls, 1, ctx_dim)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # (n_cls, *, ctx_dim)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, shape (dim0, n_ctx, ctx_dim)
        # prefix: shape (n_cls, 1, ctx_dim)
        # suffix: shape (n_cls, *, ctx_dim)
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,   # (dim0, 1, dim)
                ctx,      # (dim0, n_ctx, dim)
                suffix,   # (dim0, *, dim)
            ],
            dim=1,
        )
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix    # (n_cls, 1, ctx_dim)
        suffix = self.token_suffix    # (n_cls, *, ctx_dim)
        ctx = self.ctx                # (n_ctx, ctx_dim)

        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)          # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)            # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias          # (batch, n_ctx, ctx_dim)

        # 한 번의 forward에서 batch만큼의 이미지가 들어오므로
        # 각 이미지별로 instance-conditioned prompt를 구성
        prompts = []
        for ctx_shifted_i in ctx_shifted:  # (n_ctx, ctx_dim)
            # 이 context는 n_cls 모두 공유하지만, batch dimension마다 달라짐
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)  # (n_cls, n_ctx, ctx_dim)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)          # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        # prompts: list of length batch, each item shape (n_cls, n_tkn, ctx_dim)
        prompts = torch.stack(prompts)  # (batch, n_cls, n_tkn, ctx_dim)
        return prompts


class MultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss (다중 클래스용).
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Args:
            alpha (Tensor or list, optional): 클래스별 가중치 보정을 위한 alpha.
            gamma (float): Focal Loss의 focusing parameter.
            reduction (str): 'none' | 'mean' | 'sum'.
        """
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
        """
        Args:
            inputs: 모델 출력 logits (batch, num_classes)
            targets: 실제 레이블 (batch)
        Returns:
            Focal Loss 값
        """
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha = self.alpha[targets]  # 각 샘플의 클래스에 해당하는 alpha
        else:
            alpha = 1.0

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 모델이 정답을 맞출 확률
        focal_loss = alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, criterion=None):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # CoCoOp에서 사용할 Loss 함수 (기본: Cross Entropy -> 여기서는 Focal Loss)
        self.criterion = criterion

    def forward(self, image, label=None):
        """
        image: (batch, C, H, W)
        label: (batch)
        """
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # 1. 이미지 특징 추출
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 2. instance-conditioned prompts 생성
        prompts = self.prompt_learner(image_features)  # (batch, n_cls, n_tkn, ctx_dim)

        # 3. 각 이미지별 텍스트 특징 추출 및 로짓 계산
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()  # (n_cls)
            logits.append(l_i)
        # logits = (batch, n_cls)
        logits = torch.stack(logits)

        # 4. 훈련 시 Loss 계산 (Focal Loss)
        if self.prompt_learner.training:
            return self.criterion(logits, label)

        # 추론 시 logits 반환
        return logits


@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")

        # -------------
        # Focal Loss 설정
        # -------------
        # 필요한 경우 alpha에 클래스별 비율을 넣을 수 있음
        focal_loss = MultiClassFocalLoss(alpha=None, gamma=2, reduction='mean')

        # CustomCLIP을 초기화하면서 focal_loss를 전달
        self.model = CustomCLIP(cfg, classnames, clip_model, criterion=focal_loss)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            # 프롬프트 학습 파트에만 초기 가중치가 들어가는 경우
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # 멀티 GPU 사용 설정
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

        # Epoch 마지막 배치 시 learning rate 업데이트
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} from \"{}\" (epoch = {})".format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
