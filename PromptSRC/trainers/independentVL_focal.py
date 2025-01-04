import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

####################################
# (주석) MultiClassFocalLoss 추가  #
####################################
class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        다중 클래스 Focal Loss.

        Args:
            alpha (Tensor or list, optional): 각 클래스 가중치(불균형 보정용).
            gamma (float): Focusing parameter.
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
        inputs: (batch_size, num_classes) - 모델 출력(logits)
        targets: (batch_size) - 정답 레이블 (0 ~ num_classes-1)
        """
        # 각 샘플마다 alpha 가져오기
        if self.alpha is not None:
            # 입력 device와 alpha device 일치
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha = self.alpha[targets]
        else:
            alpha = 1.0

        # 기본 CE loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = 예측 확률
        focal_loss = alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def load_clip_to_cpu(cfg):
    """CLIP 모델을 CPU 로드 후, IVLP 세팅에 맞춰 build_model() 수행."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        # 일반적인 state dict
        state_dict = torch.load(model_path, map_location="cpu")

    # IVLP용 design_details
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
    """CLIP의 text 인코더 모듈 (prompt-learner가 만든 text prompt를 인코딩)."""
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

        # eot_token(문장 끝) 임베딩만 추출
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = x @ self.text_projection
        return x


class VLPromptLearner(nn.Module):
    """Independent VLP의 Text Prompt Learner (Vision prompt 학습도 가능)."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)

        # Language prompt depth >= 1
        assert cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT >= 1, (
            "IVLP에서 language prompt depth는 최소 1 이상이어야 합니다."
        )

        n_ctx = cfg.TRAINER.IVLP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Prompt init
        if ctx_init and (n_ctx <= 4):
            # 지정된 문자열로 초기화
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # 랜덤 초기화
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.IVLP.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        # 클래스 이름 처리
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # tokenize
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # prefix/suffix (이 둘은 그대로 고정)
        self.register_buffer("token_prefix", embedding[:, :1, :])   # [SOS]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # [CLS], [EOS] 등

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # label이 있으면 해당 라벨에 해당하는 prefix/suffix만 선택
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        # (N, 1 + n_ctx + (나머지), D)
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        """학습 중이면 n_cls개 만큼 prompt를 만들어냄."""
        ctx = self.ctx
        if ctx.dim() == 2:
            # (n_ctx, ctx_dim) -> (n_cls, n_ctx, ctx_dim)
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts


class CustomCLIP(nn.Module):
    """IVLP에서의 CLIP + PromptLearner 결합 모델 (MultiClassFocalLoss 적용)."""
    def __init__(self, cfg, classnames, clip_model, focal_loss=None):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        ###############################################
        # (주석) Trainer에서 전달받은 focal_loss 적용 #
        ###############################################
        self.focal_loss = focal_loss

    def forward(self, image, label=None):
        # Text features
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Image features
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Similarity -> logit
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if self.training and label is not None:
            # (기존: loss = F.cross_entropy(logits, label))
            # (수정: focal loss 적용)
            loss = self.focal_loss(logits, label)
            return loss
        else:
            # 평가(테스트) 시에는 logits 반환
            return logits


@TRAINER_REGISTRY.register()
class IVLP(TrainerX):
    """IVLP Trainer (few-shot 세팅이 config나 스크립트로 주어져도 그대로 학습 가능, focal loss 적용)."""

    def check_cfg(self, cfg):
        # precision 설정
        assert cfg.TRAINER.IVLP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames  # dassl의 DataManager가 이미 few-shot 로드를 처리

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # FP32나 AMP이면 CLIP을 float으로 변환
        if cfg.TRAINER.IVLP.PREC in ["fp32", "amp"]:
            clip_model.float()

        ########################################
        # (주석) Focal Loss 객체 초기화 예시   #
        ########################################
        # alpha: 클래스별 가중치, 필요 없다면 None
        # gamma: focusing parameter (기본 2)
        # reduction: 'mean' or 'sum'
        alpha = None
        gamma = 2
        reduction = 'mean'
        focal_loss_fn = MultiClassFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

        print("Building custom CLIP (IVLP) with Focal Loss ...")
        self.model = CustomCLIP(
            cfg, 
            classnames, 
            clip_model, 
            focal_loss=focal_loss_fn  # focal loss 주입
        )

        print("Turning off gradients in both the image and the text encoder")
        # 오직 prompt_learner만 업데이트
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # 어느 파라미터가 업데이트되는지 출력
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        # pretrained weights 로드
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # optimizer: (prompt_learner 파라미터만)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None

        # 멀티 GPU
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        """한 번의 forward-backward 수행."""
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.IVLP.PREC
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

        # 한 epoch의 모든 batch가 끝나면 learning rate 업데이트
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        """데이터 로더에서 받은 배치를 device에 올린 뒤 반환."""
        input_ = batch["img"].to(self.device)
        label_ = batch["label"].to(self.device)
        return input_, label_

    def load_model(self, directory, epoch=None):
        """모델 로드. few-shot에서는 필요 시 base 모델(혹은 best) 로드 가능."""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            e = checkpoint["epoch"]

            # token_prefix/suffix는 로드 시 무시(클래스명이 달라질 수 있으므로)
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print(f"Loading weights to {name} from \"{model_path}\" (epoch = {e})")
            self._models[name].load_state_dict(state_dict, strict=False)
