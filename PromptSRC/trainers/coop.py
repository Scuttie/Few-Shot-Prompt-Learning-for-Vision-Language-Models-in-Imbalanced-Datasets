import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import transforms
from PIL import ImageFilter
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

############################################################
# 1) GaussianBlur & SimCLR Data Aug
############################################################

class GaussianBlur(object):
    """SimCLR 논문에서 제안된 Gaussian Blur transform"""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(sigma=[0.1, 2.0]),
    transforms.ToTensor(),
])

class SimCLRDataset(torch.utils.data.Dataset):
    """
    기존 dataset에서 (img, label)을 가져와,
    SimCLR을 위해 2개의 서로 다른 augmentation 결과 [img1, img2]를 반환.
    """
    def __init__(self, base_dataset, transform):
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]  # (PIL image, int label)
        img1 = self.transform(img)
        img2 = self.transform(img)
        return [img1, img2], label

############################################################
# 2) NT-Xent (SimCLR) Loss for Logits
############################################################

_tokenizer = _Tokenizer()

class LogitsNTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits1, logits2):
        # logits1, logits2: shape (N, K)
        z1 = F.normalize(logits1, dim=1)  # (N, K)
        z2 = F.normalize(logits2, dim=1)  # (N, K)

        # (2N, K)
        z = torch.cat([z1, z2], dim=0)
        N2 = z.shape[0]
        # 여기서 N2는 2N, 즉 original batch size의 2배
        # N은 batch size
        N = N2 // 2

        # (2N, 2N)
        sim = torch.matmul(z, z.t()) / self.temperature

        # 대각선 제거
        mask = ~torch.eye(N2, dtype=bool, device=z.device)
        sim_masked = sim[mask].view(N2, -1)  # (2N, 2N-1)

        # row별 positive 인덱스 계산
        row_idx = torch.arange(N2, device=z.device)  # (2N,)
        pos_idx = torch.arange(N2, device=z.device)  # (2N,)

        # 앞쪽 N개(0..N-1)는 pos_idx += N
        # 뒤쪽 N개(N..2N-1)는 pos_idx -= N
        pos_idx[:N] += N
        pos_idx[N:] -= N

        # row별 positive 값 (2N,)
        pos_vals = sim[row_idx, pos_idx]

        # negative 값 (2N, 2N-2)
        full_c = torch.arange(N2, device=z.device).unsqueeze(0)  # shape (1,2N)
        row_c = row_idx.unsqueeze(1)  # shape (2N,1)
        pos_c = pos_idx.unsqueeze(1)  # shape (2N,1)
        row_mask = (full_c != row_c) & (full_c != pos_c)  # shape (2N,2N)

        neg_list = []
        for i in range(N2):
            row_neg = sim[i][row_mask[i]]
            neg_list.append(row_neg.unsqueeze(0))
        neg_vals = torch.cat(neg_list, dim=0)  # (2N, 2N-2)

        # 양성 1개 + 음성 2N-2개 => (2N, 2N-1)
        pos_vals = pos_vals.unsqueeze(1)     # (2N,1)
        out = torch.cat([pos_vals, neg_vals], dim=1)  # (2N, 2N-1)

        # 전부 label=0
        labels = torch.zeros(N2, dtype=torch.long, device=z.device)

        loss = self.ce(out, labels)
        return loss



############################################################
# 3) Prompt Learner, etc.
############################################################

class MultiClassFocalLoss(nn.Module):
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
        "trainer": 'CoOp',
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

        # eot_token(문장 끝) 임베딩만 추출
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = x @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize({cfg_imsize}) != clip_imsize({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            all_prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i+1, :, :]
                class_i = suffix[i : i+1, :name_len, :]
                suffix_i = suffix[i : i+1, name_len:, :]
                ctx_i_half1 = ctx[i : i+1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i+1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                all_prompts.append(prompt)
            prompts = torch.cat(all_prompts, dim=0)
        elif self.class_token_position == "front":
            all_prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i+1, :, :]
                class_i = suffix[i : i+1, :name_len, :]
                suffix_i = suffix[i : i+1, name_len:, :]
                ctx_i = ctx[i : i+1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                all_prompts.append(prompt)
            prompts = torch.cat(all_prompts, dim=0)
        else:
            raise ValueError("Unknown class_token_position")

        return prompts

############################################################
# 4) CustomCLIP (SimCLR=logit-based)
############################################################

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # parse loss_type
        self.loss_type = cfg.TRAINER.COOP.get("LOSS_TYPE", "ce")

        if self.loss_type == "simclr":  
            # 원래 NTXentLoss( (N,D)->(2N,2N) )를 쓰셨는데
            # 여기서는 logit 기반 => LogitsNTXentLoss
            print(">> Using LogitsNTXentLoss (logit-based simclr)!")
            self.criterion_simclr = LogitsNTXentLoss(temperature=0.07)

        elif self.loss_type == "ce":
            print(">> Using CE Loss!")
            self.criterion_ce = nn.CrossEntropyLoss()

        elif self.loss_type == "focal":
            print(">> Use Focal Loss!")
            # focal loss에 필요한 alpha 계산
            # config에서 per_class 샘플 수 목록을 가져옴 (예: cfg.DATASET.PER_CLASS_SHOTS)
            per_class = cfg.DATASET.PER_CLASS_SHOTS
            alpha = None

            if per_class is not None and len(per_class) > 0:
                # 혹시 문자열 형태라면 파싱
                if isinstance(per_class, str):
                    per_class = list(map(int, per_class.strip("[]").split(",")))

                total_samples = sum(per_class)
                n_cls = len(classnames)
                # (예시) alpha = total_samples / (n_cls * count)
                alpha = [
                    total_samples / (n_cls * count) if count > 0 else 0.0
                    for count in per_class
                ]

            self.criterion_ce = MultiClassFocalLoss(alpha=alpha, gamma=2, reduction='mean')

        else:
            raise ValueError(f"Unknown loss_type = {self.loss_type}")

    def forward_once(self, image):
        """
        한 번의 forward -> logit: (N, K) shape
        """
        image_feat = self.image_encoder(image.type(self.dtype))
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()
        text_feat = self.text_encoder(prompts, self.tokenized_prompts)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feat @ text_feat.t()   # (N, K)

        return logits

    def forward(self, img1, lbl=None, img2=None):
        """
        - simclr 모드 -> img2까지 들어옴
        - ce/focal 모드 -> img2=None, lbl 있음
        """
        if self.loss_type == "simclr":
            # 1) logit1, logit2
            logits1 = self.forward_once(img1)  # (N,K)
            logits2 = self.forward_once(img2)  # (N,K)

            # 2) SimCLR loss
            loss = self.criterion_simclr(logits1, logits2)
            return loss

        elif self.loss_type in ["ce", "focal"]:
            # CE/Focal은 모두 '하나의 이미지만' 받아서 분류
            logits = self.forward_once(img1)  # (N,K)
            if self.training and lbl is not None:
                return self.criterion_ce(logits, lbl)
            else:
                return logits

        else:
            raise ValueError(f"Unsupported loss_type? {self.loss_type}")

############################################################
# 5) Trainer
############################################################

@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """
    - freeze image/text encoder
    - unfreeze prompt
    - simclr -> logit-based NT-Xent
    """
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        clip_model = load_clip_to_cpu(cfg)

        # precision
        if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP w. logit-simclr or CE")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in image+text encoder => only prompt_learner can update")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad = False

        # optimizer for prompt_learner
        from dassl.utils import load_pretrained_weights
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        """
        batch:
         - if simclr => { "img1", "img2", "label" } => label ignored
         - else => { "img", "label" }
        """
        x1, lbl, x2 = self.parse_batch_train(batch)
        model = self.model
        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            with autocast():
                loss = model(x1, lbl, x2)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss = model(x1, lbl, x2)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        loss_val = loss.item()
        loss_summary = {"loss": loss_val}

        # if CE => compute acc
        if (lbl is not None) and (x2 is None) and (self.model.loss_type=="ce"):
            with torch.no_grad():
                logits_eval = model(x1, lbl=None, img2=None)
                acc = compute_accuracy(logits_eval, lbl)[0].item()
            loss_summary["acc"] = acc

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        loss_type = self.cfg.TRAINER.COOP.LOSS_TYPE
        if loss_type == "simclr":
            x1 = batch["img1"].to(self.device)
            x2 = batch["img2"].to(self.device)
            lbl = None
            return x1, lbl, x2
        else:
            x = batch["img"].to(self.device)
            y = batch["label"].to(self.device)
            return x, y, None

    def load_model(self, directory, epoch=None):
        if not directory:
            print("no pretrained => skip")
            return
        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch:
            model_file = f"model.pth.tar-{epoch}"

        from dassl.utils import load_checkpoint
        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")
            ckpt = load_checkpoint(model_path)
            state_dict = ckpt["state_dict"]
            ep = ckpt["epoch"]
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            print(f'Loading {name} from "{model_path}" (epoch={ep})')
            self._models[name].load_state_dict(state_dict, strict=False)
