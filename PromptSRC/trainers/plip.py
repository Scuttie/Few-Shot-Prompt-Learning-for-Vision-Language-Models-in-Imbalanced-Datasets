import os.path as osp

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

# torch.backends.cuda.enable_flash_sdp(False)  # Flash Attention 비활성화
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)  # 일반적인 dot-product attention 사용

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
        "trainer": 'IVLP',
        "vision_depth": cfg.TRAINER.PLIP.PROMPT_DEPTH_VISION,
        "language_depth": cfg.TRAINER.PLIP.PROMPT_DEPTH_TEXT,
        "vision_ctx": cfg.TRAINER.PLIP.N_CTX_VISION,
        "language_ctx": cfg.TRAINER.PLIP.N_CTX_TEXT
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
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PLIP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PLIP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        if cfg.TRAINER.PLIP.REG_TYPE == "svd":
            """Compute SVD and set U, S, V with proper requires_grad settings."""
            ctx = self.ctx.float()
            U, S, Vh = torch.linalg.svd(ctx, full_matrices=False)
            # Set requires_grad=False for U and Vh (fixed), True for S (to be optimized)
            self.U = U.detach().type(dtype).cuda()  # Detach to prevent gradients
            self.S = nn.Parameter(S.type(dtype).cuda())  # Make S a learnable parameter
            self.Vh = Vh.detach().type(dtype).cuda() # Detach Vh to prevent gradients


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.REG_TYPE = cfg.TRAINER.PLIP.REG_TYPE

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def reconstruct_ctx(self):
        """Reconstruct ctx from the updated singular values."""
        return self.U @ torch.diag_embed(self.S) @ self.Vh

    def forward(self):

        if self.REG_TYPE == "svd":
            ctx = self.reconstruct_ctx()
        else:
            ctx = self.ctx

        if ctx.dim() == 2:
            # (n_ctx, ctx_dim) -> (n_cls, n_ctx, ctx_dim)
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class PLIP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PLIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.PLIP.PREC == "fp32" or cfg.TRAINER.PLIP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if cfg.TRAINER.PLIP.REG_TYPE == "svd":
                    if "S" in name:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)



        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PLIP.PREC == "amp" else None

        # Multi-GPU
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        # Check for Smoothness constraints
        print("K:", cfg.TRAINER.PLIP.K)
        print("REG_COEFF:", cfg.TRAINER.PLIP.REG_COEFF)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.PLIP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                # 기존 cross entropy 대신 focal loss 적용
                loss = self.criterion(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)

            loss = F.cross_entropy(output, label)

            if self.cfg.TRAINER.PLIP.REG_TYPE == "grad":
                gradients = torch.autograd.grad(
                            outputs=loss,
                            inputs=self.model.prompt_learner.ctx,
                            grad_outputs=torch.ones_like(loss),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True
                )[0]
                gradient_norm = gradients.norm(2, dim=1)  # Compute L2 norm
                penalty = torch.mean((gradient_norm - self.cfg.TRAINER.PLIP.K) ** 2)  # Penalize deviations from K
            elif self.cfg.TRAINER.PLIP.REG_TYPE == "svd":
                penalty = 0.0
            elif self.cfg.TRAINER.PLIP.REG_TYPE == "spectral_norm":
                penalty = self.power_iteration(self.model.prompt_learner.ctx)
            else:
                raise ValueError
            
            loss = loss + self.cfg.TRAINER.PLIP.REG_COEFF * penalty
            self.model_backward_and_update(loss)

        loss_summary = {
            "penalty" : penalty.item() if isinstance(penalty, torch.Tensor) else penalty,
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def power_iteration(self, W, num_iters=5):
        """Approximate the spectral norm using power iteration."""
        v = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)  # Random vector
        v = v / v.norm()

        for _ in range(num_iters):
            v = torch.mv(W.T @ W, v)
            v = v / v.norm()

        return torch.norm(W @ v)

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
