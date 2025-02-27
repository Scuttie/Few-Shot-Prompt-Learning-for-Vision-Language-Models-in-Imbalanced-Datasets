import numpy as np
import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from .imagenet_templates import IMAGENET_TEMPLATES

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils.layers import LoRALayer, PlainMultiheadAttentionLoRA
from typing import Dict

_tokenizer = _Tokenizer()

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}



def apply_lora(cfg, clip_model):
    list_lora_layers = []
    if cfg.TRAINER.LORA.ENCODER == 'text' or cfg.TRAINER.LORA.ENCODER == 'both':
        indices = INDEX_POSITIONS_TEXT[cfg.TRAINER.LORA.POSITION]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        # 기존 모듈과 같은 dtype으로 맞추기
                        dtype = submodule.in_proj_weight.dtype
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=cfg.TRAINER.LORA.PARAMS, 
                            r=cfg.TRAINER.LORA.R, 
                            lora_alpha=cfg.TRAINER.LORA.ALPHA, 
                            dropout_rate=cfg.TRAINER.LORA.DROPOUT_RATE).to(dtype)  # 데이터 타입 일치
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if cfg.TRAINER.LORA.ENCODER == 'vision' or cfg.TRAINER.LORA.ENCODER == 'both':
        backbone_name = cfg.MODEL.BACKBONE.NAME
        indices = INDEX_POSITIONS_VISION[backbone_name][cfg.TRAINER.LORA.POSITION]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        # 기존 모듈과 같은 dtype으로 맞추기
                        dtype = submodule.in_proj_weight.dtype
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=cfg.TRAINER.LORA.PARAMS, 
                            r=cfg.TRAINER.LORA.R, 
                            lora_alpha=cfg.TRAINER.LORA.ALPHA, 
                            dropout_rate=cfg.TRAINER.LORA.DROPOUT_RATE).to(dtype)  # 데이터 타입 일치
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def load_clip_to_cpu(cfg, zero_shot_model=False, use_lora=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if not zero_shot_model:
        design_details = {
            "trainer": 'IVLP',
            "vision_depth": cfg.TRAINER.LORA.PROMPT_DEPTH_VISION,
            "language_depth": cfg.TRAINER.LORA.PROMPT_DEPTH_TEXT,
            "vision_ctx": cfg.TRAINER.LORA.N_CTX_VISION,
            "language_ctx": cfg.TRAINER.LORA.N_CTX_TEXT
        }
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)

    list_lora_layers = None
    if use_lora:
        list_lora_layers = apply_lora(cfg, model)
        return model, list_lora_layers

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
        n_ctx = cfg.TRAINER.LORA.N_CTX_TEXT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        prompt_prefix = cfg.TRAINER.LORA.CTX_INIT
        print(f'Used context: "{prompt_prefix}"')
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # clip_model_temp = load_clip_to_cpu(cfg, True).type(dtype)
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).cuda()
            self.ZS_image_encoder = clip_model_temp_image.visual
        self.fixed_embeddings = embedding.clone().detach().mean(dim=1)
    
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.embedding = embedding

    def forward(self):
        return self.embedding


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

        if self.prompt_learner.training:
            fixed_embeddings = self.prompt_learner.fixed_embeddings
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            # print("fixed_embeddings:", fixed_embeddings.shape)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.cuda().t()

            return (
                logits,
                text_features,
                fixed_embeddings,
                zero_shot_features,
                image_features,
                zero_shot_logits
            )


        return logits


@TRAINER_REGISTRY.register()
class LoRA(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.LORA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, list_lora_layers = load_clip_to_cpu(cfg, zero_shot_model=True, use_lora=True)        

        if cfg.TRAINER.LORA.PREC == "fp32" or cfg.TRAINER.LORA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        mark_only_lora_as_trainable(clip_model)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")

        self.model.to(self.device)
        # NOTE: only give LoRA to the optimizer
        self.optim = build_optimizer(get_lora_parameters(self.model), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM) # eta_min=1e-6 is not applied
        self.lora_weights = self.make_weight(list_lora_layers)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LORA.PREC == "amp" else None

        # Multi-GPU
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        # output = self.model(image)
        output, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, zero_shot_logits = self.model(image)

        loss = F.cross_entropy(output, label)

        # Calculate the L_SCL_text loss
        if self.cfg.TRAINER.LORA.TEXT_LOSS_WEIGHT > 0:
            loss_scl_text = F.l1_loss(
                normalized_text_features, zs_clip_text_embeddings.cuda(), reduction='mean'
            ) * self.cfg.TRAINER.LORA.TEXT_LOSS_WEIGHT
            loss += loss_scl_text

        # Calculate the L_SCL_image loss
        if self.cfg.TRAINER.LORA.IMAGE_LOSS_WEIGHT > 0:        
            loss_scl_image = F.l1_loss(
                image_ft, zs_image_embedd.cuda(), reduction='mean'
            ) * self.cfg.TRAINER.LORA.IMAGE_LOSS_WEIGHT
            loss += loss_scl_image

        # Now calculate L_SCL_logits
        if self.cfg.TRAINER.LORA.LOGITS_LOSS_WEIGHT > 0:
            L_SCL_logits = F.kl_div(
                F.log_softmax(output / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / output.numel()
            loss += L_SCL_logits * self.cfg.TRAINER.LORA.LOGITS_LOSS_WEIGHT

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

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

        # to manage names like ViT-B/16
        name = "lora"
        backbone = self.cfg.MODEL.BACKBONE.NAME
        backbone = backbone.replace('/', '').replace('-', '').lower()  
        save_dir = f'{directory}/{self.cfg.DATASET.NAME}/{backbone}/{name}'
        filename="best"
        load_path = f'{save_dir}/{filename}.pt'

        if not os.path.exists(load_path):
            raise FileNotFoundError(f'File {load_path} does not exist.')

        loaded_data = torch.load(load_path)

        metadata = loaded_data['metadata']
        if metadata['r'] != self.cfg.TRAINER.LORA.R:
            raise ValueError(
                f"r mismatch: expected {self.cfg.TRAINER.LORA.R}, found {metadata['r']}")
        if metadata['alpha'] != self.cfg.TRAINER.LORA.ALPHA:
            raise ValueError(
                f"alpha mismatch: expected {self.cfg.TRAINER.LORA.ALPHA}, found {metadata['alpha']}")
        if metadata['encoder'] != self.cfg.TRAINER.LORA.ENCODER:
            raise ValueError(
                f"Encoder mismatch: expected {self.cfg.TRAINER.LORA.ENCODER}, found {metadata['encoder']}")
        if metadata['params'] != self.cfg.TRAINER.LORA.PARAMS:
            raise ValueError(
                f"Params mismatch: expected {self.cfg.TRAINER.LORA.PARAMS}, found {metadata['params']}")
        if metadata['position'] != self.cfg.TRAINER.LORA.POSITION:
            raise ValueError(
                f"Position mismatch: expected {self.cfg.TRAINER.LORA.POSITION}, found {metadata['position']}")

        # Convert weights to FP16
        weights = loaded_data['weights']
        for i, layer in enumerate(self.lora_weights):
            layer_weights = weights[f'layer_{i}']
            if 'q' in self.cfg.TRAINER.LORA.PARAMS and 'q_proj' in layer_weights:
                layer.q_proj.w_lora_A.data.copy_(
                    layer_weights['q_proj']['w_lora_A'].half())
                layer.q_proj.w_lora_B.data.copy_(
                    layer_weights['q_proj']['w_lora_B'].half())
            if 'k' in self.cfg.TRAINER.LORA.PARAMS and 'k_proj' in layer_weights:
                layer.k_proj.w_lora_A.data.copy_(
                    layer_weights['k_proj']['w_lora_A'].half())
                layer.k_proj.w_lora_B.data.copy_(
                    layer_weights['k_proj']['w_lora_B'].half())
            if 'v' in self.cfg.TRAINER.LORA.PARAMS and 'v_proj' in layer_weights:
                layer.v_proj.w_lora_A.data.copy_(
                    layer_weights['v_proj']['w_lora_A'].half())
                layer.v_proj.w_lora_B.data.copy_(
                    layer_weights['v_proj']['w_lora_B'].half())
            if 'o' in self.cfg.TRAINER.LORA.PARAMS and 'proj' in layer_weights:
                layer.proj.w_lora_A.data.copy_(
                    layer_weights['proj']['w_lora_A'].half())
                layer.proj.w_lora_B.data.copy_(
                    layer_weights['proj']['w_lora_B'].half())


        print(f'LoRA weights loaded from {load_path}')

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        name = "lora"

        weights = self.lora_weights
        metadata = {
            'r': self.cfg.TRAINER.LORA.R,
            'alpha': self.cfg.TRAINER.LORA.ALPHA,
            'encoder': self.cfg.TRAINER.LORA.ENCODER,
            'params': self.cfg.TRAINER.LORA.PARAMS,
            'position': self.cfg.TRAINER.LORA.POSITION
        }

        save_data = {
            'weights': weights,
            'metadata': metadata
        }

        # to manage names like ViT-B/16
        backbone = self.cfg.MODEL.BACKBONE.NAME
        backbone = backbone.replace('/', '').replace('-', '').lower()  
        save_dir = f'{directory}/{self.cfg.DATASET.NAME}/{backbone}/{name}'
        os.makedirs(save_dir, exist_ok=True)
        filename = "best"
        save_path = f'{save_dir}/{filename}.pt'
        torch.save(save_data, save_path)
        print(f'LoRA weights saved to {save_path}')            


    def make_weight(self, list_lora_layers):
        weights = {}
        for i, layer in enumerate(list_lora_layers):
            layer_weights = {}
            if 'q' in self.cfg.TRAINER.LORA.PARAMS:
                layer_weights['q_proj'] = {
                    'w_lora_A': layer.q_proj.w_lora_A.data,
                    'w_lora_B': layer.q_proj.w_lora_B.data
                }
            if 'k' in self.cfg.TRAINER.LORA.PARAMS:
                layer_weights['k_proj'] = {
                    'w_lora_A': layer.k_proj.w_lora_A.data,
                    'w_lora_B': layer.k_proj.w_lora_B.data
                }
            if 'v' in self.cfg.TRAINER.LORA.PARAMS:
                layer_weights['v_proj'] = {
                    'w_lora_A': layer.v_proj.w_lora_A.data,
                    'w_lora_B': layer.v_proj.w_lora_B.data
                }
            if 'o' in self.cfg.TRAINER.LORA.PARAMS:
                layer_weights['proj'] = {
                    'w_lora_A': layer.proj.w_lora_A.data,
                    'w_lora_B': layer.proj.w_lora_B.data
                }

            weights[f'layer_{i}'] = layer_weights
        return weights
    
