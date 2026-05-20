import math
import torch
import torch.nn as nn
from math import ceil
from typing import List, Optional, Union, Tuple

import numpy as np

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import Qwen2ForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenModel
from llava.model.llava_arch import LlavaMetaForCausalLM
from utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX, HIS_TOKEN_INDEX

from llava.utils import rank0_print

from vggt.models.vggt import VGGT

class GAVLNModel(LlavaQwenModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GAVLNModel, self).__init__(config)
        
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False
        

class GAVLNForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Qwen2ForCausalLM, self).__init__(config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        config.delay_load = True
        
        self.model = GAVLNModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bev_grid_size = config.bev_grid_size
        self.bev_range = config.bev_range
        self.bev_pos_temp = config.bev_pos_temp

        self.bev_grid_num = int(2 * self.bev_range / self.bev_grid_size)  # 80
        self.feature_dim = 1152
        self.register_buffer('pos_encoding', self._generate_real_coord_encoding())

        self.vggt = VGGT.from_pretrained(config.vggt_model_path)
        self.bev_proj_ln = nn.LayerNorm(2048, eps=1e-6)
        self.bev_proj_head = nn.Sequential(
                nn.Linear(2048, 4096),
                nn.GELU(),
                nn.Linear(4096, 1152),
            )

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_model(self):
        return self.model

    def _generate_real_coord_encoding(self):
                
        coords_1d = torch.linspace(
            - self.bev_range + self.bev_grid_size / 2,
            self.bev_range - self.bev_grid_size / 2,
            self.bev_grid_num
        )
        real_y, real_x = torch.meshgrid(coords_1d, coords_1d, indexing='ij')

        pos_encoding = torch.zeros(self.bev_grid_num, self.bev_grid_num, self.feature_dim)
        
        dim_per_axis = self.feature_dim // 2
        div_term = torch.exp(torch.arange(0, dim_per_axis, 2).float() * 
                           -(math.log(self.bev_pos_temp) / dim_per_axis))
        
        for i in range(0, dim_per_axis, 2):
            freq_idx = i // 2
            if freq_idx < len(div_term):
                pos_encoding[:, :, i] = torch.sin(real_x / div_term[freq_idx])
                if i + 1 < dim_per_axis:
                    pos_encoding[:, :, i + 1] = torch.cos(real_x / div_term[freq_idx])
         
        for i in range(dim_per_axis, self.feature_dim, 2):
            freq_idx = (i - dim_per_axis) // 2
            if freq_idx < len(div_term):
                pos_encoding[:, :, i] = torch.sin(real_y / div_term[freq_idx])
                if i + 1 < self.feature_dim:
                    pos_encoding[:, :, i + 1] = torch.cos(real_y / div_term[freq_idx])
        
        pos_encoding = pos_encoding.reshape(-1, self.feature_dim)
        return pos_encoding
    
    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [ceil(height / stride), ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1).contiguous()
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature
    
    def encode_images(self, 
        images,
        bev_images_vggt,
        positions, 
        rotations, 
        world_xy,
        positions_vggt,
        rotations_vggt,
        world_xy_vggt,
        img_lens,
        img_lens_vggt,
        frames_id_dict
    ):
        
        batch_size, num_view, _, H, W = images.shape
        num_patches_per_side = self.get_model().get_vision_tower().num_patches_per_side

        image_features = self.get_model().get_vision_tower()(images.flatten(0,1))
        image_features = image_features.permute(0, 2, 1).reshape(batch_size, num_view, -1, num_patches_per_side, num_patches_per_side).contiguous()

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(bev_images_vggt)
            vggt_features = aggregated_tokens_list[-2][:, :, patch_start_idx:]

        bev_features = []
        front_image_features = []
        his_image_features = []

        for b, image_feature in enumerate(image_features):
            
            #################### 1. front view ####################
            image_feature = image_feature[:img_lens[b], ...]
            image_feature = image_feature.flatten(2,3).permute(0,2,1).contiguous()

            feature_dim = image_feature.shape[-1]

            frame_ids_bev = frames_id_dict[b]['frame_ids_bev'].to(image_feature.device)
            frame_ids_front = frames_id_dict[b]['frame_ids_front'].to(image_feature.device)
            frame_ids_his = frames_id_dict[b]['frame_ids_his'].to(image_feature.device)

            if len(frame_ids_his):
                frames_ids_his_and_front = torch.cat((frame_ids_his, frame_ids_front))
            else:
                frames_ids_his_and_front = frame_ids_front

            his_and_front_image_feature = image_feature[frames_ids_his_and_front]

            bev_image_feature = image_feature[frame_ids_bev]

            patch_world_xy = world_xy[b]
            patch_world_xy = patch_world_xy.reshape(-1, num_patches_per_side*num_patches_per_side, 2)

            rotation_2d = - rotations[b]
            current_pos_2d = positions[b]
            rel_positions = patch_world_xy - current_pos_2d

            rel_r_positions = torch.matmul(rel_positions, rotation_2d)

            bev_image_feature = bev_image_feature.reshape(-1, feature_dim)
            rel_r_positions = rel_r_positions.reshape(-1, 2)

            bev_grid_indices = rel_r_positions // self.bev_grid_size + int(self.bev_range / self.bev_grid_size)
            bev_grid_indices = bev_grid_indices.long()

            bev_valid_mask = (
                (bev_grid_indices[:, 0] >= 0) & (bev_grid_indices[:, 0] < self.bev_grid_num) &
                (bev_grid_indices[:, 1] >= 0) & (bev_grid_indices[:, 1] < self.bev_grid_num)
            )  

            valid_indices = bev_grid_indices[bev_valid_mask]
            valid_features = bev_image_feature[bev_valid_mask]

            #################### 2. vggt feature ####################
            vggt_feature = vggt_features[b]
            vggt_feature = vggt_feature[:img_lens_vggt[b], ...]

            vggt_feature = self.bev_proj_head(self.bev_proj_ln(vggt_feature))

            vggt_patch_num = vggt_feature.shape[1]

            patch_world_xy_vggt = world_xy_vggt[b]
            patch_world_xy_vggt = patch_world_xy_vggt.reshape(-1, vggt_patch_num, 2)

            rotation_2d_vggt = - rotations_vggt[b]
            current_pos_2d_vggt = positions_vggt[b]

            rel_positions_vggt = patch_world_xy_vggt - current_pos_2d_vggt
            rel_r_positions_vggt = torch.matmul(rel_positions_vggt, rotation_2d_vggt) 

            vggt_bev_image_feature = vggt_feature.reshape(-1, self.feature_dim)
            rel_r_positions_vggt = rel_r_positions_vggt.reshape(-1, 2)

            vggt_bev_grid_indices = rel_r_positions_vggt // self.bev_grid_size + int(self.bev_range / self.bev_grid_size)
            vggt_bev_grid_indices = vggt_bev_grid_indices.long()
            vggt_bev_valid_mask = (
                (vggt_bev_grid_indices[:, 0] >= 0) & (vggt_bev_grid_indices[:, 0] < self.bev_grid_num) &
                (vggt_bev_grid_indices[:, 1] >= 0) & (vggt_bev_grid_indices[:, 1] < self.bev_grid_num)
            )
            vggt_valid_indices = vggt_bev_grid_indices[vggt_bev_valid_mask]
            vggt_valid_features = vggt_bev_image_feature[vggt_bev_valid_mask]

            #################### 3. bev fusin ####################

            bev_feature = torch.zeros(self.bev_grid_num * self.bev_grid_num, self.feature_dim, dtype=image_feature.dtype, device=image_feature.device)
            bev_counts = torch.zeros(self.bev_grid_num * self.bev_grid_num, dtype=image_feature.dtype, device=image_feature.device)

            flat_indices = valid_indices[:, 1] * self.bev_grid_num + valid_indices[:, 0]
            bev_feature.scatter_add_(0, flat_indices.unsqueeze(1).expand(-1, self.feature_dim), valid_features)
            bev_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=bev_counts.dtype))

            vggt_flat_indices = vggt_valid_indices[:, 1] * self.bev_grid_num + vggt_valid_indices[:, 0]
            bev_feature.scatter_add_(0, vggt_flat_indices.unsqueeze(1).expand(-1, self.feature_dim), vggt_valid_features)
            bev_counts.scatter_add_(0, vggt_flat_indices, torch.ones_like(vggt_flat_indices, dtype=bev_counts.dtype))

            nonzero_mask = bev_counts > 0
            bev_feature[nonzero_mask] = bev_feature[nonzero_mask] / bev_counts[nonzero_mask].unsqueeze(-1)

            bev_feature = bev_feature + self.pos_encoding
            nonzero_features = bev_feature[nonzero_mask].contiguous()

            nonzero_features = self.get_model().mm_projector(nonzero_features)
            bev_features.append(nonzero_features)

            #################### 4. history & front ####################

            his_and_front_image_feature = self.get_model().mm_projector(his_and_front_image_feature)
            his_and_front_image_feature = self.get_2dPool(his_and_front_image_feature, 2)
            
            if len(frame_ids_his):
                his_image_features.append(his_and_front_image_feature[0:len(frame_ids_his)])
            else:
                his_image_features.append(None)
            front_image_features.append(his_and_front_image_feature[len(frame_ids_his):])
            
        return bev_features, front_image_features, his_image_features

    def encode_latest_image(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        image_features = self.get_2dPool(image_features, 2)
        return image_features
    
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        all_images,
        bev_images_vggt,
        image_sizes,
        img_lens,
        img_lens_vggt,
        positions,
        rotations,
        world_xy,
        positions_vggt,
        rotations_vggt,
        world_xy_vggt,
        frames_id_dict
    ):  
        vision_tower = self.get_vision_tower()
        if vision_tower is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        bev_features, \
        front_image_features, \
        history_image_features = self.encode_images(all_images, \
                                                    bev_images_vggt, \
                                                    positions, \
                                                    rotations, \
                                                    world_xy, \
                                                    positions_vggt, \
                                                    rotations_vggt, \
                                                    world_xy_vggt, \
                                                    img_lens, \
                                                    img_lens_vggt, \
                                                    frames_id_dict)
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_memories = (cur_input_ids == MEMORY_TOKEN_INDEX).sum()
            num_historys = (cur_input_ids == HIS_TOKEN_INDEX).sum()
            num_specials = num_images + num_memories + num_historys

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            memory_token_indices = torch.where(cur_input_ids == MEMORY_TOKEN_INDEX)[0].tolist()
            history_token_indices = torch.where(cur_input_ids == HIS_TOKEN_INDEX)[0].tolist()

            special_token_indices = sorted(image_token_indices + memory_token_indices + history_token_indices)
            special_tokens = [cur_input_ids[indice] for indice in special_token_indices]
            special_token_indices = [-1] + special_token_indices + [cur_input_ids.shape[0]]

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            
            for i in range(len(special_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[special_token_indices[i]+1:special_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[special_token_indices[i]+1:special_token_indices[i+1]])
            
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))

            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []            
            cur_img_id = 0

            for i in range(num_specials + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_specials:
                    # print(f"Batch Index: {batch_idx}\n, Current Image Index: {cur_image_idx}\n, Num Images: {num_images}")
                    special_token = special_tokens[i]
                
                    if special_token == IMAGE_TOKEN_INDEX:
                        cur_image_feature = front_image_features[batch_idx][cur_img_id]
                        cur_img_id += 1
                        cur_new_input_embeds.append(cur_image_feature)
                        cur_new_labels.append(torch.full((cur_image_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        
                    elif special_token == MEMORY_TOKEN_INDEX:
                        cur_memory_feature = bev_features[batch_idx]
                        cur_new_input_embeds.append(cur_memory_feature)
                        cur_new_labels.append(torch.full((cur_memory_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                    elif special_token == HIS_TOKEN_INDEX:
                        cur_history_feature = history_image_features[batch_idx].reshape(-1, history_image_features[batch_idx].shape[-1])
                        cur_new_input_embeds.append(cur_history_feature)
                        cur_new_labels.append(torch.full((cur_history_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                    else:
                        raise NotImplementedError
            
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
            
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        
        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        all_images: torch.FloatTensor = None,
        bev_images_vggt: torch.FloatTensor = None,
        image_sizes: Optional[List[List[int]]] = None,
        img_lens: Optional[List[int]] = None,
        img_lens_vggt: Optional[List[int]] = None,
        positions: torch.FloatTensor = None,
        rotations: torch.FloatTensor = None,
        world_xy: torch.FloatTensor = None,
        positions_vggt: torch.FloatTensor = None,
        rotations_vggt: torch.FloatTensor = None,
        world_xy_vggt: torch.FloatTensor = None,
        frames_id_dict: torch.FloatTensor = None,
        task_type: torch.FloatTensor = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                all_images,
                bev_images_vggt,
                image_sizes,
                img_lens,
                img_lens_vggt,
                positions,
                rotations,
                world_xy,
                positions_vggt,
                rotations_vggt,
                world_xy_vggt,
                frames_id_dict
            )
    
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        all_images: torch.FloatTensor = None,
        image_sizes: Optional[List[List[int]]] = None,
        img_lens: Optional[List[int]] = None,
        positions: torch.FloatTensor = None,
        rotations: torch.FloatTensor = None,
        world_xy: torch.FloatTensor = None,
        frames_id_dict: torch.FloatTensor = None,
        bev_images_vggt: torch.FloatTensor = None,
        img_lens_vggt: Optional[List[int]] = None,
        positions_vggt: torch.FloatTensor = None,
        rotations_vggt: torch.FloatTensor = None,
        world_xy_vggt: torch.FloatTensor = None,

        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        if inputs_embeds is None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                all_images,
                bev_images_vggt,
                image_sizes,
                img_lens,
                img_lens_vggt,
                positions,
                rotations,
                world_xy,
                positions_vggt,
                rotations_vggt,
                world_xy_vggt,
                frames_id_dict
            )
        
        
        env_id = kwargs.pop("env_id", None)

        if self.curr_t[env_id] == 0:
            self.cache[env_id]["inputs_embeds"] = inputs_embeds
        else:
            self.cache[env_id]["inputs_embeds"] = torch.cat([self.cache[env_id]["inputs_embeds"], inputs_embeds],dim=1)
        self.curr_t[env_id] += 1

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=self.cache[env_id]["inputs_embeds"],
            **kwargs
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
                
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # print('cache_position_prepare:', cache_position, len(cache_position))
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        elif inputs_embeds is not None and len(cache_position) > 1:
            model_inputs = {"inputs_embeds": inputs_embeds[:, -len(cache_position):], "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": None,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        if images is not None:
            model_inputs['images'] = images
        if image_sizes is not None:
            model_inputs['image_sizes'] = image_sizes
        return model_inputs
    
    def reset(self, env_num):
        self.curr_t = [0] * env_num
        self.cache = [dict()] * env_num
    
    def reset_for_env(self, env_idx):
        self.curr_t[env_idx] = 0
        self.cache[env_idx] = dict()