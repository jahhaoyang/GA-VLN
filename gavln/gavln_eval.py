import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import tqdm
import torch
import copy
import json
import random
import argparse
import itertools
import quaternion
import transformers
import numpy as np

from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from depth_camera_filtering import filter_depth
from transformers.image_utils import to_numpy_array

from torchvision import transforms as TF

import habitat
from habitat import logger, Env
from habitat_extensions import measures
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from model.model_gavln import GAVLNForCausalLM
from utils.utils import dict_to_cuda
from utils.dist import *
from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN
from utils.utils import DEFAULT_HIS_TOKEN, HIS_TOKEN_INDEX

import torch.nn.functional as F

from functools import wraps
import threading

_log_files = {}
_log_lock = threading.Lock()

def setup_rank_logger(output_path, rank):
    log_dir = os.path.join(output_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'rank_{rank}.log')
    
    with _log_lock:
        if rank not in _log_files:
            _log_files[rank] = open(log_file, 'w', buffering=1)
    
    return log_file

def rank_print(*args, rank=None, **kwargs):
    if rank is None:
        try:
            rank = get_rank()
        except:
            rank = 0
    
    message = ' '.join(str(arg) for arg in args)
    
    with _log_lock:
        if rank in _log_files:
            _log_files[rank].write(message + '\n')
            _log_files[rank].flush()
        else:
            print(f"[Rank {rank}] {message}", **kwargs)


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        tokenizer: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
        gpu_id: int = 0,
    ):
        self.args = args

        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.device = torch.device(f'cuda:{gpu_id}')

        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            # self.config.habitat.task.measurements.success.success_distance=3.0
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        rank_print(f"config = {type(self.config)}")
        rank_print(OmegaConf.to_yaml(self.config))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        self.image_processor = model.get_vision_tower().image_processor
        self.model = model
        self.tokenizer = tokenizer
        prompt = f"You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })

        self.conjunctions = ['you can see ']

        self.bev_grid_size = args.bev_grid_size
        self.bev_range = args.bev_range
        self.bev_pos_temp = args.bev_pos_temp
        self.num_steps = args.bev_num_steps
        self.max_steps = args.bev_max_steps

        self.num_front_view = args.num_front_view
        self.dia_round = args.dia_round
        
        self.patch_grid_size = args.patch_grid_size
        self.patch_size_pixel = args.patch_size_pixel
        self.image_height_width = self.image_processor.crop_size['height']
        
    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env

    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        sensor_cfg = self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx 
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [ 0.0, fy, cy, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]
        ])
        
        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result_{idx}.json')):
            with open(os.path.join(self.output_path, f'result_{idx}.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    sucs.append(res['success'])
                    spls.append(res['spl'])
                    oss.append(res['os'])
                    ones.append(res['ne'])

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]

            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                episode_id = episode.episode_id

                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue
                self.model.reset_for_env(idx)
                env.current_episode = episode
                observations = env.reset()

                step_id = 0
                rotation_list = []
                position_list = []
                world_xy_list = []
                action_seq = []
                vis_frames = []

                past_key_values = None
                output_ids = None
                frame_ids = []

                images_vggt = []
                world_xy_vggt_list = []
                to_tensor = TF.ToTensor()

                while not env.episode_over:
                    self.model.eval()

                    if len(action_seq) == 0:
                        
                        if output_ids is None:
                            sources = copy.deepcopy(self.conversation)
                            sources[0]["value"] += f' These are your bird eye view feature map of historical observations: {DEFAULT_MEMORY_TOKEN}.'
                            if self.num_front_view != 0 and step_id >= self.dia_round * self.num_steps:
                                sources[0]["value"] += f' You have been given a video of historical observations: {DEFAULT_HIS_TOKEN}.'
                            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', episode.instruction.instruction_text)

                            input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=True)
                            frame_ids = np.arange(max(int(np.floor((step_id - self.max_steps) / self.num_steps) * self.num_steps), 0), step_id+1, self.num_steps) // self.num_steps
                            frame_ids_bev = torch.from_numpy(frame_ids)
                            frame_ids_front = torch.tensor([step_id // self.num_steps])
                            frame_ids_his = frame_ids_bev[-(1+self.num_front_view):-1]

                        else:
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=False)
                            input_ids = torch.cat([output_ids,input_ids.to(output_ids.device)], dim=1)
                            frame_ids = torch.tensor([step_id // self.num_steps])
                            frame_ids_bev = frame_ids
                            frame_ids_front = frame_ids
                            frame_ids_his = frame_ids_bev[-1:-1]

                        frame_ids_bev_idx = frame_ids_bev - frame_ids_bev[0]
                        frame_ids_front_idx = frame_ids_front - frame_ids_bev[0]
                        frame_ids_his_idx = frame_ids_his - frame_ids_bev[0]

                        frames_id_dict = {'frame_ids_bev': frame_ids_bev_idx, \
                                        'frame_ids_front': frame_ids_front_idx, \
                                        'frame_ids_his': frame_ids_his_idx
                                        }


                        rgb = observations["rgb"]
                        image = Image.fromarray(rgb).convert('RGB')
                        
                        ############ rgb vggt ############
                        image_vggt = image
                        new_width = 518
                        new_height = round(height * (new_width / width) / 14) * 14
                        
                        image_vggt = image_vggt.resize((new_width, new_height), Image.Resampling.BICUBIC)
                        image_vggt = to_tensor(image_vggt)
                        images_vggt.append(image_vggt)

                        bev_images_vggt = torch.stack(images_vggt)[frame_ids_bev].to(torch.bfloat16)

                        ############ rbg front view ############
                        image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0] 
                        vis_frames.append(image)

                        latest_images = vis_frames[-1]
                        all_images = torch.stack(vis_frames)[frame_ids].to(torch.bfloat16)

                        depth = observations["depth"]
                        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                        depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                        depth = depth * 1000
                        depth_image = Image.fromarray(depth.astype(np.uint16), mode='I;16')
                        depth_image_vggt = depth_image

                        ############ depth & all_info front view ############
                        resized_depth_image = depth_image.resize((self.image_height_width, self.image_height_width), Image.NEAREST)
                        depth_img = to_numpy_array(resized_depth_image)
                        depth_img = depth_img / 1000
                        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0).float()
                        depth_image = F.interpolate(depth_tensor, size=(self.patch_grid_size,self.patch_grid_size), scale_factor=None, mode='nearest')
                        depth_image = depth_image.squeeze().numpy()

                        patch_centers = torch.arange(self.patch_grid_size, dtype=torch.bfloat16).to(local_rank)
                        patch_centers = patch_centers * self.patch_size_pixel + self.patch_size_pixel / 2 
                        v_coords, u_coords = torch.meshgrid(patch_centers, patch_centers, indexing='ij')
                        u_coords_original = u_coords * width // self.image_height_width
                        v_coords_original = v_coords * height // self.image_height_width

                        patch_depth = torch.from_numpy(depth_image).to(local_rank)

                        cam_x = (u_coords_original - cx) * patch_depth / fx
                        cam_y = (v_coords_original - cy) * patch_depth / fy
                        cam_z = patch_depth
                        cam_coords_homo = torch.stack([cam_x, cam_y, cam_z, torch.ones_like(cam_x)], dim=-1).to(local_rank)

                        agent_state = env.sim.get_agent_state()
                        rotation = agent_state.rotation
                        translation = agent_state.position
                        rotation_matrix = quaternion.as_rotation_matrix(rotation)
                        transformation_matrix = np.eye(4)
                        transformation_matrix[:3, :3] = - rotation_matrix
                        transformation_matrix[:3, 3] = translation

                        camera_pose = torch.from_numpy(transformation_matrix).to(torch.float32).to(local_rank)
                        world_coords = torch.matmul(cam_coords_homo, camera_pose.T)

                        rotation_out = torch.from_numpy(rotation_matrix)[[0, 2], :][:, [0, 2]].to(torch.bfloat16)
                        rotation_list.append(rotation_out)

                        translation_out = torch.from_numpy(translation)[[0, 2]].to(torch.bfloat16)
                        position_list.append(translation_out)
                        
                        world_xy = world_coords[..., [0,2]].cpu().to(torch.bfloat16)
                        world_xy_list.append(world_xy)

                        latest_rotation = torch.stack(rotation_list)[frame_ids_front[0]]
                        latest_position = torch.stack(position_list)[frame_ids_front[0]]
                        world_xy = torch.stack(world_xy_list)[frame_ids_bev]


                        ############ depth & all_info vggt ############
                        resized_depth_image = depth_image_vggt.resize((518, 392), Image.Resampling.BICUBIC)
                        depth_img = to_numpy_array(resized_depth_image)
                        depth_img = depth_img / 1000
                        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0).float()
                        depth_image = F.interpolate(depth_tensor, size=(28,37), scale_factor=None, mode='nearest')
                        depth_image = depth_image.squeeze().numpy()

                        patch_grid_size_w = 37
                        patch_grid_size_h = 28

                        patch_centers_w = torch.arange(37, dtype=torch.bfloat16).to(local_rank)
                        patch_centers_h = torch.arange(28, dtype=torch.bfloat16).to(local_rank)

                        patch_centers_w = patch_centers_w * 14 + 14 / 2 
                        patch_centers_h = patch_centers_h * 14 + 14 / 2

                        v_coords, u_coords = torch.meshgrid(patch_centers_h, patch_centers_w, indexing='ij')
                        u_coords_original = u_coords * width // 518
                        v_coords_original = v_coords * height // 392

                        patch_depth = torch.from_numpy(depth_image).to(local_rank)

                        cam_x = (u_coords_original - cx) * patch_depth / fx
                        cam_y = (v_coords_original - cy) * patch_depth / fy
                        cam_z = patch_depth
                        cam_coords_homo = torch.stack([cam_x, cam_y, cam_z, torch.ones_like(cam_x)], dim=-1).to(local_rank)

                        world_coords = torch.matmul(cam_coords_homo, camera_pose.T)
       
                        world_xy_vggt = world_coords[..., [0,2]].cpu().to(torch.bfloat16)
                        world_xy_vggt_list.append(world_xy_vggt)

                        latest_position_vggt = latest_position
                        latest_rotation_vggt = latest_rotation
                        world_xy_vggt = torch.stack(world_xy_vggt_list)[frame_ids_bev]

                        input_dict = {'input_ids': input_ids, \
                                        'position_ids': None, \
                                        'attention_mask': None, \
                                        'labels': None, \
                                        'all_images': all_images.unsqueeze(0), \
                                        'image_sizes': None, \
                                        'img_lens': [len(frame_ids)], \
                                        'positions': latest_position.unsqueeze(0), \
                                        'rotations': latest_rotation.unsqueeze(0), \
                                        'world_xy': world_xy.unsqueeze(0), \
                                        'bev_images_vggt': bev_images_vggt.unsqueeze(0), \
                                        'img_lens_vggt': [bev_images_vggt.shape[0]], \
                                        'positions_vggt': latest_position.unsqueeze(0), \
                                        'rotations_vggt': latest_rotation.unsqueeze(0), \
                                        'world_xy_vggt': world_xy_vggt.unsqueeze(0), \
                                        'frames_id_dict': [frames_id_dict], \
                                        'env_id':idx
                        }
                    
                        input_dict = dict_to_cuda(input_dict, self.device)

                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            outputs = self.model.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=past_key_values)
                        
                        output_ids = outputs.sequences
                        past_key_values = outputs.past_key_values
                        llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                        action_seq = self.parse_actions(llm_outputs)


                        if len(action_seq) == 0: ## if generated llm without Specific values
                            action_seq = [0]
                    
                    action = action_seq.pop(0)
                    observations = env.step(action)
                    step_id += 1
                    if step_id % (self.dia_round * self.num_steps) == 0:
                        self.model.reset_for_env(idx)
                        output_ids = None
                        past_key_values = None
                        time_ids = []
                                            
                metrics = env.get_metrics()

                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                rank_print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }
                
                with open(os.path.join(self.output_path, f'result_{idx}.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)     

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def preprocess_qwen(self, sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.", add_system: bool = False):
        # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        roles = {"human": "user", "gpt": "assistant"}

        # Add image tokens to tokenizer as a special tokens
        # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
        tokenizer = copy.deepcopy(tokenizer)
        # When there is actually an image, we add the image tokens as a special token
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)
            tokenizer.add_tokens(["<history>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        history_token_index = tokenizer.convert_tokens_to_ids("<history>")

        im_start, im_end = tokenizer.additional_special_tokens_ids
        # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
        unmask_tokens_idx =  [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids

        # Reset Qwen chat templates so that it won't include system message every time we apply
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # _system = tokenizer("system").input_ids + nl_tokens
        # _user = tokenizer("user").input_ids + nl_tokens
        # _assistant = tokenizer("assistant").input_ids + nl_tokens

        # Apply prompt templates
        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else: 
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                # Skip the first one if it is not from human
                source = source[1:]

            input_id, target = [], []

            # import ipdb; ipdb.set_trace()
            # New version, use apply chat template
            # Build system message for each sentence
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            for conv in source:
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role =  roles.get(role, role)
                
                conv = [{"role" : role, "content" : content}]
                # import ipdb; ipdb.set_trace()
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id
            
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
                if encode_id == history_token_index:
                    input_id[idx] = HIS_TOKEN_INDEX

            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids, conversations


def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/gavln')

    parser.add_argument("--vision_tower_path", type=str, default='./model/siglip-so400m-patch14-384')
    parser.add_argument("--vggt_path", type=str, default='./model/VGGT-1B')

    parser.add_argument("--bev_num_steps", type=int, default=4)
    parser.add_argument("--bev_max_steps", type=int, default=32)
    parser.add_argument("--bev_grid_size", type=float, default=0.25)
    parser.add_argument("--bev_range", type=float, default=10.0)
    parser.add_argument("--bev_pos_temp", type=float, default=10000)

    parser.add_argument("--num_front_view", type=int, default=0)
    parser.add_argument("--dia_round", type=int, default=2)

    parser.add_argument("--patch_grid_size", type=int, default=27)
    parser.add_argument("--patch_size_pixel", type=int, default=14)

    parser.add_argument("--save_video", action="store_true", default=False)
    
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    args = parser.parse_args()

    init_distributed_mode(args)

    local_rank = args.local_rank
    rank = get_rank()
    
    setup_rank_logger(args.output_path, rank)

    torch.cuda.set_device(local_rank)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path,
                                                        model_max_length=args.model_max_length,
                                                        padding_side="right")
    
    config = transformers.AutoConfig.from_pretrained(args.model_path)

    config.mm_vision_tower = args.vision_tower_path
    config.vggt_model_path = args.vggt_path

    model = GAVLNForCausalLM.from_pretrained(
                args.model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=False,
                )

    model.model.bev_grid_size = args.bev_grid_size
    model.model.bev_range = args.bev_range
    model.model.bev_pos_temp = args.bev_pos_temp

    model.requires_grad_(False)
    model.to(local_rank)
    
    evaluate(model, tokenizer, args, local_rank)



def evaluate(model, tokenizer, args, local_rank):
    
    world_size = get_world_size()
    model.eval()
    model.reset(world_size)
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        epoch=0,
        args=args,
        gpu_id=local_rank
    )

    print("### get_rank()", get_rank())
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank()) 

    print("### get_rank() done", get_rank())

    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    result_all = {
                    "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
                    "spls_all": (sum(spls_all)/len(spls_all)).item(),
                    "oss_all": (sum(oss_all)/len(oss_all)).item(),
                    "ones_all": (sum(ones_all)/len(ones_all)).item(),
                    'length': len(sucs_all)
                }
    
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))


if __name__ == "__main__":
    eval()
