"""
Hierarchical VLA Evaluation Wrapper for BEHAVIOR-1K

This wrapper implements the full hierarchical inference pipeline:
- Dense skill prediction at every frame
- Dynamic memory building from generated skills
- EOS detection for skill boundary identification
- Comprehensive logging for debugging and analysis
"""

import json
import logging
import numpy as np
import torch
import traceback
from pathlib import Path
from collections import deque
from datetime import datetime
from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad

logger = logging.getLogger(__name__)

RESIZE_SIZE = 224


class HierarchicalB1KPolicyWrapper:
    """
    Hierarchical evaluation wrapper that generates skills and actions.
    
    Features:
    - Dense skill generation at every frame
    - Dynamic memory buffer (max ~15 skills)
    - EOS detection for skill completion
    - Comprehensive logging to file
    - Console prints for memory updates
    """
    
    def __init__(
        self,
        policy: BasePolicy,
        text_prompt: str = "Turn on the radio receiver that's on the table in the living room.",
        control_mode: str = "temporal_ensemble",
        action_horizon: int = 50,
        log_dir: str = "./eval_logs",
        max_memory_skills: int = 15,
    ) -> None:
        self.policy = policy
        self.text_prompt = text_prompt
        self.control_mode = control_mode
        self.action_horizon = action_horizon
        self.max_memory_skills = max_memory_skills
        
        # Action control
        self.action_queue = deque([], maxlen=action_horizon)
        self.last_action = {"actions": np.zeros((action_horizon, 23), dtype=np.float64)}
        
        # Hierarchical state
        self.memory_buffer = []  # List of completed skill JSONs
        self.current_skill = None
        self.skill_start_frame = 0
        self.step_counter = 0
        self.episode_counter = 0
        
        # Temporal ensemble params
        self.replan_interval = 10
        self.max_len = 50
        self.temporal_ensemble_max = 5
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = None
        self.log_buffer = []
        
        logger.info(f"Initialized HierarchicalB1KPolicyWrapper")
        logger.info(f"  Control mode: {control_mode}")
        logger.info(f"  Action horizon: {action_horizon}")
        logger.info(f"  Max memory skills: {max_memory_skills}")
        logger.info(f"  Log directory: {log_dir}")
    
    def reset(self):
        """Reset for new episode."""
        # Reset action control
        self.action_queue = deque([], maxlen=self.action_horizon)
        self.last_action = {"actions": np.zeros((self.action_horizon, 23), dtype=np.float64)}
        
        # Reset hierarchical state
        self.memory_buffer = []
        self.current_skill = None
        self.skill_start_frame = 0
        self.step_counter = 0
        self.episode_counter += 1
        
        # Flush and create new log file
        if self.current_log_file is not None:
            self._flush_log()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = self.log_dir / f"episode_{self.episode_counter:04d}_{timestamp}.jsonl"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {self.episode_counter} started")
        logger.info(f"Log file: {self.current_log_file}")
        logger.info(f"{'='*60}\n")
    
    def process_obs(self, obs: dict) -> dict:
        """Process observation to model input format."""
        prop_state = obs["robot_r1::proprio"][None]
        img_obs = np.stack(
            [
                resize_with_pad(
                    obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
            ],
            axis=1,
        )
        return {
            "observation": img_obs,  # Shape: (1, 3, H, W, C)
            "proprio": prop_state,
            "prompt": self.text_prompt,
        }
    
    def _build_memory_text(self) -> str:
        """Build memory text from completed skills."""
        if not self.memory_buffer:
            return None
        
        # Format: <PAST_SKILL>{skill_json}</PAST_SKILL>
        memory_parts = []
        for skill_json in self.memory_buffer[-self.max_memory_skills:]:
            memory_parts.append(f"<PAST_SKILL>{skill_json}</PAST_SKILL>")
        
        return "".join(memory_parts)
    
    def _infer_with_skill_generation(self, batch: dict) -> tuple:
        """
        Run inference with skill generation.
        
        Returns:
            (actions, skill_json, has_eos)
        """
        # Build memory text from buffer
        memory_text = self._build_memory_text()
        
        # Call policy inference with memory_text as keyword argument
        # This triggers hierarchical skill generation in Policy.infer()
        try:
            result = self.policy.infer(batch, memory_text=memory_text if memory_text else None)
            actions = result["actions"]
            
            # Extract skill prediction if available
            skill_json = result.get("skill_json", None)
            has_eos = result.get("has_eos", False)
            
            # Debug: Log if skill generation is working
            if self.step_counter == 0:
                logger.info(f"Policy result keys: {result.keys()}")
                logger.info(f"Skill generation check: skill_json={skill_json}, has_eos={has_eos}")
                logger.info(f"Memory text provided: {bool(memory_text)}")
            
            return actions, skill_json, has_eos
            
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            traceback.print_exc()
            # Fallback to last action
            return self.last_action["actions"], None, False
    
    def _update_memory(self, skill_json: str):
        """Add completed skill to memory buffer."""
        self.memory_buffer.append(skill_json)
        
        # Keep only last N skills
        if len(self.memory_buffer) > self.max_memory_skills:
            self.memory_buffer = self.memory_buffer[-self.max_memory_skills:]
        
        # Console print for visibility
        skill_duration = self.step_counter - self.skill_start_frame
        print(f"\n{'='*60}")
        print(f"[Frame {self.step_counter}] Skill Completed!")
        print(f"  Skill: {skill_json}")
        print(f"  Duration: {skill_duration} frames")
        print(f"  Memory size: {len(self.memory_buffer)} skills")
        print(f"{'='*60}\n")
    
    def _log_step(self, skill_json: str, has_eos: bool, actions: np.ndarray):
        """Log current step to buffer."""
        log_entry = {
            "episode": self.episode_counter,
            "frame": self.step_counter,
            "skill_json": skill_json,
            "has_eos": has_eos,
            "memory_size": len(self.memory_buffer),
            "action_shape": list(actions.shape) if actions is not None else None,
        }
        
        # Add memory snapshot if EOS detected
        if has_eos:
            log_entry["memory_buffer"] = self.memory_buffer.copy()
            log_entry["skill_duration"] = self.step_counter - self.skill_start_frame
        
        self.log_buffer.append(log_entry)
        
        # Flush every 100 steps to avoid memory issues
        if len(self.log_buffer) >= 100:
            self._flush_log()
    
    def _flush_log(self):
        """Write log buffer to file."""
        if not self.log_buffer or self.current_log_file is None:
            return
        
        with open(self.current_log_file, "a") as f:
            for entry in self.log_buffer:
                f.write(json.dumps(entry) + "\n")
        
        self.log_buffer = []
    
    def act(self, input_obs):
        """
        Main action generation with hierarchical skill prediction.
        
        This implements the full hierarchical pipeline:
        1. Generate skill prediction (dense, every frame)
        2. Detect EOS and update memory
        3. Generate actions using temporal ensemble
        """
        input_obs = self.process_obs(input_obs)
        
        # Prepare batch for policy
        nbatch = input_obs.copy()
        if nbatch["observation"].shape[-1] != 3:
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))
        
        joint_positions = nbatch["proprio"][0]
        batch = {
            "observation/egocentric_camera": nbatch["observation"][0, 0],
            "observation/wrist_image_left": nbatch["observation"][0, 1],
            "observation/wrist_image_right": nbatch["observation"][0, 2],
            "observation/state": joint_positions,
            "prompt": self.text_prompt,
        }
        
        # ============================================
        # HIERARCHICAL INFERENCE
        # ============================================
        # Generate skill + actions (dense prediction every frame)
        actions, skill_json, has_eos = self._infer_with_skill_generation(batch)
        
        # Update current skill
        if skill_json is not None:
            self.current_skill = skill_json
        
        # Handle EOS detection (skill completed)
        if has_eos and self.current_skill is not None:
            self._update_memory(self.current_skill)
            self.skill_start_frame = self.step_counter + 1
            self.current_skill = None
        
        # Log this step
        self._log_step(skill_json, has_eos, actions)
        
        # ============================================
        # ACTION CONTROL (same as before)
        # ============================================
        target_joint_positions = actions.copy()
        
        if self.control_mode == 'receeding_horizon':
            self.action_queue = deque([a for a in target_joint_positions[:self.max_len]])
            final_action = self.action_queue.popleft()[None]
        
        elif self.control_mode == 'temporal_ensemble':
            new_actions = deque(target_joint_positions)
            self.action_queue.append(new_actions)
            actions_current_timestep = np.empty((len(self.action_queue), target_joint_positions.shape[1]))
            
            k = 0.005
            for i, q in enumerate(self.action_queue):
                actions_current_timestep[i] = q.popleft()
            
            exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
            exp_weights = exp_weights / exp_weights.sum()
            
            final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            final_action[-9] = target_joint_positions[0, -9]
            final_action[-1] = target_joint_positions[0, -1]
            final_action = final_action[None]
        else:
            final_action = target_joint_positions
        
        self.step_counter += 1
        return torch.from_numpy(final_action)
    
    def finalize(self):
        """Finalize logging at episode end."""
        self._flush_log()
        logger.info(f"Episode {self.episode_counter} completed")
        logger.info(f"  Total frames: {self.step_counter}")
        logger.info(f"  Final memory size: {len(self.memory_buffer)}")
        logger.info(f"  Log saved to: {self.current_log_file}")
