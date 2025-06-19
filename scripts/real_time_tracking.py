# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import typing
import time
import json
import numpy as np
import torch
import trio
import warp as wp
from pathlib import Path
from dataclasses import dataclass, field
import tyro
import cv2

from realsense import MultiRealsense
from embodied_gaussians.scene_builders.domain import Body, MaskedPosedImageAndDepth
from embodied_gaussians.utils.utils import ExtrinsicsData, read_extrinsics, read_ground
from embodied_gaussians import (
    EmbodiedGaussiansBuilder, 
    EmbodiedGaussiansEnvironment,
    Ground
)
from embodied_gaussians.embodied_simulator.frames import Frames, FramesBuilder
from embodied_gaussians.vis import EmbodiedGUI
from marsoom import imgui


@dataclass
class TrackingParams:
    body_json: tyro.conf.PositionalRequiredArgs[Path]
    """
    Path to the JSON file containing the body to track.
    """
    extrinsics: Path
    """
    Path to a JSON file containing the extrinsics of the cameras.
    """
    ground: Path | None = None
    """
    Path to ground plane definition file.
    """
    fps: int = 30
    """
    Camera capture FPS
    """
    tracking_fps: int = 60
    """
    Tracking update FPS (can be higher than camera FPS)
    """
    visualize: bool = True
    """
    Show visualization
    """
    save_tracking_data: bool = False
    """
    Save tracking data to file
    """
    output_dir: Path | None = None
    """
    Directory to save tracking data
    """
    convert_bgr_to_rgb: bool = False
    """
    Convert BGR to RGB format. Set to True if red/blue colors are swapped in visualization.
    """
    fix_gaussian_colors: bool = True
    """
    Fix gaussian colors by converting from RGB to BGR for proper visualization. 
    Set to False if colors appear correct without conversion.
    """


class RealTimeTracker:
    def __init__(self, params: TrackingParams):
        self.params = params
        self.extrinsics = read_extrinsics(params.extrinsics)
        self.serial_numbers = list(self.extrinsics.keys())
        
        # Initialize environment
        self.environment = self._build_environment()
        self.environment.visual_forces_settings.iterations = 7
        
        # Tracking state
        self.current_frames: Frames | None = None
        self.frames_initialized = False
        self.camera_names = []  # Store camera names for consistent ordering
        self.tracking_data = []
        self.start_time = None
        
        # Camera setup
        self.camera_intrinsics = {}
        self.camera_depth_scale = {}
        
        print(f"Cameras: {self.serial_numbers}")
    
    @staticmethod
    def get_body(name: str) -> Body:
        path = Path(f"objects/{name}.json")
        with open(path, "r") as f:
            data = json.load(f)
        return Body.model_validate(data)

    def _build_environment(self) -> EmbodiedGaussiansEnvironment:
        """Build the environment with the body to track"""

        ground_data = read_ground(self.params.ground)
        ground = Ground(plane=ground_data)
        body_1 = RealTimeTracker.get_body("controller")
        # body_2 = RealTimeTracker.get_body("tape")
        ground_body = RealTimeTracker.get_body("ground_body")
        
        builder = EmbodiedGaussiansBuilder()
        body_id_1 = builder.add_rigid_body(body_1, add_gaussians=True)
        # body_id_2 = builder.add_rigid_body(body_2, add_gaussians=True)
        builder.add_visual_body(ground_body)
        
        final_builder = EmbodiedGaussiansBuilder()
        final_builder.set_ground_plane(ground.normal(), ground.offset())
        final_builder.add_builder(builder)
        
        # Create environment
        env = EmbodiedGaussiansEnvironment(final_builder)
        
        # Disable gravity for the tracking object (we want visual forces to control it)
        # if env.num_envs() > 0:
        #     gravity_factors = wp.to_torch(env.sim.model.gravity_factor).reshape(env.num_envs(), -1)
        #     if gravity_factors.shape[1] > body_id_1:
        #         gravity_factors[:, body_id_1] = 0.0
        #     if gravity_factors.shape[1] > body_id_2:
        #         gravity_factors[:, body_id_2] = 0.0
        
        env.stash_state()
        return env
    
    def _capture_camera_data(self, realsenses: MultiRealsense) -> list[MaskedPosedImageAndDepth]:
        """Capture data from all cameras"""
        all_camera_data = realsenses.get()
        datapoints = []
        
        for serial, camera_data in all_camera_data.items():
            if serial not in self.extrinsics:
                continue
                
            K = self.camera_intrinsics[serial]
            depth_scale = self.camera_depth_scale[serial]
            color = camera_data["color"]
            depth = camera_data["depth"]
            
            # RealSense outputs BGR format, handle color channel order
            if self.params.convert_bgr_to_rgb:
                # Convert BGR to RGB by swapping red and blue channels
                color_rgb = color[:, :, [2, 1, 0]].copy()
            else:
                # Keep original BGR format
                color_rgb = color.copy()
            
            # For tracking, we use the full image as mask (no segmentation needed)
            # You could add segmentation here if you want to track only specific parts
            mask = np.ones((color.shape[0], color.shape[1]), dtype=bool)
            
            # Get the camera transform and convert from OpenCV to Blender convention
            # The extrinsics file contains transforms in OpenCV convention (since transformation.py uses identity matrix)
            # We need to convert to Blender convention for the tracking system
            X_WC_opencv = self.extrinsics[serial].X_WC
            
            datapoint = MaskedPosedImageAndDepth(
                K=K,
                X_WC=X_WC_opencv,  # Now in Blender convention
                image=color_rgb,
                format="rgb",
                depth=depth,
                depth_scale=depth_scale,
                mask=mask,
            )
            datapoints.append(datapoint)
            
        return datapoints
    
    def _initialize_frames(self, datapoints: list[MaskedPosedImageAndDepth]) -> Frames:
        """Initialize frames object once with camera setup"""
        if not datapoints:
            return None
            
        # Get dimensions from first image
        first_image = datapoints[0].image
        width = first_image.shape[1]
        height = first_image.shape[0]
        
        # Create frames using FramesBuilder
        frame_builder = FramesBuilder(width=width, height=height)
        
        # Store camera names for consistent ordering
        self.camera_names = []
        for i, dp in enumerate(datapoints):
            # Use actual camera serial if available, otherwise use index
            if i < len(self.serial_numbers):
                camera_name = self.serial_numbers[i]
            else:
                camera_name = f"camera_{i}"
            self.camera_names.append(camera_name)
            frame_builder.add_camera(camera_name, dp.K, dp.X_WC)
        
        # Finalize the frames
        frames = frame_builder.finalize(device="cuda")
        self.frames_initialized = True
        
        return frames
    
    def _update_frames_colors(self, datapoints: list[MaskedPosedImageAndDepth]):
        """Update colors in existing frames object"""
        if not self.current_frames or not datapoints:
            return
            
        current_time = time.time() - self.start_time if self.start_time else 0.0
        
        # Update colors for all cameras
        for i, dp in enumerate(datapoints):
            if i < len(self.camera_names):
                camera_name = self.camera_names[i]
                # Convert image to torch tensor and move to GPU
                color_tensor = torch.from_numpy(dp.image.astype(np.float32) / 255.0).cuda()
                self.current_frames.update_colors(camera_name, current_time, color_tensor)
    
    async def track_with_cameras(self):
        """Main tracking loop with real cameras"""
        with MultiRealsense(
            serial_numbers=self.serial_numbers,
            enable_depth=True,
            capture_fps=self.params.fps,
            put_fps=self.params.fps
        ) as realsenses:
            print("Setting up cameras...")
            # realsenses.set_exposure(177, 70)
            # realsenses.set_white_balance(4600)
            # await trio.sleep(1)  # Give cameras time to adjust
            
            # Get camera intrinsics and depth scale
            self.camera_intrinsics = realsenses.get_intrinsics()
            self.camera_depth_scale = realsenses.get_depth_scale()
            
            print("Starting tracking...")
            self.start_time = time.time()
            
            # Tracking loop
            dt_tracking = 1.0 / self.params.tracking_fps
            
            while True:
                loop_start = time.time()
                
                # Capture camera data
                try:
                    datapoints = self._capture_camera_data(realsenses)
                    if datapoints:
                        # Initialize frames on first iteration
                        if not self.frames_initialized:
                            frames = self._initialize_frames(datapoints)
                            if frames:
                                self.environment.set_frames(frames)
                                self.current_frames = frames
                                print("Frames initialized for visualization")
                        else:
                            # Update existing frames with new images
                            self._update_frames_colors(datapoints)
                except Exception as e:
                    print(f"Error capturing camera data: {e}")
                    continue
                
                # Update environment (physics + visual forces)
                self.environment.step()
                
                # Record tracking data
                if self.params.save_tracking_data:
                    self._record_tracking_data()
                
                # Sleep to maintain tracking FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, dt_tracking - elapsed)
                await trio.sleep(sleep_time)
    
    def _record_tracking_data(self):
        """Record current tracking state"""
        if self.start_time is None:
            return
            
        current_time = time.time() - self.start_time
        
        # Get current gaussian positions and physics state
        with torch.no_grad():
            gaussian_means = self.environment.sim.gaussian_state.means.cpu().numpy()
            gaussian_colors = self.environment.sim.gaussian_state.colors.cpu().numpy()
            gaussian_opacities = self.environment.sim.gaussian_state.opacities.cpu().numpy()
            
            # Get physics body transforms
            body_transforms = wp.to_torch(self.environment.sim.state_0.body_q).cpu().numpy()
        
        tracking_record = {
            'timestamp': current_time,
            'gaussian_means': gaussian_means.tolist(),
            'gaussian_colors': gaussian_colors.tolist(),
            'gaussian_opacities': gaussian_opacities.tolist(),
            'body_transforms': body_transforms.tolist(),
        }
        
        self.tracking_data.append(tracking_record)
    
    def save_tracking_results(self):
        """Save tracking results to file"""
        if not self.params.save_tracking_data or not self.tracking_data:
            return
            
        output_dir = self.params.output_dir or Path("tracking_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"tracking_{self.body_name}_{timestamp_str}.json"
        
        tracking_summary = {
            'body_name': self.body_name,
            'start_time': self.start_time,
            'duration': time.time() - self.start_time if self.start_time else 0,
            'num_frames': len(self.tracking_data),
            'camera_serials': self.serial_numbers,
            'tracking_data': self.tracking_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(tracking_summary, f, indent=2)
        
        print(f"Saved tracking results to {output_file}")


class TrackingGUI:
    def __init__(self, tracker: RealTimeTracker):
        self.tracker = tracker
        self.show_gaussian_info = True
        self.show_camera_info = True
        self.show_tracking_stats = True
        
    def draw(self):
        """Draw tracking information GUI"""
        if self.show_tracking_stats:
            self._draw_tracking_stats()
        
        if self.show_gaussian_info:
            self._draw_gaussian_info()
            
        if self.show_camera_info:
            self._draw_camera_info()
    
    def _draw_tracking_stats(self):
        """Draw tracking statistics"""
        imgui.begin("Tracking Stats")
        
        if self.tracker.start_time:
            elapsed = time.time() - self.tracker.start_time
            imgui.text(f"Tracking time: {elapsed:.1f}s")
            imgui.text(f"Recorded frames: {len(self.tracker.tracking_data)}")
        
        imgui.text(f"Cameras: {len(self.tracker.serial_numbers)}")
        
        # Control buttons
        if imgui.button("Reset Tracking"):
            self.tracker.environment.reset()
            self.tracker.tracking_data.clear()
            self.tracker.start_time = time.time()
        
        imgui.same_line()
        if imgui.button("Save Results"):
            self.tracker.save_tracking_results()
        
        imgui.end()
    
    def _draw_gaussian_info(self):
        """Draw gaussian particle information"""
        imgui.begin("Gaussian Particles")
        
        sim = self.tracker.environment.sim
        num_gaussians = sim.gaussian_state.num_gaussians
        imgui.text(f"Total gaussians: {num_gaussians}")
        
        if num_gaussians > 0:
            with torch.no_grad():
                means = sim.gaussian_state.means
                colors = sim.gaussian_state.colors
                opacities = sim.gaussian_state.opacities
                
                # Show statistics
                mean_pos = means.mean(dim=0).cpu().numpy()
                imgui.text(f"Mean position: ({mean_pos[0]:.3f}, {mean_pos[1]:.3f}, {mean_pos[2]:.3f})")
                
                mean_opacity = opacities.mean().cpu().item()
                imgui.text(f"Mean opacity: {mean_opacity:.3f}")
                
                # Show first few gaussian positions
                imgui.text("First 5 gaussian positions:")
                for i in range(min(5, num_gaussians)):
                    pos = means[i].cpu().numpy()
                    imgui.text(f"  {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        imgui.end()
    
    def _draw_camera_info(self):
        """Draw camera information"""
        imgui.begin("Camera Info")
        
        imgui.text(f"Camera count: {len(self.tracker.serial_numbers)}")
        imgui.text(f"Frames initialized: {self.tracker.frames_initialized}")
        
        for i, serial in enumerate(self.tracker.serial_numbers):
            imgui.text(f"Camera {i}: {serial}")
            if serial in self.tracker.camera_intrinsics:
                K = self.tracker.camera_intrinsics[serial]
                imgui.text(f"  fx: {K[0,0]:.1f}, fy: {K[1,1]:.1f}")
                imgui.text(f"  cx: {K[0,2]:.1f}, cy: {K[1,2]:.1f}")
            
            # Show camera name used in frames
            if i < len(self.tracker.camera_names):
                imgui.text(f"  Frame name: {self.tracker.camera_names[i]}")
        
        if self.tracker.current_frames:
            imgui.text("Visual tracking: Active")
            imgui.text(f"Frame size: {self.tracker.current_frames.width}x{self.tracker.current_frames.height}")
            
            # Show last update time
            if self.tracker.start_time:
                current_time = time.time() - self.tracker.start_time
                imgui.text(f"Current time: {current_time:.2f}s")
        else:
            imgui.text("Visual tracking: Inactive")
        
        imgui.end()


async def main():
    params = tyro.cli(TrackingParams)
    
    # Validate input files
    if not params.body_json.exists():
        print(f"Error: Body JSON file not found: {params.body_json}")
        return
    
    if not params.extrinsics.exists():
        print(f"Error: Extrinsics file not found: {params.extrinsics}")
        return
    
    if params.ground and not params.ground.exists():
        print(f"Error: Ground file not found: {params.ground}")
        return
    
    # Initialize Warp
    wp.config.quiet = True
    wp.init()
    
    # Create tracker
    tracker = RealTimeTracker(params)
    
    if params.visualize:
        # Setup visualization
        visualizer = EmbodiedGUI()
        visualizer.set_environment(tracker.environment)
        
        # Add tracking GUI
        tracking_gui = TrackingGUI(tracker)
        visualizer.callbacks_render.append(tracking_gui.draw)
        
        # Run with visualization
        async with trio.open_nursery() as nursery:
            nursery.start_soon(tracker.track_with_cameras)
            await visualizer.run()
            nursery.cancel_scope.cancel()
    else:
        # Run without visualization
        await tracker.track_with_cameras()
    
    # Save results if requested
    if params.save_tracking_data:
        tracker.save_tracking_results()


if __name__ == "__main__":
    trio.run(main) 