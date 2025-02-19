import numpy as np
import sapien
from transforms3d.euler import euler2quat
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

from typing import Optional, Union
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actors.common import _build_by_type


def build_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    primary_color=np.array([194, 19, 22, 255]) / 255,
    secondary_color=np.array([255, 255, 255, 255]) / 255,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """
    Build a target with alternating colors
    
    Args:
        primary_color: Main color of the target (default: red)
        secondary_color: Alternating color of the target (default: white)
        ... (other args remain the same)
    """
    builder = scene.create_actor_builder()
    
    for i in range(5):
        current_radius = radius * (5 - i) / 5
        current_color = primary_color if i % 2 == 0 else secondary_color
        
        builder.add_cylinder_visual(
            radius=current_radius,
            half_length=thickness / 2 + i * 1e-5,
            material=sapien.render.RenderMaterial(base_color=current_color),
        )
        
        if add_collision:
            builder.add_cylinder_collision(
                radius=current_radius,
                half_length=thickness / 2 + i * 1e-5,
            )
            
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_pyramid(
    scene: ManiSkillScene,
    base_size: float,
    height: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates a pyramid shape using multiple boxes"""
    builder = scene.create_actor_builder()
    
    # Base box
    if add_collision:
        builder.add_box_collision(half_size=[base_size, base_size, height/4])
    builder.add_box_visual(
        half_size=[base_size, base_size, height/4],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    # Middle box
    middle_size = base_size * 0.7
    if add_collision:
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, height/2]),
            half_size=[middle_size, middle_size, height/4],
        )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, height/2]),
        half_size=[middle_size, middle_size, height/4],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    # Top box
    top_size = base_size * 0.4
    if add_collision:
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, height]),
            half_size=[top_size, top_size, height/4],
        )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, height]),
        half_size=[top_size, top_size, height/4],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_cross(
    scene: ManiSkillScene,
    arm_length: float,
    width: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates a cross shape using two boxes"""
    builder = scene.create_actor_builder()
    
    # Horizontal bar
    if add_collision:
        builder.add_box_collision(half_size=[arm_length, width/2, width/2])
    builder.add_box_visual(
        half_size=[arm_length, width/2, width/2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    # Vertical bar
    if add_collision:
        builder.add_box_collision(
            half_size=[width/2, arm_length, width/2],
        )
    builder.add_box_visual(
        half_size=[width/2, arm_length, width/2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_torus(
    scene: ManiSkillScene,
    radius: float,
    tube_radius: float,
    segments: int = 8,
    color = None,
    name: str = "torus",
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates a torus-like shape using multiple cylinders"""
    builder = scene.create_actor_builder()
    
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        next_angle = 2 * np.pi * (i + 1) / segments
        
        # Calculate positions for current segment
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        next_x = radius * np.cos(next_angle)
        next_y = radius * np.sin(next_angle)
        
        # Calculate segment properties
        center_x = (x + next_x) / 2
        center_y = (y + next_y) / 2
        segment_length = np.sqrt((next_x - x)**2 + (next_y - y)**2)
        rotation_angle = np.arctan2(next_y - y, next_x - x)
        
        # Create segment
        if add_collision:
            builder.add_cylinder_collision(
                pose=sapien.Pose(
                    p=[center_x, center_y, 0],
                    q=euler2quat(0, np.pi/2, rotation_angle),
                ),
                radius=tube_radius,
                half_length=segment_length/2,
            )
        builder.add_cylinder_visual(
            pose=sapien.Pose(
                p=[center_x, center_y, 0],
                q=euler2quat(0, np.pi/2, rotation_angle),
            ),
            radius=tube_radius,
            half_length=segment_length/2,
            material=sapien.render.RenderMaterial(base_color=color),
        )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_stairs(
    scene: ManiSkillScene,
    base_size: float,
    step_height: float,
    num_steps: int = 3,
    color = None,
    name: str = "stairs",
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates a stair-like shape using multiple boxes"""
    builder = scene.create_actor_builder()
    
    for i in range(num_steps):
        if add_collision:
            builder.add_box_collision(
                pose=sapien.Pose(p=[i * base_size/2, 0, i * step_height/2]),
                half_size=[base_size/2, base_size/2, (i+1) * step_height/2],
            )
        builder.add_box_visual(
            pose=sapien.Pose(p=[i * base_size/2, 0, i * step_height/2]),
            half_size=[base_size/2, base_size/2, (i+1) * step_height/2],
            material=sapien.render.RenderMaterial(base_color=color),
        )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_star(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    points: int = 5,
    color = None,
    name: str = "star",
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates a star shape using multiple boxes"""
    builder = scene.create_actor_builder()
    
    for i in range(points):
        angle = 2 * np.pi * i / points
        
        if add_collision:
            builder.add_box_collision(
                pose=sapien.Pose(
                    p=[radius * np.cos(angle)/2, radius * np.sin(angle)/2, 0],
                    q=euler2quat(0, 0, angle),
                ),
                half_size=[radius/2, thickness/2, thickness/2],
            )
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[radius * np.cos(angle)/2, radius * np.sin(angle)/2, 0],
                q=euler2quat(0, 0, angle),
            ),
            half_size=[radius/2, thickness/2, thickness/2],
            material=sapien.render.RenderMaterial(base_color=color),
        )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_helix(
    scene: ManiSkillScene,
    radius: float,
    height: float,
    thickness: float,
    turns: int = 2,
    segments_per_turn: int = 8,
    color = None,
    name: str = "helix",
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates a helical (spiral) shape using multiple cylinders"""
    builder = scene.create_actor_builder()
    
    total_segments = turns * segments_per_turn
    height_per_segment = height / total_segments
    
    for i in range(total_segments):
        angle = 2 * np.pi * i / segments_per_turn
        next_angle = 2 * np.pi * (i + 1) / segments_per_turn
        
        z = height * i / total_segments
        next_z = height * (i + 1) / total_segments
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        next_x = radius * np.cos(next_angle)
        next_y = radius * np.sin(next_angle)
        
        center_x = (x + next_x) / 2
        center_y = (y + next_y) / 2
        center_z = (z + next_z) / 2
        
        dx = next_x - x
        dy = next_y - y
        dz = next_z - z
        
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Calculate rotation to align cylinder with segment
        direction = np.array([dx, dy, dz])
        direction = direction / np.linalg.norm(direction)
        
        # Calculate rotation quaternion
        up = np.array([0, 0, 1])
        rotation_axis = np.cross(up, direction)
        if np.all(rotation_axis == 0):
            rotation_quat = [1, 0, 0, 0]
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.dot(up, direction))
            rotation_quat = [np.cos(angle/2)] + list(rotation_axis * np.sin(angle/2))
        
        if add_collision:
            builder.add_cylinder_collision(
                pose=sapien.Pose(
                    p=[center_x, center_y, center_z],
                    q=rotation_quat,
                ),
                radius=thickness,
                half_length=length/2,
            )
        builder.add_cylinder_visual(
            pose=sapien.Pose(
                p=[center_x, center_y, center_z],
                q=rotation_quat,
            ),
            radius=thickness,
            half_length=length/2,
            material=sapien.render.RenderMaterial(base_color=color),
        )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_arch(
    scene: ManiSkillScene,
    width: float,
    height: float,
    thickness: float,
    segments: int = 8,
    color = None,
    name: str = "arch",
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates an arch shape using cylinders for the curve and boxes for the pillars"""
    builder = scene.create_actor_builder()
    
    # Add pillars
    for x in [-width/2, width/2]:
        if add_collision:
            builder.add_box_collision(
                pose=sapien.Pose(p=[x, 0, height/2]),
                half_size=[thickness/2, thickness/2, height/2],
            )
        builder.add_box_visual(
            pose=sapien.Pose(p=[x, 0, height/2]),
            half_size=[thickness/2, thickness/2, height/2],
            material=sapien.render.RenderMaterial(base_color=color),
        )
    
    # Add curved top
    for i in range(segments):
        angle = np.pi * i / (segments-1)
        next_angle = np.pi * (i + 1) / (segments-1)
        
        x = width/2 * np.cos(angle)
        z = height + width/2 * np.sin(angle)
        next_x = width/2 * np.cos(next_angle)
        next_z = height + width/2 * np.sin(next_angle)
        
        center_x = (x + next_x) / 2
        center_z = (z + next_z) / 2
        
        segment_length = np.sqrt((next_x - x)**2 + (next_z - z)**2)
        rotation_angle = np.arctan2(next_z - z, next_x - x)
        
        if add_collision:
            builder.add_cylinder_collision(
                pose=sapien.Pose(
                    p=[center_x, 0, center_z],
                    q=euler2quat(0, np.pi/2, rotation_angle),
                ),
                radius=thickness/2,
                half_length=segment_length/2,
            )
        builder.add_cylinder_visual(
            pose=sapien.Pose(
                p=[center_x, 0, center_z],
                q=euler2quat(0, np.pi/2, rotation_angle),
            ),
            radius=thickness/2,
            half_length=segment_length/2,
            material=sapien.render.RenderMaterial(base_color=color),
        )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_crescent(
    scene: ManiSkillScene,
    outer_radius: float,
    thickness: float,
    height: float,
    segments: int = 12,
    color = None,
    name: str = "crescent",
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates a crescent moon shape by subtracting two cylinders"""
    builder = scene.create_actor_builder()
    
    inner_radius = outer_radius - thickness
    
    # Create segments for the outer arc
    for i in range(segments):
        angle = np.pi * i / (segments-1)
        next_angle = np.pi * (i + 1) / (segments-1)
        
        # Calculate positions for outer arc
        x = outer_radius * np.cos(angle)
        y = outer_radius * np.sin(angle)
        next_x = outer_radius * np.cos(next_angle)
        next_y = outer_radius * np.sin(next_angle)
        
        # Calculate segment properties
        center_x = (x + next_x) / 2
        center_y = (y + next_y) / 2
        segment_length = np.sqrt((next_x - x)**2 + (next_y - y)**2)
        rotation_angle = np.arctan2(next_y - y, next_x - x)
        
        # Create outer arc segment
        if add_collision:
            builder.add_cylinder_collision(
                pose=sapien.Pose(
                    p=[center_x, center_y, 0],
                    q=euler2quat(0, np.pi/2, rotation_angle),
                ),
                radius=height/2,
                half_length=segment_length/2,
            )
        builder.add_cylinder_visual(
            pose=sapien.Pose(
                p=[center_x, center_y, 0],
                q=euler2quat(0, np.pi/2, rotation_angle),
            ),
            radius=height/2,
            half_length=segment_length/2,
            material=sapien.render.RenderMaterial(base_color=color),
        )
    
    # Create segments for the inner arc (shifted to create crescent shape)
    offset_x = thickness * 0.7  # Shift the inner circle to create crescent shape
    for i in range(segments):
        angle = np.pi * i / (segments-1)
        next_angle = np.pi * (i + 1) / (segments-1)
        
        # Calculate positions for inner arc
        x = inner_radius * np.cos(angle) + offset_x
        y = inner_radius * np.sin(angle)
        next_x = inner_radius * np.cos(next_angle) + offset_x
        next_y = inner_radius * np.sin(next_angle)
        
        # Calculate segment properties
        center_x = (x + next_x) / 2
        center_y = (y + next_y) / 2
        segment_length = np.sqrt((next_x - x)**2 + (next_y - y)**2)
        rotation_angle = np.arctan2(next_y - y, next_x - x)
        
        # Create inner arc segment (using negative space)
        if add_collision:
            builder.add_cylinder_collision(
                pose=sapien.Pose(
                    p=[center_x, center_y, 0],
                    q=euler2quat(0, np.pi/2, rotation_angle),
                ),
                radius=height/2,
                half_length=segment_length/2,
            )
        builder.add_cylinder_visual(
            pose=sapien.Pose(
                p=[center_x, center_y, 0],
                q=euler2quat(0, np.pi/2, rotation_angle),
            ),
            radius=height/2,
            half_length=segment_length/2,
            material=sapien.render.RenderMaterial(base_color=[0, 0, 0, 0]),  # Transparent
        )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

# def build_t_shape(
#     scene: ManiSkillScene,
#     width: float,
#     height: float,
#     thickness: float,
#     color = None,
#     name: str = "t_shape",
#     body_type: str = "dynamic",
#     add_collision: bool = True,
#     scene_idxs: Optional[Array] = None,
#     initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
# ):
#     """Creates a T-shaped object using boxes"""
#     builder = scene.create_actor_builder()
    
#     # Horizontal bar
#     if add_collision:
#         builder.add_box_collision(
#             pose=sapien.Pose(p=[0, 0, height - thickness/2]),
#             half_size=[width/2, thickness/2, thickness/2],
#         )
#     builder.add_box_visual(
#         pose=sapien.Pose(p=[0, 0, height - thickness/2]),
#         half_size=[width/2, thickness/2, thickness/2],
#         material=sapien.render.RenderMaterial(base_color=color),
#     )
    
#     # Vertical bar
#     if add_collision:
#         builder.add_box_collision(
#             pose=sapien.Pose(p=[0, 0, height/2]),
#             half_size=[thickness/2, thickness/2, height/2],
#         )
#     builder.add_box_visual(
#         pose=sapien.Pose(p=[0, 0, height/2]),
#         half_size=[thickness/2, thickness/2, height/2],
#         material=sapien.render.RenderMaterial(base_color=color),
#     )
    
#     return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_t_shape(
    scene: ManiSkillScene,
    width: float,
    height: float,
    thickness: float,
    color = None,
    name: str = "t_shape",
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates a T-shaped object using boxes, lying flat"""
    builder = scene.create_actor_builder()
    
    # Horizontal bar
    if add_collision:
        builder.add_box_collision(
            pose=sapien.Pose(
                p=[0, height - thickness/2, 0],
                q=euler2quat(np.pi/2, 0, 0)  # Rotate 90 degrees around X axis
            ),
            half_size=[width/2, thickness/2, thickness/2],
        )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[0, height - thickness/2, 0],
            q=euler2quat(np.pi/2, 0, 0)  # Rotate 90 degrees around X axis
        ),
        half_size=[width/2, thickness/2, thickness/2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    # Vertical bar
    if add_collision:
        builder.add_box_collision(
            pose=sapien.Pose(
                p=[0, height/2, 0],
                q=euler2quat(np.pi/2, 0, 0)  # Rotate 90 degrees around X axis
            ),
            half_size=[thickness/2, thickness/2, height/2],
        )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[0, height/2, 0],
            q=euler2quat(np.pi/2, 0, 0)  # Rotate 90 degrees around X axis
        ),
        half_size=[thickness/2, thickness/2, height/2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

# def build_l_shape(
#     scene: ManiSkillScene,
#     width: float,
#     height: float,
#     thickness: float,
#     color = None,
#     name: str = "l_shape",
#     body_type: str = "dynamic",
#     add_collision: bool = True,
#     scene_idxs: Optional[Array] = None,
#     initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
# ):
#     """Creates an L-shaped object using boxes"""
#     builder = scene.create_actor_builder()
    
#     # Vertical bar
#     if add_collision:
#         builder.add_box_collision(
#             pose=sapien.Pose(p=[0, 0, height/2]),
#             half_size=[thickness/2, thickness/2, height/2],
#         )
#     builder.add_box_visual(
#         pose=sapien.Pose(p=[0, 0, height/2]),
#         half_size=[thickness/2, thickness/2, height/2],
#         material=sapien.render.RenderMaterial(base_color=color),
#     )
    
#     # Horizontal bar
#     if add_collision:
#         builder.add_box_collision(
#             pose=sapien.Pose(p=[width/2, 0, thickness/2]),
#             half_size=[width/2, thickness/2, thickness/2],
#         )
#     builder.add_box_visual(
#         pose=sapien.Pose(p=[width/2, 0, thickness/2]),
#         half_size=[width/2, thickness/2, thickness/2],
#         material=sapien.render.RenderMaterial(base_color=color),
#     )
    
#     return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_l_shape(
    scene: ManiSkillScene,
    width: float,
    height: float,
    thickness: float,
    color = None,
    name: str = "l_shape",
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates an L-shaped object using boxes, lying flat"""
    builder = scene.create_actor_builder()
    
    # Vertical bar
    if add_collision:
        builder.add_box_collision(
            pose=sapien.Pose(
                p=[0, height/2, 0],
                q=euler2quat(np.pi/2, 0, 0)  # Rotate 90 degrees around X axis
            ),
            half_size=[thickness/2, thickness/2, height/2],
        )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[0, height/2, 0],
            q=euler2quat(np.pi/2, 0, 0)  # Rotate 90 degrees around X axis
        ),
        half_size=[thickness/2, thickness/2, height/2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    # Horizontal bar
    if add_collision:
        builder.add_box_collision(
            pose=sapien.Pose(
                p=[width/2, thickness/2, 0],
                q=euler2quat(np.pi/2, 0, 0)  # Rotate 90 degrees around X axis
            ),
            half_size=[width/2, thickness/2, thickness/2],
        )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[width/2, thickness/2, 0],
            q=euler2quat(np.pi/2, 0, 0)  # Rotate 90 degrees around X axis
        ),
        half_size=[width/2, thickness/2, thickness/2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_arrow(
    scene: ManiSkillScene,
    length: float,
    head_size: float,
    thickness: float,
    color = None,
    name: str = "arrow",
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates an arrow shape using boxes"""
    builder = scene.create_actor_builder()
    
    # Shaft
    if add_collision:
        builder.add_box_collision(
            pose=sapien.Pose(p=[-length/4, 0, 0]),
            half_size=[length/4, thickness/2, thickness/2],
        )
    builder.add_box_visual(
        pose=sapien.Pose(p=[-length/4, 0, 0]),
        half_size=[length/4, thickness/2, thickness/2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    # Arrow head
    for angle in [-np.pi/4, np.pi/4]:
        if add_collision:
            builder.add_box_collision(
                pose=sapien.Pose(
                    p=[0, head_size/2 * np.sin(angle), 0],
                    q=euler2quat(0, 0, angle),
                ),
                half_size=[head_size/2, thickness/2, thickness/2],
            )
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[0, head_size/2 * np.sin(angle), 0],
                q=euler2quat(0, 0, angle),
            ),
            half_size=[head_size/2, thickness/2, thickness/2],
            material=sapien.render.RenderMaterial(base_color=color),
        )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_y_shape(
    scene: ManiSkillScene,
    width: float,
    height: float,
    thickness: float,
    color = None,
    name: str = "y_shape",
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Creates a Y-shaped object using boxes, lying flat"""
    builder = scene.create_actor_builder()
    
    # Vertical stem
    if add_collision:
        builder.add_box_collision(
            pose=sapien.Pose(
                p=[height/2, 0, 0],
                q=euler2quat(np.pi/2, 0, 0)  # Rotate 90 degrees around X axis
            ),
            half_size=[height/2, thickness/2, thickness/2],
        )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[height/2, 0, 0],
            q=euler2quat(np.pi/2, 0, 0)  # Rotate 90 degrees around X axis
        ),
        half_size=[height/2, thickness/2, thickness/2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    # Left diagonal arm
    angle = np.pi/4  # 45 degrees
    arm_length = width/2
    if add_collision:
        builder.add_box_collision(
            pose=sapien.Pose(
                p=[0, arm_length/2 * np.sin(angle), 0],
                q=euler2quat(np.pi/2, 0, angle)  # Rotate around X and Z axes
            ),
            half_size=[arm_length/2, thickness/2, thickness/2],
        )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[0, arm_length/2 * np.sin(angle), 0],
            q=euler2quat(np.pi/2, 0, angle)  # Rotate around X and Z axes
        ),
        half_size=[arm_length/2, thickness/2, thickness/2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    # Right diagonal arm
    if add_collision:
        builder.add_box_collision(
            pose=sapien.Pose(
                p=[0, arm_length/2 * np.sin(-angle), 0],
                q=euler2quat(np.pi/2, 0, -angle)  # Rotate around X and negative Z axes
            ),
            half_size=[arm_length/2, thickness/2, thickness/2],
        )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[0, arm_length/2 * np.sin(-angle), 0],
            q=euler2quat(np.pi/2, 0, -angle)  # Rotate around X and negative Z axes
        ),
        half_size=[arm_length/2, thickness/2, thickness/2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_diagonal_half_cube(
    scene: ManiSkillScene,
    half_size: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
    upper_half: bool = True,
    angle_degrees: float = 45,
):
    builder = scene.create_actor_builder()
    
    n_steps = 5
    size = half_size * 2
    angle_rad = np.radians(angle_degrees)
    
    for i in range(n_steps):
        step_size = size / n_steps
        if upper_half:
            base_height = size * (i + 1) / n_steps
            y_offset = size/2 - step_size * (n_steps - i - 1)
        else:
            base_height = size * (n_steps - i) / n_steps
            y_offset = -size/2 + step_size * i

        # Adjust height based on angle
        height = base_height * np.sin(angle_rad)
        z_pos = height/2
        
        if add_collision:
            builder.add_box_collision(
                half_size=[size/2, step_size/2, height/2],
                pose=sapien.Pose(p=[0, y_offset, z_pos]),
                material=sapien.pysapien.physx.PhysxMaterial(
                    static_friction=0.5,
                    dynamic_friction=0.5,
                    restitution=0.5
                )
            )

        builder.add_box_visual(
            half_size=[size/2, step_size/2, height/2],
            pose=sapien.Pose(p=[0, y_offset, z_pos]),
            material=sapien.render.RenderMaterial(
                base_color=color,
            ),
        )

    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

            # self.cubes[key] = actors.build_cube(
            #     self.scene,
            #     half_size=self.CUBE_HALFSIZE,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            # )

            # self.cubes[key] = shapes.build_pyramid(
            #     self.scene,
            #     base_size=self.CUBE_HALFSIZE,
            #     height=self.CUBE_HALFSIZE,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            # )

            # self.cubes[key] = shapes.build_cross(
            #     self.scene,
            #     arm_length=self.CUBE_HALFSIZE,
            #     width=self.CUBE_HALFSIZE/2,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            # )

            # self.cubes[key] = shapes.build_torus(
            #     self.scene,
            #     radius=self.CUBE_HALFSIZE,
            #     tube_radius=self.CUBE_HALFSIZE/2,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            # )

            # self.cubes[key] = shapes.build_star(
            #     self.scene,
            #     radius=self.CUBE_HALFSIZE,
            #     thickness=self.CUBE_HALFSIZE/2,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            # )

            # self.cubes[key] = shapes.build_helix(
            #     self.scene,
            #     radius=self.CUBE_HALFSIZE,
            #     height=self.CUBE_HALFSIZE,
            #     thickness=self.CUBE_HALFSIZE/4,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            # )

            # self.cubes[key] = shapes.build_arch(
            #     self.scene,
            #     width=self.CUBE_HALFSIZE,
            #     height=self.CUBE_HALFSIZE,
            #     thickness=self.CUBE_HALFSIZE/2,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            # )

            # self.cubes[key] = shapes.build_crescent(
            #     self.scene,
            #     outer_radius=self.CUBE_HALFSIZE,
            #     height=self.CUBE_HALFSIZE,
            #     thickness=self.CUBE_HALFSIZE/2,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            # )

            # self.cubes[key] = shapes.build_t_shape(
            #     self.scene,
            #     width=self.CUBE_HALFSIZE,
            #     height=self.CUBE_HALFSIZE*1.5,
            #     thickness=self.CUBE_HALFSIZE/2,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE], q=euler2quat(0, np.pi / 2, 0)),
            # )

            # self.cubes[key] = shapes.build_l_shape(
            #     self.scene,
            #     width=self.CUBE_HALFSIZE*1.25,
            #     height=self.CUBE_HALFSIZE*1.5,
            #     thickness=self.CUBE_HALFSIZE/2,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            # )

            # self.cubes[key] = shapes.build_arrow(
            #     self.scene,
            #     length=self.CUBE_HALFSIZE*5,
            #     head_size=self.CUBE_HALFSIZE*1,
            #     thickness=self.CUBE_HALFSIZE/2,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            # )
            
            # self.cubes[key] = shapes.build_y_shape(
            #     self.scene,
            #     width=self.CUBE_HALFSIZE*5,
            #     height=self.CUBE_HALFSIZE*5,
            #     thickness=self.CUBE_HALFSIZE/2,
            #     color=color,
            #     name=f"cube_{key}",
            #     body_type="dynamic",
            #     initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            # )

            # cylinder

            # sphere