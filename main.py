import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xml.etree.ElementTree as ET
import importlib.util
from mpl_toolkits.mplot3d import Axes3D

# Import BVH modules directly from file paths
bvh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bvh', 'bvh.py')
bvhvisualize_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bvh', 'bvhvisualize.py')

# Import BVH module
spec = importlib.util.spec_from_file_location('bvh_module', bvh_path)
bvh_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bvh_module)
BVH = bvh_module.BVH

# Import BVHVisualize module
spec = importlib.util.spec_from_file_location('bvhvisualize_module', bvhvisualize_path)
bvhvisualize_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bvhvisualize_module)
BVHAnimator = bvhvisualize_module.BVHAnimator
BVHVisualizeFrame = bvhvisualize_module.BVHVisualizeFrame


class GO1Model:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.joints = {}
        self.links = {}
        self.parse_urdf()

    def parse_urdf(self):
        """Parse GO1 URDF file to extract joint and link information"""
        print("Parsing GO1 URDF file...")
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        # Extract joints
        for joint in root.findall(".//joint"):
            joint_name = joint.get("name")

            # Skip rotor joints as they're not part of the kinematic chain
            if "rotor" in joint_name:
                continue

            joint_type = joint.get("type")
            origin = joint.find("origin")

            if origin is not None:
                xyz = origin.get("xyz")
                rpy = origin.get("rpy")
                xyz = [float(v) for v in xyz.split()]
                rpy = [float(v) for v in rpy.split()] if rpy else [0, 0, 0]
            else:
                xyz = [0, 0, 0]
                rpy = [0, 0, 0]

            parent = joint.find("parent")
            child = joint.find("child")

            if parent is not None and child is not None:
                parent_link = parent.get("link")
                child_link = child.get("link")
            else:
                parent_link = None
                child_link = None

            self.joints[joint_name] = {
                "type": joint_type,
                "xyz": xyz,
                "rpy": rpy,
                "parent": parent_link,
                "child": child_link,
            }

        # Extract links
        for link in root.findall(".//link"):
            link_name = link.get("name")
            visual = link.find("visual")

            if visual is not None:
                origin = visual.find("origin")
                if origin is not None:
                    xyz = origin.get("xyz")
                    rpy = origin.get("rpy")
                    xyz = [float(v) for v in xyz.split()]
                    rpy = [float(v) for v in rpy.split()] if rpy else [0, 0, 0]
                else:
                    xyz = [0, 0, 0]
                    rpy = [0, 0, 0]
            else:
                xyz = [0, 0, 0]
                rpy = [0, 0, 0]

            self.links[link_name] = {"xyz": xyz, "rpy": rpy}

    def convert_coords_xyz_to_zxy(self):
        """Convert GO1 coordinates from xyz to zxy"""
        print("Converting coordinates from xyz to zxy...")

        # Conversion for joints
        for joint_name, joint in self.joints.items():
            x, y, z = joint["xyz"]
            # Swap x with z to convert from xyz to zxy
            joint["xyz"] = [z, x, y]

            rx, ry, rz = joint["rpy"]
            # Adjust rotation angles for coordinate swap
            joint["rpy"] = [rz, rx, ry]

        # Conversion for links
        for link_name, link in self.links.items():
            x, y, z = link["xyz"]
            # Swap x with z to convert from xyz to zxy
            link["xyz"] = [z, x, y]

            rx, ry, rz = link["rpy"]
            # Adjust rotation angles for coordinate swap
            link["rpy"] = [rz, rx, ry]

    def calculate_link_lengths(self):
        """Calculate link lengths based on joint positions"""
        print("Calculating link lengths...")

        self.link_lengths = {}

        # Build a tree structure to calculate link lengths
        children = {}
        for joint_name, joint in self.joints.items():
            parent = joint["parent"]
            child = joint["child"]

            if parent and child:
                if parent not in children:
                    children[parent] = []
                children[parent].append((child, joint_name, joint["xyz"]))

        # Recursively calculate link lengths starting from trunk
        self._calculate_link_length_recursive("trunk", children, [0, 0, 0])

    def _calculate_link_length_recursive(self, link_name, children, parent_pos):
        if link_name not in children:
            return

        for child_link, joint_name, joint_pos in children[link_name]:
            # Calculate absolute position
            child_pos = [parent_pos[i] + joint_pos[i] for i in range(3)]

            # Calculate link length (distance between joints)
            length = np.sqrt(sum((joint_pos[i]) ** 2 for i in range(3)))

            self.link_lengths[joint_name] = length

            # Continue recursion
            self._calculate_link_length_recursive(child_link, children, child_pos)


class GO1Retargeter:
    def __init__(self, go1_model, bvh_file):
        self.go1 = go1_model
        self.bvh = BVH()
        print(f"Loading BVH file: {bvh_file}")
        self.bvh.load(bvh_file)
        self.retargeted_frames = []

    def match_end_effector_positions(self):
        """Match the end effector positions between GO1 and BVH"""
        print("Matching end effector positions...")
        
        # Read the first frame to get BVH skeleton information
        self.bvh.readFrame(0)
        
        # Print available joints for debugging
        print("Available joints in BVH file:")
        for i in range(self.bvh.numJoints()):
            joint = self.bvh.jointById(i)
            print(f"  {i}: {joint.name}")
        
        # Determine BVH structure
        # Extract root and end effector joints
        # In GO1 model, the end effectors are the foot joints
        self.end_effectors = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        
        # Try to map BVH joints to GO1 joints based on available names
        self.bvh_end_effectors = []
        
        # Check if typical quadruped joints exist, otherwise use human-like mapping
        quad_foot_exists = False
        try:
            quad_foot_exists = self.bvh.jointByName('FR_foot') is not None
        except:
            quad_foot_exists = False
            
        if quad_foot_exists:
            # Quadruped mapping - direct mapping
            self.bvh_end_effectors = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
            print("Using quadruped joint mapping")
        else:
            # Human-like mapping (assuming human BVH)
            # Try to find foot joints in human skeleton
            potential_foot_joints = ['RightFoot', 'LeftFoot', 'RightToe', 'LeftToe', 
                                     'RightAnkle', 'LeftAnkle', 'rFoot', 'lFoot', 'RightLeg', 'LeftLeg']
            foot_joints = []
            
            for joint_name in potential_foot_joints:
                try:
                    if self.bvh.jointByName(joint_name) is not None:
                        foot_joints.append(joint_name)
                except:
                    continue
            
            # If we found at least 2 foot joints, we can use them
            if len(foot_joints) >= 2:
                # Map to quadruped legs (front legs use human arms, rear legs use human legs)
                
                # Try to find arm/hand joints
                potential_hand_joints = ['RightHand', 'LeftHand', 'RightWrist', 'LeftWrist', 
                                        'rHand', 'lHand', 'RightArm', 'LeftArm']
                hand_joints = []
                
                for joint_name in potential_hand_joints:
                    try:
                        if self.bvh.jointByName(joint_name) is not None:
                            hand_joints.append(joint_name)
                    except:
                        continue
                
                # If we have both hands and feet, map GO1 legs to human limbs
                if len(hand_joints) >= 2:
                    # Front right leg -> Right arm
                    # Front left leg -> Left arm
                    # Rear right leg -> Right leg
                    # Rear left leg -> Left leg
                    self.bvh_end_effectors = [hand_joints[0], hand_joints[1], 
                                             foot_joints[0], foot_joints[1]]
                    print(f"Using human-like joint mapping: {self.bvh_end_effectors}")
                else:
                    # Just map all GO1 legs to human legs
                    print("Could not find hand joints - mapping all GO1 legs to human legs")
                    if len(foot_joints) >= 4:
                        self.bvh_end_effectors = foot_joints[:4]
                    else:
                        # Duplicate the found foot joints to fill all 4 slots
                        self.bvh_end_effectors = foot_joints * (4 // len(foot_joints) + 1)
                        self.bvh_end_effectors = self.bvh_end_effectors[:4]
            else:
                # Fallback - just use the first 4 joints as end effectors
                print("Could not find appropriate end effector joints - using first 4 joints")
                self.bvh_end_effectors = []
                count = 0
                
                for i in range(self.bvh.numJoints()):
                    joint = self.bvh.jointById(i)
                    if not joint.children:  # This is an end site
                        self.bvh_end_effectors.append(joint.name)
                        count += 1
                        if count >= 4:
                            break
                
                # If we still don't have 4 joints, use any joints
                while len(self.bvh_end_effectors) < 4 and self.bvh.numJoints() > 0:
                    for i in range(self.bvh.numJoints()):
                        joint = self.bvh.jointById(i)
                        if joint.name not in self.bvh_end_effectors:
                            self.bvh_end_effectors.append(joint.name)
                            if len(self.bvh_end_effectors) >= 4:
                                break
        
        print(f"Mapped end effectors: {self.bvh_end_effectors}")

    def compute_inverse_kinematics(self):
        """Compute inverse kinematics to retarget BVH motion to GO1 robot, focusing only on leg end-effectors"""
        print("Computing inverse kinematics...")
        
        # For faster development
        num_frames = min(200, self.bvh.numFrames())
        
        print(f"Processing {self.bvh.numFrames()} frames...")
        self.retargeted_frames = []
        
        # Extract fixed link lengths from URDF for all legs
        leg_info = {}
        for prefix in ['FR', 'FL', 'RR', 'RL']:
            hip_joint = f"{prefix}_hip_joint"
            thigh_joint = f"{prefix}_thigh_joint"
            calf_joint = f"{prefix}_calf_joint"
            
            # Get link lengths directly from URDF data
            hip_length = self.go1.link_lengths.get(hip_joint, 0.08) * 1000
            thigh_length = self.go1.link_lengths.get(thigh_joint, 0.20) * 1000
            calf_length = self.go1.link_lengths.get(calf_joint, 0.20) * 1000
            
            leg_info[prefix] = {
                'hip_joint': hip_joint,
                'thigh_joint': thigh_joint,
                'calf_joint': calf_joint,
                # Remove hip_length, it's just an offset
                'thigh_length': thigh_length,
                'calf_length': calf_length,
                # Removed total_leg_length, calculate reach dynamically
                # Get joint positions from URDF, scaled to mm
                'hip_offset': [val * 1000 for val in self.go1.joints[hip_joint]['xyz']] if hip_joint in self.go1.joints else [0, 0, 0]
            }
        
        for frame in range(num_frames):
            if frame % 10 == 0:
                print(f"Processing frame {frame}/{num_frames}")
            
            # Read BVH frame
            self.bvh.readFrame(frame)
            
            # Initialize retargeted pose
            retargeted_pose = {}
            
            # Find the root joint in BVH
            root_joint = None
            for i in range(self.bvh.numJoints()):
                joint = self.bvh.jointById(i)
                if joint.parent is None or 'hip' in joint.name.lower() or 'root' in joint.name.lower():
                    root_joint = joint
                    break
            
            # Check if root joint is found
            if root_joint is None:
                print("Error: Could not find root joint in BVH file.")
                continue
            
            # Get root position
            root_pos = root_joint.globalPos()
            retargeted_pose['root_pos'] = root_pos
            retargeted_pose['root_rot'] = [0, 0, 0]  # Add initial root rotation
            
            # Get end effector positions
            ee_positions = []
            for i, ee_name in enumerate(self.bvh_end_effectors):
                if i >= len(self.end_effectors):
                    break
                    
                try:
                    bvh_joint = self.bvh.jointByName(ee_name)
                    if bvh_joint:
                        ee_pos = bvh_joint.globalPos()
                        ee_positions.append(ee_pos)
                    else:
                        ee_positions.append([0.0, 0.0, 0.0])
                except:
                    ee_positions.append([0.0, 0.0, 0.0])
            
            # Map each BVH end effector to a GO1 leg
            for i, ee_name in enumerate(self.end_effectors):
                if i >= len(ee_positions):
                    break
                    
                # Extract the leg prefix (e.g., 'FR' from 'FR_foot')
                prefix = ee_name.split('_')[0]
                if prefix not in leg_info:
                    continue
                    
                # Get leg joint data
                leg_data = leg_info[prefix]
                hip_joint = leg_data['hip_joint']
                thigh_joint = leg_data['thigh_joint']
                calf_joint = leg_data['calf_joint']
                
                # Get fixed link lengths for this leg
                thigh_length = leg_data['thigh_length']
                calf_length = leg_data['calf_length']
                
                # Get hip position relative to root (using BVH root + URDF offset)
                hip_pos_abs = root_pos.copy()
                hip_offset_rel = leg_data['hip_offset']
                hip_pos_abs = [hip_pos_abs[j] + hip_offset_rel[j] for j in range(3)]
                
                # Get target end effector position from BVH (absolute)
                target_ee_pos_abs = ee_positions[i]
                
                # Compute vector from absolute hip position to absolute target end effector position
                hip_to_ee_vec = [target_ee_pos_abs[j] - hip_pos_abs[j] for j in range(3)]
                
                # Target position relative to the hip joint coordinate system
                # GO1 URDF: X forward, Y left, Z up
                # BVH might be different, assume same for now (X forward, Y left, Z up)
                x = hip_to_ee_vec[0]
                y = hip_to_ee_vec[1]
                z = hip_to_ee_vec[2]

                # Default angles
                hip_roll_angle = 0.0  # Angle for *_hip_joint (around X)
                hip_pitch_angle = 0.0 # Angle for *_thigh_joint (around Y)
                knee_pitch_angle = 0.0 # Angle for *_calf_joint (around Y)

                # --- 3-DOF IK Calculation ---
                # 1. Calculate Hip Roll (Abduction/Adduction) - Rotation around X axis
                # Projects target onto YZ plane relative to hip
                hip_roll_angle = np.arctan2(-y, z) # atan2(opposite, adjacent) -> angle from Z axis towards -Y

                # Clamp roll angle based on URDF limits if available (approx +- 49.4 deg)
                # limit +/- 0.863 radians
                hip_limit = 0.863
                hip_roll_angle = np.clip(hip_roll_angle, -hip_limit, hip_limit)

                # 2. Transform target point into the leg's sagittal plane (after roll)
                # Effective distance from hip Y-axis (in the rolled plane)
                d1 = np.sqrt(y**2 + z**2) # Distance in the YZ plane
                # Project the target onto the new Z' axis (aligned with the hip link after roll)
                z_prime = d1 * np.cos(hip_roll_angle) # Should be positive if target is below hip
                # Use the original X coordinate
                x_prime = x
                
                # Target coordinates in the X-Z' plane relative to the hip pitch joint
                target_dist_sq = x_prime**2 + z_prime**2
                target_dist = np.sqrt(target_dist_sq)

                # 3. Calculate Knee Pitch - Using Law of Cosines in the X-Z' plane
                # Angle between thigh and calf links
                max_reach = thigh_length + calf_length
                min_reach = abs(thigh_length - calf_length) # Simplification, actual min reach depends on joint limits

                if target_dist >= max_reach:
                    # Target is too far or exactly at max reach, straighten leg
                    knee_pitch_angle = 0.0
                    target_dist = max_reach # Clamp distance for pitch calculation
                elif target_dist <= min_reach:
                     # Target is too close, bend knee maximally (use URDF limit if possible)
                     # limit [-2.818, -0.888] radians => bend is negative
                     knee_pitch_angle = -0.888 # Max flexion allowed by lower limit
                     target_dist = min_reach # Clamp distance
                else:
                    # Use law of cosines: target^2 = thigh^2 + calf^2 - 2*thigh*calf*cos(pi - knee_angle)
                    # cos(pi - knee_angle) = -cos(knee_angle)
                    # cos(knee_angle) = (thigh^2 + calf^2 - target^2) / (2 * thigh * calf)
                    cos_knee_angle_arg = (thigh_length**2 + calf_length**2 - target_dist_sq) / (2 * thigh_length * calf_length)
                    cos_knee_angle_arg = np.clip(cos_knee_angle_arg, -1.0, 1.0)
                    knee_angle_positive = np.arccos(cos_knee_angle_arg) # Angle between 0 and pi

                    # Knee angle (*_calf_joint) in URDF is negative for flexion (bending inwards)
                    knee_pitch_angle = -knee_angle_positive

                    # Clamp knee angle based on URDF limits [-2.818, -0.888]
                    knee_pitch_angle = np.clip(knee_pitch_angle, -2.818, -0.888)


                # 4. Calculate Hip Pitch - Angle of the thigh link in the X-Z' plane
                # Angle of target vector in X-Z' plane + angle between target vector and thigh link
                
                # Angle of the target vector relative to the Z' axis (vertical down)
                angle_target = np.arctan2(x_prime, z_prime) # atan2(opposite=x', adjacent=z')

                # Angle between target vector and thigh link (alpha in law of cosines notation)
                # target^2 = thigh^2 + calf^2 - 2*thigh*calf*cos(knee_angle_complement)
                # calf^2 = thigh^2 + target^2 - 2*thigh*target*cos(angle_alpha)
                cos_alpha_arg = (thigh_length**2 + target_dist_sq - calf_length**2) / (2 * thigh_length * target_dist)
                cos_alpha_arg = np.clip(cos_alpha_arg, -1.0, 1.0)
                angle_alpha = np.arccos(cos_alpha_arg)

                # Hip pitch angle (*_thigh_joint)
                hip_pitch_angle = angle_target - angle_alpha

                # Adjust hip pitch based on joint definition (URDF Y-axis)
                # Clamp hip pitch based on URDF limits [-0.686, 4.501]
                hip_pitch_angle = np.clip(hip_pitch_angle, -0.686, 4.501)

                # Store computed joint angles in the retargeted pose
                retargeted_pose[hip_joint] = hip_roll_angle   # Corresponds to FR_hip_joint etc. (X-axis rotation)
                retargeted_pose[thigh_joint] = hip_pitch_angle # Corresponds to FR_thigh_joint etc. (Y-axis rotation)
                retargeted_pose[calf_joint] = knee_pitch_angle # Corresponds to FR_calf_joint etc. (Y-axis rotation)
                
            # Store retargeted pose for this frame
            self.retargeted_frames.append(retargeted_pose)

    def visualize_comparison(self):
        """Visualize side-by-side comparison of BVH and retargeted GO1"""
        print("Visualizing side-by-side comparison...")
        
        if not self.retargeted_frames:
            print("No retargeted frames to visualize.")
            return
        
        # Create 2D figure and subplots for side view with more space
        fig = plt.figure(figsize=(20, 10))  # Increased height for better spacing
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Set up plot details with larger fonts
        plt.rcParams.update({'font.size': 12})  # Increase default font size
        
        # Add more space at the bottom and left for axis labels
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.9, top=0.9)
        
        # Define leg positions based on URDF data
        self.leg_positions = {}
        
        # Get hip joint offsets from URDF for each leg
        for prefix in ['FR', 'FL', 'RR', 'RL']:
            hip_joint = f"{prefix}_hip_joint"
            if hip_joint in self.go1.joints:
                # Get xyz offset from URDF and apply scale consistently
                self.leg_positions[prefix] = [val * 1000 for val in self.go1.joints[hip_joint]['xyz']]
            else:
                # Default values if not found in URDF
                if prefix == 'FR':
                    self.leg_positions[prefix] = [0.1834 * 1000, -0.047 * 1000, 0]  # Front Right
                elif prefix == 'FL':
                    self.leg_positions[prefix] = [0.1834 * 1000, 0.047 * 1000, 0]   # Front Left
                elif prefix == 'RR':
                    self.leg_positions[prefix] = [-0.1834 * 1000, -0.047 * 1000, 0] # Rear Right
                else:  # RL
                    self.leg_positions[prefix] = [-0.1834 * 1000, 0.047 * 1000, 0]  # Rear Left
        
        # Set default plot limits for each panel separately
        bvh_plot_limits = {
            'z_min': -1.0, 'z_max': 1.0,
            'y_min': -1.0, 'y_max': 1.0
        }
        
        go1_plot_limits = {
            'x_min': -1.0, 'x_max': 1.0,
            'z_min': -1.0, 'z_max': 1.0
        }
        
        # Build GO1 kinematic tree from URDF data
        self.build_kinematic_tree()
        
        # Global variable for debugging
        global frame
        
        # Function to update animation
        def update(frame_idx):
            global frame
            frame = frame_idx
            
            ax1.clear()
            ax2.clear()
            
            # Set up plot details again with larger fonts
            ax1.set_title("Original BVH Motion (Z-Y View)", pad=20, fontsize=14)
            ax2.set_title("Retargeted GO1 Motion (X-Z View)", pad=20, fontsize=14)
            
            # Set axis labels with more padding and larger font
            ax1.set_xlabel('Z', labelpad=15, fontsize=12)
            ax1.set_ylabel('Y', labelpad=15, fontsize=12)
            ax2.set_xlabel('X', labelpad=15, fontsize=12)
            ax2.set_ylabel('Z', labelpad=15, fontsize=12)
            
            # Increase tick label size and add padding
            ax1.tick_params(axis='both', which='major', labelsize=12, pad=8)
            ax2.tick_params(axis='both', which='major', labelsize=12, pad=8)
            
            # Read BVH frame
            self.bvh.readFrame(frame)
            
            # Plot BVH skeleton as 2D side view (Z-Y plane)
            num_joints = self.bvh.numJoints()
            bvh_z_coords = []
            bvh_y_coords = []
            
            # Draw the BVH skeleton
            for i in range(num_joints):
                joint = self.bvh.jointById(i)
                pos = joint.globalPos()
                
                # Add joint position (using Z and Y coordinates)
                bvh_z_coords.append(pos[2])
                bvh_y_coords.append(pos[1])
                
                # Draw joint
                ax1.scatter(pos[2], pos[1], color="g", s=30)
                
                # Draw connection to parent (if any)
                if joint.parent:
                    parent_pos = joint.parent.globalPos()
                    ax1.plot([parent_pos[2], pos[2]],  # Z coordinates
                            [parent_pos[1], pos[1]],  # Y coordinates
                            'r-', linewidth=1.5)
            
            # Add joint labels to BVH visualization (only for main joints)
            main_joints = ['Hips', 'Spine', 'Head', 'RightHand', 'LeftHand', 'RightFoot', 'LeftFoot']
            for joint_name in main_joints:
                try:
                    joint = self.bvh.jointByName(joint_name)
                    if joint:
                        pos = joint.globalPos()
                        ax1.text(pos[2], pos[1], joint_name, fontsize=8)
                except:
                    pass  # Joint doesn't exist
            
            # Update plot limits based on BVH data
            if bvh_z_coords and bvh_y_coords:
                bvh_plot_limits['z_min'] = min(bvh_plot_limits['z_min'], min(bvh_z_coords))
                bvh_plot_limits['z_max'] = max(bvh_plot_limits['z_max'], max(bvh_z_coords))
                bvh_plot_limits['y_min'] = min(bvh_plot_limits['y_min'], min(bvh_y_coords))
                bvh_plot_limits['y_max'] = max(bvh_plot_limits['y_max'], max(bvh_y_coords))
            
            # Plot retargeted GO1 as 2D top view (X-Z plane)
            if frame < len(self.retargeted_frames):
                retargeted_pose = self.retargeted_frames[frame]
                
                # Get root position from the retargeted pose
                root_pos = retargeted_pose['root_pos']
                
                # Store all GO1 joint positions for plot scaling
                go1_x_coords = []
                go1_z_coords = []
                
                # Draw GO1 robot using full kinematic tree (in X-Z plane)
                self.draw_go1_robot_2d_xz(ax2, root_pos, retargeted_pose, go1_x_coords, go1_z_coords)
                
                # Update plot limits based on GO1 data - only if we have data
                if go1_x_coords and go1_z_coords:
                    go1_plot_limits['x_min'] = min(go1_plot_limits['x_min'], min(go1_x_coords))
                    go1_plot_limits['x_max'] = max(go1_plot_limits['x_max'], max(go1_x_coords))
                    go1_plot_limits['z_min'] = min(go1_plot_limits['z_min'], min(go1_z_coords))
                    go1_plot_limits['z_max'] = max(go1_plot_limits['z_max'], max(go1_z_coords))
            
            # Calculate a reasonable range for each plot
            padding = 0.5  # Add padding around the objects
            
            # Process BVH plot limits
            # Ensure the plot limits have some meaningful range
            bvh_z_range = bvh_plot_limits['z_max'] - bvh_plot_limits['z_min']
            bvh_y_range = bvh_plot_limits['y_max'] - bvh_plot_limits['y_min']
            
            if bvh_z_range < 0.1:
                mid_z = (bvh_plot_limits['z_max'] + bvh_plot_limits['z_min']) / 2
                bvh_plot_limits['z_min'] = mid_z - 0.5
                bvh_plot_limits['z_max'] = mid_z + 0.5
            
            if bvh_y_range < 0.1:
                mid_y = (bvh_plot_limits['y_max'] + bvh_plot_limits['y_min']) / 2
                bvh_plot_limits['y_min'] = mid_y - 0.5
                bvh_plot_limits['y_max'] = mid_y + 0.5
                
            # Process GO1 plot limits
            go1_x_range = go1_plot_limits['x_max'] - go1_plot_limits['x_min']
            go1_z_range = go1_plot_limits['z_max'] - go1_plot_limits['z_min']
            
            if go1_x_range < 0.1:
                mid_x = (go1_plot_limits['x_max'] + go1_plot_limits['x_min']) / 2
                go1_plot_limits['x_min'] = mid_x - 0.5
                go1_plot_limits['x_max'] = mid_x + 0.5
            
            if go1_z_range < 0.1:
                mid_z = (go1_plot_limits['z_max'] + go1_plot_limits['z_min']) / 2
                go1_plot_limits['z_min'] = mid_z - 0.5
                go1_plot_limits['z_max'] = mid_z + 0.5
            
            # Set the axis limits with padding - different for each plot
            ax1.set_xlim(bvh_plot_limits['z_min'] - padding, bvh_plot_limits['z_max'] + padding)
            ax1.set_ylim(bvh_plot_limits['y_min'] - padding, bvh_plot_limits['y_max'] + padding)
            
            ax2.set_xlim(go1_plot_limits['x_min'] - padding, go1_plot_limits['x_max'] + padding)
            ax2.set_ylim(go1_plot_limits['z_min'] - padding, go1_plot_limits['z_max'] + padding)
            
            # Show frame number with adjusted position and larger font
            frame_text_props = dict(
                verticalalignment='top',
                fontsize=12,
                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.8,
                    pad=5
                )
            )
            
            ax1.text(0.02, 0.98, f"Frame: {frame}", transform=ax1.transAxes, **frame_text_props)
            ax2.text(0.02, 0.98, f"Frame: {frame}", transform=ax2.transAxes, **frame_text_props)
            
            # Add axis grid with better visibility
            ax1.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
            ax2.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
            
            # Set aspect ratio to be equal
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            
            return ax1, ax2
        
        # Create animation with blit=False
        num_frames = min(200, self.bvh.numFrames())  # Limit number of frames for faster visualization
        anim = animation.FuncAnimation(fig, update, frames=num_frames, 
                                     interval=1000/30,  # 30 FPS
                                     blit=False)  # Changed to False
        
        # Show the animation
        plt.show()

    def build_kinematic_tree(self):
        """Build a kinematic tree of the GO1 robot from URDF data"""
        
        # Create a dictionary to store the tree structure
        self.kinematic_tree = {}
        
        # Dictionary mapping parent links to their child joints
        parent_to_joints = {}
        
        # First pass: organize joints by parent link
        for joint_name, joint_data in self.go1.joints.items():
            parent_link = joint_data['parent']
            if parent_link not in parent_to_joints:
                parent_to_joints[parent_link] = []
            
            parent_to_joints[parent_link].append({
                'joint_name': joint_name,
                'joint_data': joint_data,
                'child_link': joint_data['child']
            })
        
        # Second pass: build tree starting from trunk
        self.kinematic_tree = self._build_tree_recursive('trunk', parent_to_joints)
        
    def _build_tree_recursive(self, link_name, parent_to_joints):
        """Recursively build the kinematic tree starting from the given link"""
        node = {
            'link_name': link_name,
            'joints': []
        }
        
        # If this link has no child joints, return the node as a leaf
        if link_name not in parent_to_joints:
            return node
        
        # Add child joints and their child links
        for joint_info in parent_to_joints[link_name]:
            joint = {
                'joint_name': joint_info['joint_name'],
                'joint_data': joint_info['joint_data'],
                'child_link': self._build_tree_recursive(joint_info['child_link'], parent_to_joints)
            }
            node['joints'].append(joint)
        
        return node

    def draw_go1_robot_2d_xz(self, ax, root_pos, pose, x_coords, z_coords):
        """Draw the GO1 robot in 2D top-down view (X-Z plane) strictly following URDF data"""
        
        # Define scaling factor
        scale = 1000.0
        
        # Print debug info
        print("Root position:", root_pos)
        
        # Find trunk dimensions (using defaults for simplicity)
        trunk_link = 'trunk'
        trunk_dims = [0.45, 0.15, 0.10]  # Default values (length, width, height)
            
        # Scale dimensions
        trunk_length = trunk_dims[0] * scale # Along X
        trunk_width = trunk_dims[1] * scale  # Along Y
        
        # Invert Z coordinate for root position
        root_pos_z_inverted = [-root_pos[2]]
        
        # Body rectangle bottom-left corner in X-Z coords
        rect_x = root_pos[0] - trunk_length/2  # Center trunk at root X
        rect_z = root_pos_z_inverted[0] - trunk_width/2   # Use trunk width for the Z dimension
        
        # Create and add rectangle to represent the body
        body_rect = plt.Rectangle((rect_x, rect_z), trunk_length, trunk_width, 
                                 fill=True, color='lightblue', alpha=0.5, 
                                 edgecolor='blue', linewidth=2)
        ax.add_patch(body_rect)
        
        # Draw center of body (X, Z)
        ax.scatter(root_pos[0], root_pos_z_inverted[0], color="blue", s=100, label="Trunk")
        
        # Add trunk position to coordinate lists
        x_coords.append(root_pos[0])
        z_coords.append(root_pos_z_inverted[0])
        
        # ------------ DRAW LEGS (X-Z PLANE) ------------
        # Define hip positions directly from URDF data
        hip_offsets = {
            'FR': [0.1881 * scale, -0.04675 * scale, 0],  # Front Right
            'FL': [0.1881 * scale, 0.04675 * scale, 0],   # Front Left
            'RR': [-0.1881 * scale, -0.04675 * scale, 0], # Rear Right
            'RL': [-0.1881 * scale, 0.04675 * scale, 0]   # Rear Left
        }
        
        # Thigh and calf lengths from URDF
        thigh_length = 0.213 * scale
        calf_length = 0.213 * scale
        
        leg_prefixes = ['FR', 'FL', 'RR', 'RL']
        colors = ['red', 'green', 'blue', 'purple']
        
        # Draw each leg (X-Z plane view - from top down)
        for i, prefix in enumerate(leg_prefixes):
            color = colors[i]
            
            # Get joint names
            hip_joint = f"{prefix}_hip_joint"
            thigh_joint = f"{prefix}_thigh_joint"
            calf_joint = f"{prefix}_calf_joint"
            
            # Get angles from pose (or default to 0)
            hip_angle = pose.get(hip_joint, 0)   # Roll - affects Y not visible in XZ plane
            thigh_angle = pose.get(thigh_joint, 0) # Pitch - visible as X-Z angle change
            knee_angle = pose.get(calf_joint, 0)  # Pitch - visible as X-Z angle change
            
            # 1. Calculate hip position (trunk + offset) - fixed from URDF
            hip_offset = hip_offsets[prefix]
            hip_pos = [
                root_pos[0] + hip_offset[0],  # X offset
                root_pos[1] + hip_offset[1],  # Y offset - not visible in XZ plane
                -(root_pos[2] + hip_offset[2])   # Z offset (inverted)
            ]
            
            # 2. Calculate knee position with thigh_angle in X-Z plane
            # For realistic visualization, we need to project the joint angles to the X-Z plane
            # Apply thigh rotation around Y axis (pitching the thigh forward/backward)
            knee_pos = [
                hip_pos[0] + thigh_length * np.sin(thigh_angle),  # X
                hip_pos[1],                                       # Y (stays the same)
                hip_pos[2] - thigh_length * np.cos(thigh_angle)   # Z (inverted cosine)
            ]
            
            # 3. Calculate foot position with knee_angle relative to thigh
            # Calculate the absolute angle of the calf in the X-Z plane
            # knee_angle is negative when bending, so we're actually adding here
            calf_angle = thigh_angle + knee_angle
            
            foot_pos = [
                knee_pos[0] + calf_length * np.sin(calf_angle),   # X
                knee_pos[1],                                       # Y (stays the same)
                knee_pos[2] - calf_length * np.cos(calf_angle)    # Z (inverted cosine)
            ]
            
            # Draw hip joint
            ax.scatter(hip_pos[0], hip_pos[2], color=color, s=60, marker='o')
            ax.text(hip_pos[0], hip_pos[2], f"{prefix}_hip", fontsize=8)
            
            # Draw thigh segment
            ax.plot([hip_pos[0], knee_pos[0]], [hip_pos[2], knee_pos[2]], color=color, linewidth=3)
            
            # Draw knee joint
            ax.scatter(knee_pos[0], knee_pos[2], color=color, s=50, marker='o')
            ax.text(knee_pos[0], knee_pos[2], f"{prefix}_knee", fontsize=8)
            
            # Draw calf segment
            ax.plot([knee_pos[0], foot_pos[0]], [knee_pos[2], foot_pos[2]], color=color, linewidth=3)
            
            # Draw foot
            ax.scatter(foot_pos[0], foot_pos[2], color=color, s=50, marker='s')
            ax.text(foot_pos[0], foot_pos[2], f"{prefix}_foot", fontsize=8)
            
            # Add to coordinate lists for plot scaling
            x_coords.extend([hip_pos[0], knee_pos[0], foot_pos[0]])
            z_coords.extend([hip_pos[2], knee_pos[2], foot_pos[2]])
        
        # Add debug check message
        print(f"x_coords range: {min(x_coords)} to {max(x_coords)}")
        print(f"z_coords range: {min(z_coords)} to {max(z_coords)}")
        
        # Add color-coded legend
        for i, prefix in enumerate(leg_prefixes):
            ax.plot([], [], color=colors[i], linestyle='-', linewidth=2, label=f'{prefix} Leg')
        ax.legend(loc='upper right', fontsize=8)


if __name__ == "__main__":
    urdf_path = "go1.urdf"
    # --- UPDATE BVH PATH ---
    bvh_filename = "D1_061z_KAN01_002.bvh" # Use the new filename
    bvh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", bvh_filename)
    
    # Check if BVH file exists
    if not os.path.exists(bvh_path):
        print(f"Error: BVH file not found at {bvh_path}")
        print("Please ensure the BVH file is placed in the 'dataset' directory.")
        sys.exit(1)

    print("Starting GO1 motion retargeting...")

    # Create GO1 model from URDF
    go1_model = GO1Model(urdf_path)

    # Calculate link lengths (Might still be useful for other purposes or verification)
    go1_model.calculate_link_lengths()
    # Print calculated lengths for debugging (optional)
    # print("Calculated link lengths:", go1_model.link_lengths)

    # Create retargeter
    retargeter = GO1Retargeter(go1_model, bvh_path)

    # Match end effector positions
    retargeter.match_end_effector_positions()

    # Compute inverse kinematics
    retargeter.compute_inverse_kinematics()

    # Visualize comparison
    retargeter.visualize_comparison()
