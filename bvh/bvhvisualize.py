# bvhvisualizer.py, Aline Normoyle, 2024

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from bvh import *

def BVHVisualizeFrame(bvh, frame):
    bvh.readFrame(frame)

    # Visualize global locations of joints
    num = bvh.numJoints()
    x = np.zeros(num)
    y = np.zeros(num)
    z = np.zeros(num)
    for i in range(num):
        p = bvh.jointById(i).globalPos()
        x[i] = p[0]
        y[i] = p[1]
        z[i] = p[2]

    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(vertical_axis='y', share=True)
    
    # Draw joints
    ax.scatter(x, y, z, color="g", edgecolor="k")
    
    # Draw links (bones) between joints
    for i in range(num):
        joint = bvh.jointById(i)
        if joint.parent is not None:
            parent_pos = joint.parent.globalPos()
            joint_pos = joint.globalPos()
            ax.plot([parent_pos[0], joint_pos[0]], 
                    [parent_pos[1], joint_pos[1]], 
                    [parent_pos[2], joint_pos[2]], 'r-', linewidth=1.5)
    
    ax.set_title("Frame %d"%frame)
    plt.show()

class BVHAnimator:
    def __init__(self, bvh):
        self.bvh = bvh
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
            frames = range(bvh.numFrames()), init_func=self.setup_plot, blit=False)
        plt.show()

    def read_frame(self, f):
        self.bvh.readFrame(f)
        num = self.bvh.numJoints()
        x = np.zeros(num)
        y = np.zeros(num)
        z = np.zeros(num)
        
        # Store joint positions and parent-child relationships
        self.links = []
        
        for i in range(num):
            joint = self.bvh.jointById(i)
            p = joint.globalPos()
            x[i] = p[0]
            y[i] = p[1]
            z[i] = p[2]
            
            # Store parent-child link data
            if joint.parent is not None:
                parent_pos = joint.parent.globalPos()
                self.links.append((
                    [parent_pos[0], p[0]],
                    [parent_pos[1], p[1]],
                    [parent_pos[2], p[2]]
                ))
                
        return x, y, z

    def setup_plot(self):
        x, y, z = self.read_frame(0)
        
        # Draw joints
        self.scat = self.ax.scatter(x, y, z, color='g', edgecolor="k")
        
        # Draw links (bones)
        self.lines = []
        for link in self.links:
            line, = self.ax.plot(link[0], link[1], link[2], 'r-', linewidth=1.5)
            self.lines.append(line)
        
        self.ax.view_init(vertical_axis='y', share=True)
        
        # Coordinate axes
        self.axes = []
        self.axes.append(self.ax.plot([0.0, 100.0], [0.0, 0.0], [0.0, 0.0], color='r')[0])
        self.axes.append(self.ax.plot([0.0, 0.0], [0.0, 100.0], [0.0, 0.0], color='g')[0])
        self.axes.append(self.ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, 100.0], color='b')[0])
        
        self.title = self.ax.set_title('Frame 0')

        # For FuncAnimation's sake
        return (self.scat,)

    def update(self, i):
        idx = min(i, self.bvh.numFrames()-1)
        x, y, z = self.read_frame(idx)
        
        # Update joints
        self.title.set_text("Frame %d"%idx)
        self.scat._offsets3d = (x, y, z)
        
        # Update links
        for i, link in enumerate(self.links):
            if i < len(self.lines):
                self.lines[i].set_data([link[0], link[1]])
                self.lines[i].set_3d_properties(link[2])
            else:
                # If we need more lines (shouldn't happen, but just in case)
                line, = self.ax.plot(link[0], link[1], link[2], 'r-', linewidth=1.5)
                self.lines.append(line)
        
        # For blit=True, need to return all artists that were updated
        return [self.scat] + self.lines

