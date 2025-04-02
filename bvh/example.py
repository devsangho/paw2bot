# example.py, Aline Normoyle, 2024
from bvh import *
from bvhvisualize import *
import numpy as np
import matplotlib.pyplot as plt
import glm

animation = BVH()
animation.load("dataset/D1_009_KAN01_001.bvh")
BVHAnimator(animation)

# To visualize a single frame (range [0, animator.numFrames()-1]
# BVHVisualizeFrame(bvh, frame):
