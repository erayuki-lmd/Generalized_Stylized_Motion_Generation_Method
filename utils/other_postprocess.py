import os
import sys
import numpy as np
import torch
import argparse
from tqdm import tqdm

import utils.BVH as BVH
from utils.InverseKinematics import JacobianInverseKinematics
from utils.animation_data import AnimationData


def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r


def nrot2anim(nrot):
    anim = AnimationData.from_network_output(nrot)
    bvh, names, ftime = anim.get_BVH()
    anim = AnimationData.from_rotations_and_root_positions(np.array(bvh.rotations), bvh.positions[:, 0, :])
    glb = anim.get_global_positions(trim=False)

    return (bvh, names, ftime), glb

#####################CHANGE#####################################
def save_bvh_from_network_output(nrot, output_path):
    nrot = nrot.to('cpu').detach().numpy().copy()
    save_array = np.zeros((len(nrot),int(len(nrot[0])/3),3))
    for i in range(len(nrot)):
        for j in range(int(len(nrot[0])/3)):
            save_array[i][j][0] = nrot[i][3*j]
            save_array[i][j][1] = nrot[i][3*j+1]
            save_array[i][j][2] = nrot[i][3*j+2]
    np.save(output_path,save_array)
# def save_bvh_from_network_output(nrot, output_path):
#     anim = AnimationData.from_network_output(nrot)
#     bvh, names, ftime = anim.get_BVH()
#     if not os.path.exists(os.path.dirname(output_path)):
#         os.makedirs(os.path.dirname(output_path))
#     BVH.save(output_path, bvh, names, ftime)
#####################CHANGE#####################################


def remove_fs(anim, foot, output_path, fid_l=(4, 5), fid_r=(9, 10), interp_length=5, force_on_floor=True):
    (anim, names, ftime), glb = nrot2anim(anim)
    T = len(glb)

    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  # [T, 2] -> [T]
    # print(np.min(foot_heights))
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    # print(floor_height)
    glb[:, :, 1] -= floor_height
    anim.positions[:, 0, 1] -= floor_height

    for i, fidx in enumerate(fid):
        fixed = foot[i]  # [T]

        """
        for t in range(T):
            glb[t, fidx][1] = max(glb[t, fidx][1], 0.25)
        """

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            if force_on_floor:
                avg[1] = 0.0

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()

            # print(fixed[s - 1:t + 2])

            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    targetmap = {}
    for j in range(glb.shape[1]):
        targetmap[j] = glb[:, j]

    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=4.0,
                                   silent=False)
    ik()

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    BVH.save(output_path, anim, names, ftime)
