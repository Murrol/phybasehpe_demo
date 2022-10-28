import os
from poselib.skeleton.skeleton3d import SkeletonMotion_dof, SkeletonMotion, SkeletonState
from poselib.visualization.common import plot_skeleton_motion_interactive, plot_skeleton_motion, plot_skeleton_state
from poselib.visualization.skeleton_plotter_tasks import Draw3DSkeletonState
from poselib.visualization.plt_plotter import Matplotlib3DPlotter

data_root1 = "/home/datassd/yuxuan/amass_with_babel"
data_root2 = "/home/datassd/yuxuan/amass_with_babel_precomputed"
motion_file = "8a4d4974-11dc-4948-8a9c-9df873090c09.npy"

curr_file1 = os.path.join(data_root1, motion_file)
curr_motion1 = SkeletonMotion.from_file(curr_file1)

curr_file2 = os.path.join(data_root2, motion_file)
curr_motion2 = SkeletonMotion_dof.from_file(curr_file2)

########### To see motion in matplotlib window, automatic play ###########
# plot_skeleton_motion(curr_motion2)
# plot_skeleton_motion_interactive(curr_motion2)


########### To see a specific frame in a motion sequence ###########
t = 0
dt = 1 / curr_motion2.fps
frame_index = int(t / dt)
curr_state = curr_motion2.clone()
curr_state.tensor = curr_state.tensor[frame_index, :]
skeleton_state_task = Draw3DSkeletonState(
                        "draw_skeleton_state",
                        curr_state,
                    )

plotter = Matplotlib3DPlotter(skeleton_state_task)

# to play the sequence using
# for frame_id in range(len(curr_motion2)):
#     curr_state = curr_motion2.clone()
#     curr_state.tensor = curr_state.tensor[frame_id, :]
#     skeleton_state_task.update(curr_state)
#     plotter.update()

plotter.show()