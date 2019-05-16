import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

''' H36M joint names '''
joints = {
    'Hip': 0,
    'RHip': 1,
    'RKnee': 2,
    'RFoot': 3,
    'LHip': 4,
    'LKnee': 5,
    'LFoot': 6,
    'Spine': 7,
    'Thorax': 8,
    'Neck/Nose': 9,
    'Head': 10,
    'LShoulder': 11,
    'LElbow': 12,
    'LWrist': 13,
    'RShoulder': 14,
    'RElbow': 15,
    'RWrist': 16
}

def show3Dpose(pose, paco): # blue, orange
  """
  Visualize a 3d skeleton
  Args
    pose: 96x1 vector. The pose to plot.
    ax:

  Returns
    Nothing. Draws on ax.
  """
  print('Visualising 3D skeleton')


  fig = plt.figure(figsize=(19.2, 10.8))
  gs1 = gridspec.GridSpec(2, 1)  # 5 rows, 9 columns
  gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
  subplot_idx, exidx = 1, 1

  ax = plt.subplot(gs1[subplot_idx], projection='3d') # matplotlib 3d axis to draw on

  add_labels = True # whether to add coordinate labels
  lcolor = "#3498db" # color for left part of the body
  rcolor = "#e74c3c" # color for right part of the body

  # assert pose.size == 17*3, "channels should have 96 entries, it has %d instead" % pose.size
  vals = np.reshape( pose, (17, -1) )

  I   = np.array([1,2,4,5,7,9,10,11,12,14,15])-1 # start points
  J   = np.array([2,3,5,6,0,7, 9,12,13,15,16])-1 # end points
  LR  = np.array([0,0,1,1,0,0,0,1,1,0,0], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)
    # ax.plot(x, y, z)
  plt.show()

  # RADIUS = 750 # space around the subject
  # xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  # ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  # ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
  # ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

  # Get rid of the ticks and tick labels
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])

  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])
  ax.set_zticklabels([])
  ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  # ax.w_xaxis.line.set_color(white)
  # ax.w_yaxis.line.set_color(white)
  # ax.w_zaxis.line.set_color(white)
  plt.show()




if __name__ == '__main__':
  # keypoints_3d = np.load('../../data/action_db/hmr_3d_keypoints.npz', encoding='latin1')['positions_3d'].item()
  print('Loading keypoints')
  keypoints_3d = np.load('../../data/paco/paco_keypoints.npz', encoding='latin1')['positions_3d'].item()
  print('Done')
  paco = True
  if paco:
      for subject in keypoints_3d:
        for action in keypoints_3d[subject]:
          for emotion in keypoints_3d[subject][action]:
            for i, data in enumerate(keypoints_3d[subject][action][emotion]):
              print('Visualising vid ' + str(
                i) + ' for subject: ' + subject + ', emotion: ' + emotion)
              kpts = data['keypoints']
              for frame in kpts:
                show3Dpose(frame, paco)
  else:
      for subject in keypoints_3d:
          for action in keypoints_3d[subject]:
              for emotion in keypoints_3d[subject][action]:
                  for intensity in keypoints_3d[subject][action][emotion]:
                      for i, data in enumerate(keypoints_3d[subject][action][emotion][intensity]):
                          print('Visualising vid ' + str(
                              i) + ' for subject: ' + subject + ', emotion: ' + emotion + ', intensity: ' + intensity)
                          for frame in data:
                              show3Dpose(frame, paco)
