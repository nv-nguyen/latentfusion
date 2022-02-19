from latentfusion.datasets.realsense import RealsenseDataset
from latentfusion.meshutils import Object3D
from latentfusion.pointcloud import load_ply
from pathlib import Path
import os
from latentfusion import three
import numpy as np
import matplotlib.pyplot as plt

dataset_dir = "/home/nguyen/Documents/datasets/pizza_datasets/moped2/"
obj_name = "black_drill"
pointcloud = load_ply(os.path.join(dataset_dir, obj_name, "reference", 'pointcloud_eval.ply'))
diameter = three.points_diameter(pointcloud)
print("diameter", diameter)
object_scale = 1.0 / diameter

input_scene_dir = os.path.join(dataset_dir, obj_name, "reference")
input_paths = [x for x in Path(input_scene_dir).iterdir() if x.is_dir()]
input_dataset = RealsenseDataset(input_paths,
                                 image_scale=1.0,
                                 object_scale=object_scale,
                                 center_object=True,
                                 odometry_type='open3d',
                                 ref_points=pointcloud)
save_path = "./draft"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for idx_frame, data in enumerate(input_dataset):
    image = data["color"].detach().numpy()
    image = np.moveaxis(image, 0, -1)

    pose = data["extrinsic"].detach().numpy()
    K = data["intrinsic"].detach().numpy()[:3, :3]
    pointcloud_scale = pointcloud
    transformed_points = np.matmul(pose[:3, :3], pointcloud_scale.T) + pose[:3, 3].reshape(3, 1)
    print(transformed_points.shape)
    xyn2 = np.matmul(K, transformed_points)
    x2 = xyn2[0] / xyn2[2]
    y2 = xyn2[1] / xyn2[2]
    plt.imshow(image)
    plt.scatter(x2, y2, s=0.5, color="red")
    plt.savefig(os.path.join(save_path, "{:02d}.png".format(idx_frame)), bbox_inches='tight',
                pad_inches=0.1)
    plt.close("all")