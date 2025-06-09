import numpy as np
from scipy import sparse
import pyssdr
import trimesh
import open3d as o3d
import numpy as np
import matplotlib.colors as mcolors
import os
import pyvista as pv
from pyvista import examples
import time
from tqdm import tqdm

def visualize_4d_skinning(points_sequence, weights, bone_positions_sequence, fps=30, output_path=None, loop=True):
    """
    创建4D点云动画可视化 - 优化版本
    
    参数:
    points_sequence: 列表，每个元素是一个(N, 3)的点云坐标数组
    weights: (N, num_bones)的权重矩阵
    bone_positions_sequence: 列表，每个元素是一个(num_bones, 3)的骨骼位置数组
    fps: 每秒帧数
    output_path: 如果提供，将动画保存为视频文件
    loop: 是否循环播放动画（仅在交互模式下有效）
    """
    import matplotlib.colors as mcolors
    from scipy import sparse
    import time
    
    # 确保权重是密集矩阵
    if sparse.issparse(weights):
        weights = weights.toarray()
    
    # 检查权重形状并转置如果需要
    if weights.shape[0] == len(bone_positions_sequence[0]) and weights.shape[1] == len(points_sequence[0]):
        weights = weights.T
    
    # 生成骨骼颜色
    num_bones = weights.shape[1]
    colors = generate_distinct_colors(num_bones)
    
    # 计算每个点的混合颜色
    blended_colors = np.zeros((len(points_sequence[0]), 3))
    for i in range(num_bones):
        weight = weights[:, i].reshape(-1, 1)
        color = np.array(colors[i]).reshape(1, 3)
        blended_colors += weight * color
    
    # 确保RGB值在有效范围内
    blended_colors = np.clip(blended_colors, 0, 1)
    
    # 确定是否使用离屏渲染
    off_screen = output_path is not None
    
    # 创建绘图器
    plotter = pv.Plotter(off_screen=off_screen)
    
    # 如果要保存视频
    if output_path:
        plotter.open_movie(output_path)
    
    # 设置相机位置 - XZ平面视角但稍微后移
    plotter.camera_position = 'yz'  # 先设置为XZ平面(前视图)
    # 获取当前相机位置
    cam_pos = plotter.camera.position
    focal_point = plotter.camera.focal_point
    # 计算方向向量
    direction = np.array(cam_pos) - np.array(focal_point)
    # 将相机后移（增加距离）
    distance_factor = 4  # 调整此值以控制后移距离
    new_cam_pos = np.array(focal_point) + direction * distance_factor + np.array([0, 0, 1])
    # 设置新的相机位置
    plotter.camera.position = new_cam_pos
    
    # 性能优化：预先创建点云和骨骼对象
    cloud = pv.PolyData(points_sequence[0])
    cloud.point_data["colors"] = blended_colors * 255
    cloud_actor = plotter.add_mesh(cloud, scalars="colors", rgb=True, point_size=5)
    
    # 预先创建骨骼球体
    is_sequence = isinstance(bone_positions_sequence, list) or (
        isinstance(bone_positions_sequence, np.ndarray) and len(bone_positions_sequence.shape) > 2)
    
    # 创建骨骼球体和对应的actors
    bone_spheres = []
    bone_actors = []
    
    # 获取第一帧的骨骼位置
    first_frame_bones = bone_positions_sequence[0] if is_sequence else bone_positions_sequence
    
    for j, pos in enumerate(first_frame_bones):
        sphere = pv.Sphere(radius=0.03, center=pos)
        sphere.point_data["color"] = np.tile(np.array(colors[j]) * 255, (sphere.n_points, 1))
        actor = plotter.add_mesh(sphere, scalars="color", rgb=True)
        bone_spheres.append(sphere)
        bone_actors.append(actor)
    
    # 对每一帧进行渲染，如果交互式渲染，则循环播放
    if output_path:
        # 如果是保存视频，只播放一次
        for i in tqdm(range(len(points_sequence))):
            # 更新点云位置而不是重新创建
            cloud.points = points_sequence[i]
            
            # 更新骨骼位置
            if is_sequence:
                bone_positions = bone_positions_sequence[i]
                for j, (pos, sphere, actor) in enumerate(zip(bone_positions, bone_spheres, bone_actors)):
                    # 计算位移向量
                    translation = pos - sphere.center
                    # 更新球体位置
                    sphere.translate(translation, inplace=True)
                    # 更新渲染器中的位置
                    plotter.update_coordinates(sphere, sphere.points, render=False)
            
            # 一次性渲染所有更新
            plotter.render()
            plotter.write_frame()
    else:
        # 交互式显示模式
        # 添加键盘回调函数来控制播放
        play_state = {'playing': True, 'frame_index': 0}
        
        # 创建无参数的回调函数
        def space_press_callback():
            # 空格键暂停/继续
            play_state['playing'] = not play_state['playing']
        
        def r_press_callback():
            # r键重置到第一帧
            play_state['frame_index'] = 0
        
        def q_press_callback():
            # q键退出
            plotter.close()
        
        plotter.add_key_event('space', space_press_callback)
        plotter.add_key_event('r', r_press_callback)
        plotter.add_key_event('q', q_press_callback)
        
        # 添加文本说明
        plotter.add_text("Space: Pause/Continue  R: Reset  Q: Exit  Mouse: Rotate View", position='lower_right', font_size=12)
        
        # 明确启用交互控制
        plotter.enable_trackball_style()  # 启用轨迹球样式的相机控制
        plotter.enable_eye_dome_lighting()  # 增强点云视觉效果
        
        # 使用自定义计时器事件进行循环播放
        def timer_callback(step):
            if play_state['playing']:
                # 获取当前帧索引
                i = play_state['frame_index']
                
                # 更新点云位置
                cloud.points = points_sequence[i]
                
                # 更新骨骼位置
                if is_sequence:
                    bone_positions = bone_positions_sequence[i]
                    for j, (pos, sphere, actor) in enumerate(zip(bone_positions, bone_spheres, bone_actors)):
                        translation = pos - sphere.center
                        sphere.translate(translation, inplace=True)
                        plotter.update_coordinates(sphere, sphere.points, render=False)
                
                # 更新帧索引
                play_state['frame_index'] = (i + 1) % len(points_sequence) if loop else min(i + 1, len(points_sequence) - 1)
                
                # 如果到达末尾且不循环，则停止播放
                if not loop and play_state['frame_index'] == len(points_sequence) - 1:
                    play_state['playing'] = False
                
                # 渲染更新但不阻塞交互
                plotter.update()  # 使用update而不是render以保持交互性
        
        # 设置计时器，按照fps控制帧率
        # 对于循环播放，设置足够大的max_steps
        max_steps = 100000 if loop else len(points_sequence)
        duration = int(1000/fps)  # 毫秒为单位
        plotter.add_timer_event(max_steps=max_steps, duration=duration, callback=timer_callback)
        
        # 启动交互式窗口，确保启用交互控制
        plotter.show(interactive=True, auto_close=False)
    
    # 关闭绘图器
    plotter.close()


def compute_ssdr(poses, rest_pose, skeleton, faces, num_bones=None, tolerance=1e-3, patience=3, 
                 max_iters=100, nnz=4, weights_smooth=0):
    """
    Compute Smooth Skinning Decomposition with Rigid Bones (SSDR)
    
    Parameters:
    -----------
    poses : numpy.ndarray, shape (T, N, 3)
        Animation sequence with T frames, N vertices, and 3D coordinates
    rest_pose : numpy.ndarray, shape (N, 3)
        Rest pose of the mesh
    skeleton : numpy.ndarray, shape (J, 3)
        Initial skeleton joint positions
    faces : numpy.ndarray, shape (F, 3)
        Face indices of the mesh
    num_bones : int, optional
        Number of bones to use. If None, uses number of joints in skeleton
    tolerance : float, default=1e-3
        Convergence tolerance
    patience : int, default=3
        Number of iterations with minimal improvement before stopping
    max_iters : int, default=100
        Maximum number of iterations
    nnz : int, default=4
        Maximum number of non-zero weights per vertex
    weights_smooth : float, default=0 我们不用，因为我们只有点云，用不了！
        Weight smoothness parameter
        
    Returns:
    --------
    weights : scipy.sparse.csc_matrix, shape (num_bones, N)
        Skinning weights
    transformations : numpy.ndarray, shape (T, num_bones, 4, 3)
        Bone transformations for each frame
    rmse : float
        Root mean square error of the reconstruction
    """
    # Initialize SSDR
    ssdr = pyssdr.MyDemBones()
    
    # Set parameters
    ssdr.tolerance = tolerance
    ssdr.patience = patience
    ssdr.nIters = max_iters
    ssdr.nnz = nnz
    ssdr.weightsSmooth = weights_smooth
    ssdr.bindUpdate = 2  # Regroup joints under one root
    
    # Reshape trajectory for SSDR input format
    T, N, D = poses.shape
    trajectory = poses.transpose((0, 2, 1))  # (T, 3, N)
    trajectory = trajectory.reshape((D*T, N))  # (3*T, N)
    
    # Convert faces to the format expected by SSDR
    face_list = [] if faces is None else [list(face) for face in faces]
    
    # Load data into SSDR
    ssdr.load_data(trajectory, face_list)
    
    # Set number of bones
    if num_bones is None:
        num_bones = skeleton.shape[0]
    
    # Initialize labels based on nearest joints if skeleton is provided
    if skeleton is not None:
        # Find closest joint for each vertex
        from scipy.spatial import cKDTree
        tree = cKDTree(skeleton)
        _, indices = tree.query(rest_pose)
        
        # Set labels based on closest joints
        ssdr.label = indices
        ssdr.nB = num_bones
        ssdr.labelToWeights()
        ssdr.computeTransFromLabel()
    
    # Run SSDR
    weights, transformations, rmse = ssdr.run_ssdr(num_bones,"here.fbx")
    
    # Get reconstruction for visualization/validation
    reconstruction = ssdr.compute_reconstruction(list(range(N)))
    deformed_vertices = np.zeros((T, N, 3))
    deformed_vertices[:, :, 0] = reconstruction[list(range(0, 3*T, 3))]
    deformed_vertices[:, :, 1] = reconstruction[list(range(1, 3*T, 3))]
    deformed_vertices[:, :, 2] = reconstruction[list(range(2, 3*T, 3))]
    
    return weights, transformations, rmse, deformed_vertices

def generate_distinct_colors(n):
    """Generate n visually distinct colors using HSV color space with maximized differences"""
    if n <= 10:
        # For small number of bones, use a predefined color palette
        # These colors are chosen to be visually distinct
        base_colors = [
            (0.9, 0.1, 0.1),  # Red
            (0.1, 0.6, 0.9),  # Blue
            (0.1, 0.9, 0.1),  # Green
            (0.9, 0.6, 0.1),  # Orange
            (0.6, 0.1, 0.9),  # Purple
            (0.9, 0.9, 0.1),  # Yellow
            (0.1, 0.9, 0.6),  # Teal
            (0.9, 0.1, 0.6),  # Pink
            (0.5, 0.5, 0.5),  # Gray
            (0.7, 0.3, 0.0),  # Brown
        ]
        return base_colors[:n]
    
    # For larger sets, use the golden ratio to space hues evenly
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = 0.1  # Starting hue
    
    for i in range(n):
        # Generate color with varying saturation and value to increase contrast
        s = 0.7 + 0.3 * ((i % 3) / 2.0)  # Vary saturation
        v = 0.8 + 0.2 * ((i % 2) / 1.0)  # Vary value/brightness
        
        rgb = mcolors.hsv_to_rgb((h, s, v))
        colors.append(rgb)
        
        # Use golden ratio to get next hue - this creates maximally distinct colors
        h = (h + golden_ratio_conjugate) % 1.0
    
    return colors

def visualize_skinning_weights(points, weights, bone_positions, alpha=0.3, bone_size=0.05, save_path=None):
    """
    Visualize skinning weights with bone positions using Open3D
    
    Parameters:
    points : numpy.ndarray (N, 3)
        Point cloud positions
    weights : numpy.ndarray or scipy.sparse.csr_matrix (num_bones, N) or (N, num_bones)
        Skinning weights matrix
    bone_positions : numpy.ndarray (num_bones, 3)
        Bone positions in rest pose
    alpha : float (0-1)
        Transparency of point cloud
    bone_size : float
        Size of bone spheres
    save_path : str, optional
        If provided, saves the visualization to this path instead of showing interactive window
    """
    from scipy import sparse
    import numpy as np
    import matplotlib.colors as mcolors
    
    # Convert sparse weights to dense if needed
    if sparse.issparse(weights):
        weights = weights.toarray()
    
    # Check weights shape and transpose if needed
    if weights.shape[0] == len(bone_positions) and weights.shape[1] == len(points):
        # Shape is (num_bones, N), transpose to (N, num_bones)
        weights = weights.T
    
    # Create color map for bones
    num_bones = weights.shape[1]
    colors = generate_distinct_colors(num_bones)
    
    # Create point cloud with blended colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Calculate blended colors for each point
    blended_colors = np.zeros((len(points), 3))  # RGB only
    for i in range(num_bones):
        weight = weights[:, i].reshape(-1, 1)  # Ensure column vector (N, 1)
        color = np.array(colors[i]).reshape(1, 3)  # Ensure row vector (1, 3)
        blended_colors += weight * color  # Broadcasting (N, 1) * (1, 3) -> (N, 3)
    
    # Ensure RGB values are in valid range [0, 1]
    blended_colors = np.clip(blended_colors, 0, 1)
    
    pcd.colors = o3d.utility.Vector3dVector(blended_colors)

    # Create bone spheres
    bone_geoms = []
    for i, pos in enumerate(bone_positions):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=bone_size)
        sphere.translate(pos)
        sphere.paint_uniform_color(colors[i])  # Use RGB for mesh
        sphere.compute_vertex_normals()
        bone_geoms.append(sphere)

    # If save_path is provided, use offscreen rendering
    if save_path:
        # Create a simple scene with all geometries
        geometries = [pcd] + bone_geoms
        
        # Use matplotlib to save a simple visualization
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        point_array = np.asarray(pcd.points)
        color_array = np.asarray(pcd.colors)
        ax.scatter(point_array[:, 0], point_array[:, 1], point_array[:, 2], 
                  c=color_array, s=1, alpha=alpha)
        
        # Plot bones
        for i, pos in enumerate(bone_positions):
            ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                      c=[colors[i]], s=100, alpha=1.0)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {save_path}")
        return
    
    # Otherwise try interactive visualization
    try:
        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add geometries
        vis.add_geometry(pcd)
        for geom in bone_geoms:
            vis.add_geometry(geom)

        # Set render options
        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.array([0.1, 0.1, 0.1])
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"Interactive visualization failed: {e}")
        print("Try using save_path parameter instead to save visualization as an image.")


def joint2part(poses, weights, skeleton):
    # 将cluster分配给各个Joint
    pass

# Example usage:
if __name__ == "__main__":
    # Load your data here
    motion_analysis_results = np.load("/home/wjx/research/code/GaussianAnimator/LBS-GS/tempexp/trex/motion_analysis_results.npy", allow_pickle=True).item()
    moving_points = motion_analysis_results['moving_points']
    bones = motion_analysis_results['bones']
    joints_index = motion_analysis_results['joints_index']
    root_indx = motion_analysis_results['root_index']
    moving_mask = motion_analysis_results['moving_mask']
    motion_magnitudes = motion_analysis_results['motion_magnitudes']
    poses = motion_analysis_results['poses']
    rest_pose = motion_analysis_results['rest_pose']


    # poses = np.load("poses.npy")  # Shape (T, N, 3)
    # rest_pose = poses[0]          # Shape (N, 3)
    # skeleton = np.load("skeleton.npy")  # Shape (J, 3)
    # faces = np.load("faces.npy")  # Shape (F, 3)
    
    if os.path.exists("result_can_be_visualize.npy"):
        result_can_be_visualize = np.load("result_can_be_visualize.npy", allow_pickle=True).item()
        weights = result_can_be_visualize["weights"]
        skeleton = result_can_be_visualize["skeleton"]
        print(f"weights shape: {weights.shape}")
        print(f"skeleton shape: {skeleton.shape}")
    else:
        # Run SSDR
        weights, transformations, rmse, reconstruction = compute_ssdr(
            poses, rest_pose, moving_points[joints_index], None, max_iters=10)
        skeleton = moving_points[joints_index]
        result_can_be_visualize = {
            "rest_pose": rest_pose,
            "weights": weights,
            "skeleton": skeleton,
        }
        np.save("result_can_be_visualize.npy", result_can_be_visualize)
    # 进行每个点的weights可视化，同时将joints也可视化，观察二者重合的部分
    
    # visualize_skinning_weights(poses[0], weights, skeleton, alpha=0.3, bone_size=0.05)
    visualize_4d_skinning(poses, weights.T, skeleton, fps=30, output_path="output.mp4")

    # 进行骨骼的绑定，然后使用我们自己的SSDR再跑3个iter进行微调
    # 骨骼绑定部分，其实应该去掉一些点，即和skinning分区非常不匹配的点，但是这样似乎又太相信SSDR的Joint提取了，如何折中?
    optimized_skeleton = joint2part(poses, weights, skeleton)
    # 
    
    # print(f"RMSE: {rmse}")
    # print(f"Weights shape: {weights.shape}")
    # print(f"Transformations shape: {transformations.shape}")