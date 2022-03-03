import os
import random
import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from data_loader.load_vkitti import load_vkitti_data
from data_loader.load_kitti import load_kitti_data, plot_kitti_poses
from utils.prepare_input_helper import *
from utils.neural_scene_graph_manipulation import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
from config.utils import read_config
from config.arguments import config_parser
from models.NeRF import *
from models.Render import *

tf.compat.v1.enable_eager_execution()


def train():
    # 命令行参数
    parser = config_parser()
    # 获取参数字典
    args = parser.parse_args()
    # 读取config文件
    args = read_config(args)
    # 如果输入了随机种子
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        # numpy和tensorflow都要初始化随机种子
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)
    # 仅渲染物体/仅渲染背景
    if args.obj_only and args.bckg_only:
        print('Object and background can not set as train only at the same time.')
        return
    # 如果仅渲染物体/背景，使用物体属性
    if args.bckg_only or args.obj_only:
        # print('Deactivating object models to increase performance for training the background model only.')
        args.use_object_properties = True

    # Support first and last frame int
    # 获取开始结束帧组
    args.first_frame = str(args.first_frame)
    args.last_frame = str(args.last_frame)
    starts = args.first_frame.split(',')
    ends = args.last_frame.split(',')
    # 开始结束帧个数必须匹配
    if len(starts) != len(ends):
        # 不匹配只取组的首帧
        print('Number of sequences is not defined. Using the first sequence')
        args.first_frame = int(starts[0])
        args.last_frame = int(ends[0])
    else:
        # 列表化
        args.first_frame = [int(val) for val in starts]
        args.last_frame = [int(val) for val in ends]
    # 如果是kitti数据集
    if args.dataset_type == 'kitti':
        # tracking2txt('../../CenterTrack/results/default_0006_results.json')
        # 读取kitti数据集
        # # 返回图片集，双目姿态，单目姿态，图片长宽和焦点，数据集的划分，双目可视物体，所有物体的信息，单目可视物体，None，None，None，None
        # visible_objects: 帧数*相机数，最大物体数 [帧号、相机号、物体类别号、物体类别、长宽高、物体姿态]
        images, poses, render_poses, hwf, i_split, visible_objects, objects_meta, render_objects, bboxes, \
        kitti_obj_metadata, time_stamp, render_time_stamp = \
            load_kitti_data(args.datadir,
                            selected_frames=[args.first_frame, args.last_frame] if args.last_frame else None,
                            use_obj=True,
                            row_id=True,
                            remove=args.remove_frame,
                            use_time=args.use_time,
                            exp=True if 'exp' in args.expname else False)
        print('Loaded kitti', images.shape,
              #render_poses.shape,
              hwf,
              args.datadir)
        # 根据可视物体数来界定物体数目最大输入
        if visible_objects is not None:
            args.max_input_objects = visible_objects.shape[1]
        else:
            args.max_input_objects = 0
        # 如果开启仅渲染
        if args.render_only:
            visible_objects = render_objects
        # 划分数据集
        i_train, i_val, i_test = i_split
        # 远近平面
        near = args.near_plane
        far = args.far_plane

        # Fix all persons at one position
        # 不固定行人
        fix_ped_pose = False
        if fix_ped_pose:
            print('Pedestrians are fixed!')
            ped_poses = np.pad(visible_objects[np.where(visible_objects[..., 3] == 4)][:, 7:11], [[0, 0], [7, 3]])
            visible_objects[np.where(visible_objects[..., 3] == 4)] -= ped_poses
            visible_objects[np.where(visible_objects[..., 3] == 4)] += ped_poses[20]


    # 如果数据是vkitti
    elif args.dataset_type == 'vkitti':
        # TODO: Class by integer instead of hot-one-encoding for latent encoding in visible object
        # 返回图片集，双目姿态，单目姿态，图片长宽和焦点，数据集的划分，双目可视物体，所有物体的信息，单目可视物体，None，None，None，None
        images, instance_segm, poses, frame_id, render_poses, hwf, i_split, visible_objects, objects_meta, render_objects, bboxes = \
            load_vkitti_data(args.datadir,
                             selected_frames=[args.first_frame[0], args.last_frame[0]] if args.last_frame[0] >= 0 else -1,
                             use_obj=args.use_object_properties,
                             row_id=True if args.object_setting == 0 or args.object_setting == 1 else False,)
        render_time_stamp = None

        print('Loaded vkitti', images.shape,
              #render_poses.shape,
              hwf,
              args.datadir)
        if visible_objects is not None:
            args.max_input_objects = visible_objects.shape[1]
        else:
            args.max_input_objects = 0

        i_train, i_val, i_test = i_split

        near = args.near_plane
        far = args.far_plane
    # 其他数据集不支持
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Ploting Options for Debugging the Scene Graph
    # 关闭绘制姿态
    plot_poses = False
    if args.debug_local and plot_poses:
        plot_kitti_poses(args, poses, visible_objects)

    # Cast intrinsics to right types
    # 重整hwf数据类型
    np.linalg.norm(poses[:1, [0, 2], 3] - poses[1:, [0, 2], 3])
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    # 测试集姿态
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Extract objects positions and labels
    # 生成obj信息
    if args.use_object_properties or args.bckg_only:
        # obj_nodes: [frame*cam, n_obj, [tx, ty, tz, raw, track_row]]
        # objects_meta: obj_id,[物体id，l，h，w，类别id]
        obj_nodes, add_input_rows, obj_meta_ls, scene_objects, scene_classes = \
            extract_object_information(args, visible_objects, objects_meta)
        # obj_track_id_list = False if args.single_obj == None else [args.single_obj] #[4., 9.,, 3.] # [9.]
        # 如果单物体, 就把单物体作为列表当作唯一场景物体
        if args.single_obj is not None:
            # Train only a single object
            args.scene_objects = [args.single_obj]
        else:
            args.scene_objects = scene_objects

        args.scene_classes = scene_classes
        # 帧数
        n_input_frames = obj_nodes.shape[0]

        # Prepare object nodes [n_images, n_objects, H, W, add_input_rows, 3]
        obj_nodes = np.reshape(obj_nodes, [n_input_frames, args.max_input_objects * add_input_rows, 3])

        obj_meta_tensor = tf.cast(np.array(obj_meta_ls), tf.float32)

        if args.render_test:
            render_objects = obj_nodes[i_test]

    # Create log dir and copy the config file
    # log——实验名称和存放位置
    basedir = args.basedir
    expname = args.expname
    # 不存在log文件夹就创建
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    # 写入log
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        # 将arg参数写下来
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # 如果config文件不为空
    if args.config is not None:
        # 打开配置文件
        f = os.path.join(basedir, expname, 'config.txt')
        # 把配置文件拷贝一份到log
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf representation models
    # 创建NeRF模型
    render_kwargs_train, render_kwargs_test, start, grad_vars, models, latent_encodings, weights_path = create_nerf(
        args)

    # 仅训练物体，删除背景模型的训练参数
    if args.obj_only:
        print('removed bckg model for obj training')
        del grad_vars[:len(models['model'].trainable_variables)]
        models.pop('model')

    if args.ft_path is not None and args.ft_path != 'None':
        start = 0

    # Set bounds for point sampling along a ray
    if not args.sampling_method == 'planes' and not args.sampling_method == 'planes_plus':
        bds_dict = {
            'near': tf.cast(near, tf.float32),
            'far': tf.cast(far, tf.float32),
        }
    else:
        # TODO: Generalize for non front-facing scenarios
        plane_bds, plane_normal, plane_delta, id_planes, near, far = plane_bounds(
            poses, args.plane_type, near, far, args.N_samples)

        # planes = [plane_origin, plane_normal]
        bds_dict = {
            'near': tf.cast(near, tf.float32),
            'far': tf.cast(far, tf.float32),
            'plane_bds': tf.cast(plane_bds, tf.float32),
            'plane_normal': tf.cast(plane_normal, tf.float32),
            'id_planes': tf.cast(id_planes, tf.float32),
            'delta': tf.cast(plane_delta, tf.float32)
        }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if np.argwhere(n[:1,:,0]>0)only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            'test' if args.render_test else 'path', start))
        if args.manipulate is not None:
            testsavedir = testsavedir + '_' + args.manipulate

        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        # Select random from render_poses
        render_poses = render_poses[np.random.randint(0, len(render_poses) - 1, np.minimum(3, len(render_poses)))]

        rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                              obj=obj_nodes if args.use_object_properties and not args.bckg_only else None,
                              obj_meta=obj_meta_tensor if args.use_object_properties else None,
                              gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor,
                              render_manipulation=args.manipulate, rm_obj=args.remove_obj,
                              time_stamp=render_time_stamp)
        print('Done rendering', testsavedir)
        if args.dataset_type == 'vkitti':
            rgbs = rgbs[:, 1:, ...]
            macro_block_size = 2
        else:
            macro_block_size = 16

        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                         to8b(rgbs), fps=30, quality=10, macro_block_size=macro_block_size)

        return

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)

    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)


    N_rand = args.N_rand
    # For random ray batching.
    #
    # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
    # interpreted as,
    #   axis=0: ray origin in world space
    #   axis=1: ray direction in world space
    #   axis=2: observed RGB color of pixel
    print('get rays')
    # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
    # for each pixel in the image. This stack() adds a new dimension.
    rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
    rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
    print('done, concats')
    # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)

    if not args.use_object_properties:
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        if args.use_time:
            time_stamp_train = np.stack([time_stamp[i]
                                   for i in i_train], axis=0)
            time_stamp_train = np.repeat(time_stamp_train[:, None, :], H*W, axis=0).astype(np.float32)
            rays_rgb = np.concatenate([rays_rgb, time_stamp_train], axis=1)

    else:
        print("adding object nodes to each ray")
        rays_rgb_env = rays_rgb
        input_size = 0

        obj_nodes = np.repeat(obj_nodes[:, :, np.newaxis, ...], W, axis=2)
        obj_nodes = np.repeat(obj_nodes[:, :, np.newaxis, ...], H, axis=2)

        obj_size = args.max_input_objects * add_input_rows
        input_size += obj_size
        # [N, ro+rd+rgb+obj_nodes, H, W, 3]
        rays_rgb_env = np.concatenate([rays_rgb_env, obj_nodes], 1)

        # [N, H, W, ro+rd+rgb+obj_nodes*max_obj, 3]
        # with obj_nodes [(x+y+z)*max_obj + (track_id+is_training+0)*max_obj]
        rays_rgb_env = np.transpose(rays_rgb_env, [0, 2, 3, 1, 4])
        rays_rgb_env = np.stack([rays_rgb_env[i]
                                 for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb+ obj_pose*max_obj, 3]
        rays_rgb_env = np.reshape(rays_rgb_env, [-1, 3+input_size, 3])

        rays_rgb = rays_rgb_env.astype(np.float32)
        del rays_rgb_env

        # get all rays intersecting objects
        if (args.bckg_only or args.obj_only or args.model_library is not None or args.use_object_properties): #and not args.debug_local:
            bboxes = None
            print(rays_rgb.shape)

            if args.use_inst_segm:
                # Ray selection from segmentation (early experiments)
                print('Using segmentation map')
                if not args.scene_objects:
                    rays_on_obj = np.where(instance_segm.flatten() > 0)[0]

                else:
                    # B) Single object per scene
                    rays_on_obj = []
                    for obj_track_id in args.scene_objects:
                        rays_on_obj.append(np.where(instance_segm.flatten() == obj_track_id+1)[0])
                    rays_on_obj = np.concatenate(rays_on_obj)
            elif bboxes is not None:
                # Ray selection from 2D bounding boxes (early experiments)
                print('Using 2D bounding boxes')
                rays_on_obj = get_bbox_pixel(bboxes, i_train, hwf)
            else:
                # Preferred option
                print('Using Ray Object Node intersections')
                rays_on_obj, rays_to_remove = get_all_ray_3dbox_intersection(rays_rgb, obj_meta_tensor,
                                                                             args.netchunk, local=args.debug_local,
                                                                             obj_to_remove=args.remove_obj)

            # Create Masks for background and objects to subsample the training batches
            obj_mask = np.zeros(len(rays_rgb), np.bool)
            obj_mask[rays_on_obj] = 1

            bckg_mask = np.ones(len(rays_rgb), np.bool)
            bckg_mask[rays_on_obj] = 0

            # Remove predefined objects from the scene
            if len(rays_to_remove) > 0 and args.remove_obj is not None:
                print('Removing obj ', args.remove_obj)
                # Remove rays from training set
                remove_mask = np.zeros(len(rays_rgb), np.bool)
                remove_mask[rays_to_remove] = 1
                obj_mask[remove_mask] = 0
                # Remove objects from graph
                rays_rgb = remove_obj_from_set(rays_rgb, np.array(obj_meta_ls), args.remove_obj)
                obj_nodes = np.reshape(np.transpose(obj_nodes, [0, 2, 3, 1, 4]), [-1, args.max_input_objects*2, 3])
                obj_nodes = remove_obj_from_set(obj_nodes, np.array(obj_meta_ls), args.remove_obj)
                obj_nodes = np.reshape(obj_nodes, [len(images), H, W, args.max_input_objects*2, 3])
                obj_nodes = np.transpose(obj_nodes, [0, 3, 1, 2, 4])

            # Debugging options to display selected rays/pixels
            debug_pixel_selection = False
            if args.debug_local and debug_pixel_selection:
                for i_smplimg in range(len(i_train)):
                    rays_rgb_debug = np.array(rays_rgb)
                    rays_rgb_debug[rays_on_obj, :] += np.random.rand(3) #0.
                    # rays_rgb_debug[remove_mask, :] += np.random.rand(3)
                    plt.figure()
                    img_sample = np.reshape(rays_rgb_debug[(H * W) * i_smplimg:(H * W) * (i_smplimg + 1), 2, :],
                                            [H, W, 3])
                    plt.imshow(img_sample)

            if args.bckg_only:
                print('Removing objects from scene.')
                rays_rgb = rays_rgb[bckg_mask]
                print(rays_rgb.shape)
            elif args.obj_only and args.model_library is None or args.debug_local:
                print('Extracting objects from background.')
                rays_bckg = None
                rays_rgb = rays_rgb[obj_mask]
                print(rays_rgb.shape)
            else:
                rays_bckg = rays_rgb[bckg_mask]
                rays_rgb = rays_rgb[obj_mask]

            # Get Intersections per object and additional rays to have similar rays/object distributions VVVVV
            if not args.bckg_only:
                # # print(rays_rgb.shape)
                rays_rgb = resample_rays(rays_rgb, rays_bckg, obj_meta_tensor, objects_meta,
                                         args.scene_objects, scene_classes, args.chunk, local=args.debug_local)
            # Get Intersections per object and additional rays to have similar rays/object distributions AAAAA

    print('shuffle rays')
    np.random.shuffle(rays_rgb)
    print('done')
    i_batch = 0

    N_iters = 1000000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        batch_obj = None

        # Random over all images
        if not args.use_object_properties:
            # No object specific representations
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = tf.transpose(batch, [1, 0, 2])

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0

        batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1+max_obj, 3*?]
        batch = tf.transpose(batch, [1, 0, 2])

        # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
        # target_s[n, rgb] = example_id, observed color.
        batch_rays, target_s, batch_dyn = batch[:2], batch[2], batch[3:]

        if args.use_time:
            batch_time = batch_dyn
        else:
            batch_time = None

        if args.use_object_properties:
            # batch_obj[N_rand, max_obj, properties+0]
            batch_obj_dyn = tf.reshape(tf.transpose(
                batch_dyn, [1, 0, 2]), [batch.get_shape()[1], args.max_input_objects, add_input_rows*3])


            # xyz + roty
            batch_obj = batch_obj_dyn[..., :4]

            # [N_rand, max_obj, trackID + label + model + color + Dimension]
            # Extract static nodes and edges (latent node, id, box size) for each object at each ray
            batch_obj_metadata = tf.gather(obj_meta_tensor, tf.cast(batch_obj_dyn[:, :, 4], tf.int32), axis=0)

            batch_track_id = batch_obj_metadata[:, :, 0]
            # TODO: For generalization later Give track ID in the beginning and change model name to track ID
            batch_obj = tf.concat([batch_obj, batch_track_id[..., None]], axis=-1)
            batch_dim = batch_obj_metadata[:, :, 1:4]
            batch_label = batch_obj_metadata[:, :, 4][..., tf.newaxis]

            batch_obj = tf.concat([batch_obj, batch_dim, batch_label], axis=-1)


            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0


        #####  Core optimization loop  #####

        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays, obj=batch_obj, time_stamp=batch_time,
                verbose=i < 10, retraw=True, **render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

            # Add loss for latent code
            if args.latent_size > 0:
                reg = 1/args.latent_balance    # 1/0.01
                latent_reg = latentReg(list(render_kwargs_train['latent_vector_dict'].values()), reg)
                loss += latent_reg

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        dt = time.time()-time0
        #####           end            #####

        # Rest is logging

        def save_weights(weights, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, weights)
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k].get_weights(), k, i)
            if args.latent_size > 0:
                for k in latent_encodings:
                    save_weights(latent_encodings[k].numpy(), k, i)

        if i % args.i_print == 0 or i < 10:
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                # if args.N_importance > 0:
                #     tf.contrib.summary.scalar('psnr0', psnr0)
                # else:
                #     tf.contrib.summary.histogram('tran', trans)

                if args.latent_size > 0:
                    for latent_vector_sum in list(render_kwargs_train['latent_vector_dict'].values()):
                        tf.contrib.summary.histogram(
                            latent_vector_sum.name,
                            latent_vector_sum.value(),
                        )

            if i % args.i_img == 0 and not i == 0: # and not args.debug_local:

                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3, :4]
                time_st = time_stamp[img_i] if args.use_time else None

                if args.use_object_properties:
                    obj_i = obj_nodes[img_i, :, 0, 0, ...]
                    obj_i = tf.cast(obj_i, tf.float32)
                    obj_i = tf.reshape(obj_i, [args.max_input_objects, obj_i.shape[0] // args.max_input_objects * 3])

                    obj_i_metadata = tf.gather(obj_meta_tensor, tf.cast(obj_i[:, 4], tf.int32),
                                                       axis=0)
                    batch_track_id = obj_i_metadata[..., 0]
                    obj_i_dim = obj_i_metadata[:, 1:4]
                    obj_i_label = obj_i_metadata[:, 4][:, tf.newaxis]

                    # xyz + roty
                    obj_i = obj_i[..., :4]
                    obj_i = tf.concat([obj_i, batch_track_id[..., None], obj_i_dim, obj_i_label], axis=-1)

                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, obj=obj_i,
                                                    **render_kwargs_test)
                else:
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, time_stamp=time_st,
                                                    **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))
                
                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                if i==0 or not os.path.exists(testimgdir):
                    os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(os.path.join(testimgdir, '{:06d}.png'.format(i)), to8b(rgb))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image(
                        'disp', disp[tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image(
                        'acc', acc[tf.newaxis, ..., tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])

                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image(
                            'rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image(
                            'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                        tf.contrib.summary.image(
                            'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])

        global_step.assign_add(1)


if __name__ == '__main__':
    train()