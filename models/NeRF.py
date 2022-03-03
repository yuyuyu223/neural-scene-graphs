import imp
import tensorflow as tf
from models.Render import batchify
import os
from utils.prepare_input_helper import *
from utils.neural_scene_graph_manipulation import *
from utils.neural_scene_graph_helper import *

def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    """
        NeRF多层感知机的实现
    """
    # 如果是物体检测任务，关闭训练
    if args.obj_detection:
        trainable = False
    else:
        trainable = True
    # 根据位置编码设置，生成embed层和位置编码的输入维数
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    # 如果加入时序信息，输入多一个通道
    if args.use_time:
        input_ch += 1

    input_ch_views = 0
    embeddirs_fn = None
    # 如果使用光线方向
    if args.use_viewdirs:
        # multires_views=4：2D的方向每个都sin/cos，返回viewdirs的编码后的输入维数
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    # 初始化nerf模型，input: [xyz，time]，[viewdir]的编码
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_color_head=input_ch_views, use_viewdirs=args.use_viewdirs, trainable=trainable)
    # 可训练参数
    grad_vars = model.trainable_variables
    # 模型信息
    models = {'model': model}

    model_fine = None
    # 沿着一条光线更多次采样
    if args.N_importance > 0:
        # 建立fine网络
        model_fine = init_nerf_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_color_head=input_ch_views, use_viewdirs=args.use_viewdirs, trainable=trainable)
        # 添加训练参数
        grad_vars += model_fine.trainable_variables
        # 模型信息
        models['model_fine'] = model_fine

    models_dynamic_dict = None
    embedobj_fn = None
    # 记录latentcode
    latent_vector_dict = None if args.latent_size < 1 else {}
    latent_encodings = None if args.latent_size < 1 else {}

    if args.use_object_properties and not args.bckg_only:
        models_dynamic_dict = {}
        # obj位置的编码
        embedobj_fn, input_ch_obj = get_embedder(
            args.multires_obj, -1 if args.multires_obj == -1 else args.i_embed, input_dims=3)

        # Version a: One Network per object
        if args.latent_size < 1:
            # xyz编码维度
            input_ch = input_ch
            # viewdirs编码维度
            input_ch_color_head = input_ch_views
            # Don't add object location input for setting 1
            if args.object_setting != 1:
                # obj位置obj_x,obj_y,obj_z的编码
                input_ch_color_head += input_ch_obj
            # TODO: Change to number of objects in Frames
            # 遍历场景下的所有物体
            for object_i in args.scene_objects:
                # 定义模型名称
                model_name = 'model_obj_' + str(int(object_i)) # .zfill(5)
                # input: xyz
                # input_ch_color_head: viewdirs+obj_pos
                model_obj = init_nerf_model(
                    D=args.netdepth_fine, W=args.netwidth_fine,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_color_head=input_ch_color_head, use_viewdirs=args.use_viewdirs,trainable=trainable)
                    # latent_size=args.latent_size)

                grad_vars += model_obj.trainable_variables
                models[model_name] = model_obj
                models_dynamic_dict[model_name] = model_obj

        # Version b: One Network for all similar objects of the same class
        else:
            input_ch = input_ch + args.latent_size
            input_ch_color_head = input_ch_views
            # Don't add object location input for setting 1
            if args.object_setting != 1:
                input_ch_color_head += input_ch_obj

            for obj_class in args.scene_classes:
                model_name = 'model_class_' + str(int(obj_class)).zfill(5)

                model_obj = init_nerf_model(
                    D=args.netdepth_fine, W=args.netwidth_fine,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_color_head=input_ch_color_head,
                    # input_ch_shadow_head=input_ch_obj,
                    use_viewdirs=args.use_viewdirs, trainable=trainable)
                    # use_shadows=args.use_shadows,
                    # latent_size=args.latent_size)

                grad_vars += model_obj.trainable_variables
                models[model_name] = model_obj
                models_dynamic_dict[model_name] = model_obj

            for object_i in args.scene_objects:
                name = 'latent_vector_obj_'+str(int(object_i)).zfill(5)
                latent_vector_obj = init_latent_vector(args.latent_size, name)
                grad_vars.append(latent_vector_obj)

                latent_encodings[name] = latent_vector_obj
                latent_vector_dict[name] = latent_vector_obj

    # TODO: Remove object embedding function
    def network_query_fn(inputs, viewdirs, network_fn): return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        embedobj_fn=embedobj_fn,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'N_samples_obj': args.N_samples_obj,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'object_network_fn_dict': models_dynamic_dict,
        'latent_vector_dict': latent_vector_dict if latent_vector_dict is not None else None,
        'N_obj': args.max_input_objects if args.use_object_properties and not args.bckg_only else False,
        'obj_only': args.obj_only,
        'obj_transparency': not args.obj_opaque,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'sampling_method': args.sampling_method,
        'use_time': args.use_time,
        'obj_location': False if args.object_setting == 1 else True,
    }

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    # render_kwargs_test['obj_only'] = False

    start = 0
    basedir = args.basedir
    expname = args.expname
    weights_path = None

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    elif args.model_library is not None and args.model_library != 'None':
        obj_ckpts = {}
        ckpts = []
        for f in sorted(os.listdir(args.model_library)):
            if 'model_' in f and 'fine' not in f and 'optimizer' not in f and 'obj' not in f:
                ckpts.append(os.path.join(args.model_library, f))
            if 'obj' in f and float(f[10:][:-11]) in args.scene_objects:
                obj_ckpts[f[:-11]] = (os.path.join(args.model_library, f))
    elif args.obj_only:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('_obj_' in f)]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f and 'obj' not in f and 'class' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload and (not args.obj_only or args.model_library):
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

        if models_dynamic_dict is not None:
            for model_dyn_name, model_dyn in models_dynamic_dict.items():
                if args.model_library:
                    ft_weights_obj = obj_ckpts[model_dyn_name]
                else:
                    ft_weights_obj = '{}'.format(ft_weights[:-16]) + \
                                     model_dyn_name + '_{}'.format(ft_weights[-10:])
                print('Reloading model from', ft_weights_obj, 'for', model_dyn_name[6:])
                model_dyn.set_weights(np.load(ft_weights_obj, allow_pickle=True))

        if latent_vector_dict is not None:
            for latent_vector_name, latent_vector in latent_vector_dict.items():
                ft_weights_obj = '{}'.format(ft_weights[:-16]) + \
                                     latent_vector_name + '_{}'.format(ft_weights[-10:])
                print('Reloading objects latent vector from', ft_weights_obj)
                latent_vector.assign(np.load(ft_weights_obj, allow_pickle=True))

    elif len(ckpts) > 0 and args.obj_only:
        ft_weights = ckpts[-1]
        start = int(ft_weights[-10:-4]) + 1
        ft_weights_obj_dir = os.path.split(ft_weights)[0]
        for model_dyn_name, model_dyn in models_dynamic_dict.items():
            ft_weights_obj = os.path.join(ft_weights_obj_dir, model_dyn_name + '_{}'.format(ft_weights[-10:]))
            print('Reloading model from', ft_weights_obj, 'for', model_dyn_name[6:])
            model_dyn.set_weights(np.load(ft_weights_obj, allow_pickle=True))

        if latent_vector_dict is not None:
            for latent_vector_name, latent_vector in latent_vector_dict.items():
                ft_weights_obj = os.path.join(ft_weights_obj_dir, latent_vector_name + '_{}'.format(ft_weights[-10:]))
                print('Reloading objects latent vector from', ft_weights_obj)
                latent_vector.assign(np.load(ft_weights_obj, allow_pickle=True))

        weights_path = ft_weights

    if args.model_library:
        start = 0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models, latent_encodings, weights_path


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, embedobj_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = tf.reshape(inputs[..., :3], [-1, 3])

    embedded = embed_fn(inputs_flat)
    if inputs.shape[-1] > 3:
        if inputs.shape[-1] == 4:
            # NeRF + T w/o embedding
            time_st = tf.reshape(inputs[..., 3], [inputs_flat.shape[0], -1])
            embedded = tf.concat([embedded, time_st], -1)
        else:
            # NeRF + Latent Code
            inputs_latent = tf.reshape(inputs[..., 3:], [inputs_flat.shape[0], -1])
            embedded = tf.concat([embedded, inputs_latent], -1)

    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None, :3], inputs[..., :3].shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)

        if viewdirs.shape[-1] > 3:
            # Use global locations of objects
            input_obj_pose = tf.broadcast_to(viewdirs[:, None, 3:],
                                             shape=[inputs[..., :3].shape[0], inputs[..., :3].shape[1], 3])
            input_obj_pose_flat = tf.reshape(input_obj_pose, [-1, input_obj_pose.shape[-1]])
            embedded_obj = embedobj_fn(input_obj_pose_flat)
            embedded = tf.concat([embedded, embedded_obj], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs