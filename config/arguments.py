
def config_parser():
    """
        命令行参数
    """
    import configargparse
    parser = configargparse.ArgumentParser()
    # 配置文件路径
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    # 实验名称
    parser.add_argument("--expname", type=str, help='experiment name')
    # log存放位置
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    # 数据集位置
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    # 网络层数
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    # 每层通道数
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    # fine网络层数
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    # fine网络每层通道数
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    # 每批光线数
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    # 学习率
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    # 学习率递减
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    # 光线计算并行数
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    # 采样点计算并行数
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    # Disabled and not implemented for Neural Scene Graphs
    # 只对一张图采样光线
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    # 不读取模型ckpt
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    # 粗糙网络的npy权重路径
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    # 前后背景预训练模型权重
    parser.add_argument("--model_library", type=str, default=None,
                        help='specific weights npy file to load pretrained background and foreground models')
    # 随机种子
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    # 采样方法: None / lindisp / squaredist / plane
    parser.add_argument("--sampling_method", type=str, default=None,
                        help='method to sample points along the ray options: None / lindisp / squaredist / plane')
    # 第二阶段去模糊的图像裁剪尺寸
    parser.add_argument("--crop_size", type=int, default=16,
                        help='size of crop image for second stage deblurring')
    # 仅渲染背景
    parser.add_argument("--bckg_only", action='store_true',
                        help='removes rays associated with objects from the training set to train just the background model.')
    # 仅渲染物体
    parser.add_argument("--obj_only", action='store_true',
                        help='Train object models on rays close to the objects only.')
    # 用分割图选光线
    parser.add_argument("--use_inst_segm", action='store_true',
                        help='Use an instance segmentation map to select a subset from all sampled rays')
    # latentcode的大小
    parser.add_argument("--latent_size", type=int, default=0,
                        help='Size of the latent vector representing each of object of a class. If 0 no latent vector '
                             'is applied and a single representation per object is used.')
    # latent loss的权重
    parser.add_argument("--latent_balance", type=float, default=0.01,
                        help="Balance between image loss and latent loss")

    # pre-crop options
    # 使用中心crop训练的step和比例
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')    

    # rendering options
    # 一条光线的采样点个数
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_samples_obj", type=int, default=3,
                        help='number of samples per ray and object')
    # 光线上额外的精细采样点个数
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    # 抖动
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    # 使用5D（包含viewdirs）
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    # 使用物体姿势预测阴影的不透明度
    parser.add_argument("--use_shadows", action='store_true',
                        help='use pose of an object to predict shadow opacity')
    # 位置编码方式
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    # 位置编码长度
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_obj", type=int, default=4,
                        help='log2 of max freq for positional encoding (3D object location + heading)')
    # 噪声的标准差
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    # 使用时间参数
    parser.add_argument("--use_time", action='store_true',
                        help='time parameter for nerf baseline version')
    # 移除训练集指定帧
    parser.add_argument("--remove_frame", type=int, default=-1,
                        help="Remove the ith frame from the training set")
    # 移除训练集指定物体
    parser.add_argument("--remove_obj", type=int, default=None,
                        help="Option to remove all pixels of an object from the training")

    # render flags
    # 仅渲染，不更新模型
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    # 渲染测试集
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    # 渲染降采样比例
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    # 白色背景
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    # 仅渲染的操作方法
    parser.add_argument("--manipulate", type=str, default=None,
                        help='Renderonly manipulation argument')

    # dataset options
    # 数据集类型
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels / vkitti')
    # 测试集验证集比例采样
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    # 半分辨率
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')
    # LLFF图像降采样比例
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    # 训练时降采样比例
    parser.add_argument("--training_factor", type=int, default=0,
                        help='downsample factor for all images during training')
    # 不要使用标准化设备坐标（为非前向场景设置）
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    # 在差异而非深度上进行线性采样
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    # 360°场景
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    # 1/N的LLFF作为测试集
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # vkitti/kitti flags
    # 起始帧
    parser.add_argument("--first_frame", type=str, default="0",
                        help='specifies the beginning of a sequence if not the complete scene is taken as Input')
    # 结束帧
    parser.add_argument("--last_frame", type=str, default="1",
                        help='specifies the end of a sequence')
    # 使用物体属性
    parser.add_argument("--use_object_properties", action='store_true',
                        help='use pose and properties of visible objects as an Input')
    # 设置物体需要使用的属性
    parser.add_argument("--object_setting", type=int, default=0,
                        help='specify which properties are used')
    # 最大物体个数
    parser.add_argument("--max_input_objects", type=int, default=20,
                        help='Max number of object poses considered by the network, will be set automatically')
    # 训练的场景的所有物体列表
    parser.add_argument("--scene_objects", type=list,
                        help='List of all objects in the trained sequence')
    # 训练场景下所有的物体分类
    parser.add_argument("--scene_classes", type=list,
                        help='List of all unique classes in the trained sequence')
    # 如果为true，则光线在与第一个对象bbox相交后停止
    parser.add_argument("--obj_opaque", action='store_true',
                        help='Ray does stop after intersecting with the first object bbox if true')
    parser.add_argument("--single_obj", type=float, default=None,
                        help='Specify for sequential training.')
    # 框包含阴影的最大缩放
    parser.add_argument("--box_scale", type=float, default=1.0,
                        help="Maximum scale for boxes to include shadows")
    # 平面采样方法
    parser.add_argument("--plane_type", type=str, default='uniform',
                        help='specifies how the planes are sampled')
    # 远近平面
    parser.add_argument("--near_plane", type=float, default=0.5,
                        help='specifies the distance from the last pose to the far plane')
    parser.add_argument("--far_plane", type=float, default=150.,
                        help='specifies the distance from the last pose to the far plane')

    # logging/saving options
    # log打印频率
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    # tensorboard记录频率
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    # 模型保存频率
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')

    # Object Detection through rendering
    parser.add_argument("--obj_detection", action='store_true',
                        help='Debug local')
    parser.add_argument("--frame_number", type=int, default=0,
                        help='Frame of the datadir which should be detected')

    # Local Debugging
    parser.add_argument("--debug_local", action='store_true',
                        help='Debug local')

    return parser