import tensorflow as tf
import numpy as np
import time
from utils.prepare_input_helper import *
from utils.neural_scene_graph_manipulation import *
from utils.neural_scene_graph_helper import *

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                N_samples_obj,
                retraw=False,
                perturb=1.,
                N_importance=0,
                network_fine=None,
                object_network_fn_dict=None,
                latent_vector_dict=None,
                N_obj=None,
                obj_only=False,
                obj_transparency=True,
                white_bkgd=False,
                raw_noise_std=0.,
                sampling_method=None,
                use_time=False,
                plane_bds=None,
                plane_normal=None,
                delta=0.,
                id_planes=0,
                verbose=False,
                obj_location=True):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      object_network_fn_dict: dictinoary of functions. Model for predicting RGB and density at each point in
        object frames
      latent_vector_dict: Dictionary of latent codes
      N_obj: Maximumn amount of objects per ray
      obj_only: bool. If True, only run models from object_network_fn_dict
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      sampling_mehtod: string. Select how points are sampled in space
      plane_bds: array of shape [2, 3]. If sampling method planes, descirbing the first and last plane in space.
      plane_normal: array of shape [3]. Normal of all planes
      delta: float. Distance between adjacent planes.
      id_planes: array of shape [N_samples]. Preselected planes for sampling method planes and a given sampling distribution
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to 
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * \
            tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1./tf.maximum(1e-10, depth_map /
                                 tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map


    def sample_along_ray(near, far, N_samples, N_rays, sampling_method, perturb):
        # Sample along each ray given one of the sampling methods. Under the logic, all rays will be sampled at
        # the same times.
        t_vals = tf.linspace(0., 1., N_samples)
        if sampling_method == 'squareddist':
            z_vals = near * (1. - np.square(t_vals)) + far * (np.square(t_vals))
        elif sampling_method == 'lindisp':
            # Sample linearly in inverse depth (disparity).
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
        else:
            # Space integration times linearly between 'near' and 'far'. Same
            # integration points will be used for all rays.
            z_vals = near * (1.-t_vals) + far * (t_vals)
            if sampling_method == 'discrete':
                perturb = 0

        # Perturb sampling time along each ray. (vanilla NeRF option)
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = tf.concat([mids, z_vals[..., -1:]], -1)
            lower = tf.concat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = tf.random.uniform(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

        return tf.broadcast_to(z_vals, [N_rays, N_samples]), perturb


    ###############################
    # batch size
    N_rays = int(ray_batch.shape[0])

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, 8:11] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    if use_time:
        time_stamp = ray_batch[:, 11][:, tf.newaxis]

    # Extract object position, dimension and label
    if N_obj:
        obj_pose = ray_batch[:, 11:]
        # [N_rays, N_obj, 8] with 3D position, y rot angle, track_id, (3D dimension - length, height, width)
        obj_pose = tf.reshape(obj_pose, [N_rays, N_obj, obj_pose.shape[-1] // N_obj])
        if N_importance > 0:
            obj_pose_fine = tf.repeat(obj_pose[:, tf.newaxis, ...], N_importance + N_samples, axis=1)
    else:
        obj_pose = obj_pose_fine = None

    if not obj_only:
        # For training object models only sampling close to the objects is performed
        if (sampling_method == 'planes' or sampling_method == 'planes_plus') and plane_bds is not None:
            # Sample at ray plane intersection (Neural Scene Graphs)
            pts, z_vals = plane_pts([rays_o, rays_d], [plane_bds, plane_normal, delta], id_planes, near,
                                    method=sampling_method)
            N_importance = 0
        else:
            # Sample along ray (vanilla NeRF)
            z_vals, perturb = sample_along_ray(near, far, N_samples, N_rays, sampling_method, perturb)

            pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                z_vals[..., :, None]  # [N_rays, N_samples, 3]

    ####### DEBUG Sampling Points
    # print('TURN OFF IF NOT DEBUGING!')
    # axes_ls = plt.figure(1).axes
    # for i in range(rays_o.shape[0]):
    #     plt.arrow(np.array(rays_o)[i, 0], np.array(rays_o)[i, 2],
    #               np.array(30 * rays_d)[i, 0],
    #               np.array(30 * rays_d)[i, 2], color='red')
    #
    # plt.sca(axes_ls[1])
    # for i in range(rays_o.shape[0]):
    #     plt.arrow(np.array(rays_o)[i, 2], np.array(rays_o)[i, 1],
    #               np.array(30 * rays_d)[i, 2],
    #               np.array(30 * rays_d)[i, 1], color='red')
    #
    # plt.sca(axes_ls[2])
    # for i in range(rays_o.shape[0]):
    #     plt.arrow(np.array(rays_o)[i, 0], np.array(rays_o)[i, 1],
    #               np.array(30 * rays_d)[i, 0],
    #               np.array(30 * rays_d)[i, 1], color='red')
    ####### DEBUG Sampling Points

    # Choose input options
    if not N_obj:
        # No objects
        if use_time:
            # Time parameter input
            time_stamp_fine = tf.repeat(time_stamp[:, tf.newaxis], N_importance + N_samples,
                                        axis=1) if N_importance > 0 else None
            time_stamp = tf.repeat(time_stamp[:, tf.newaxis], N_samples, axis=1)
            pts = tf.concat([pts, time_stamp], axis=-1)
            raw = network_query_fn(pts, viewdirs, network_fn)
        else:
            raw = network_query_fn(pts, viewdirs, network_fn)
    else:
        n_intersect = None
        if not obj_pose.shape[-1] > 5:
            # If no object dimension is given all points in the scene given in object coordinates will be used as an input to each object model
            pts_obj, viewdirs_obj = world2object(pts, viewdirs, obj_pose[..., :3], obj_pose[..., 3],
                                                 dim=obj_pose[..., 5:8] if obj_pose.shape[-1] > 5 else None)

            pts_obj = tf.transpose(tf.reshape(pts_obj, [N_rays, N_samples, N_obj, 3]), [0, 2, 1, 3])

            inputs = tf.concat([pts_obj, tf.repeat(obj_pose[..., None, :3], N_samples, axis=2)], axis=3)
        else:
            # If 3D bounding boxes are given get intersecting rays and intersection points in scaled object frames
            pts_box_w, viewdirs_box_w, z_vals_in_w, z_vals_out_w,\
            pts_box_o, viewdirs_box_o, z_vals_in_o, z_vals_out_o, \
            intersection_map = box_pts(
                [rays_o, rays_d], obj_pose[..., :3], obj_pose[..., 3], dim=obj_pose[..., 5:8],
                one_intersec_per_ray=not obj_transparency)

            if z_vals_in_o is None or len(z_vals_in_o) == 0:
                if obj_only:
                    # No computation necesary if rays are not intersecting with any objects and no background is selected
                    raw = tf.zeros([N_rays, 1, 4])
                    z_vals = tf.zeros([N_rays, 1])

                    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
                        raw, z_vals, rays_d)

                    rgb_map = tf.ones([N_rays, 3])
                    disp_map = tf.ones([N_rays])*1e10
                    acc_map = tf.zeros([N_rays])
                    depth_map = tf.zeros([N_rays])

                    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
                    if retraw:
                        ret['raw'] = raw
                    return ret
                else:
                    # TODO: Do not return anything for no intersections.
                    z_vals_obj_w = tf.zeros([1])
                    intersection_map = tf.cast(tf.zeros([1, 3]), tf.int32)

            else:
                n_intersect = z_vals_in_o.shape[0]

                obj_pose = tf.gather_nd(obj_pose, intersection_map)
                obj_pose = tf.repeat(obj_pose[:, tf.newaxis, :], N_samples_obj, axis=1)
                # Get additional model inputs for intersecting rays
                if N_samples_obj > 1:
                    z_vals_box_o = tf.repeat(tf.linspace(0., 1., N_samples_obj)[tf.newaxis, :], n_intersect, axis=0) * \
                                   (z_vals_out_o - z_vals_in_o)[:, tf.newaxis]
                else:
                    z_vals_box_o = tf.repeat(tf.constant(1/2)[tf.newaxis,tf.newaxis], n_intersect, axis=0) * \
                                   (z_vals_out_o - z_vals_in_o)[:, tf.newaxis]

                pts_box_samples_o = pts_box_o[:, tf.newaxis, :] + viewdirs_box_o[:, tf.newaxis, :] \
                                        * z_vals_box_o[..., tf.newaxis]
                # pts_box_samples_o = pts_box_samples_o[:, tf.newaxis, ...]
                # pts_box_samples_o = tf.reshape(pts_box_samples_o, [-1, 3])

                obj_pose_transform = tf.reshape(obj_pose, [-1, obj_pose.shape[-1]])

                pts_box_samples_w, _ = world2object(tf.reshape(pts_box_samples_o, [-1, 3]), None,
                                                    obj_pose_transform[..., :3],
                                                    obj_pose_transform[..., 3],
                                                    dim=obj_pose_transform[..., 5:8] if obj_pose.shape[-1] > 5 else None,
                                                    inverse=True)

                pts_box_samples_w = tf.reshape(pts_box_samples_w, [n_intersect, N_samples_obj, 3])

                z_vals_obj_w = tf.norm(pts_box_samples_w - tf.gather_nd(rays_o, intersection_map[:, 0, tf.newaxis])[:, tf.newaxis, :], axis=-1)

                # else:
                #     z_vals_obj_w = z_vals_in_w[:, tf.newaxis]
                #     pts_box_samples_o = pts_box_o[:, tf.newaxis, :]
                #     pts_box_samples_w = pts_box_w[:, tf.newaxis, :]

                #####
                # print('TURN OFF IF NOT DEBUGING!')
                # axes_ls = plt.figure(1).axes
                # plt.sca(axes_ls[0])
                #
                # pts = np.reshape(pts_box_samples_w, [-1, 3])
                # plt.scatter(pts[:, 0], pts[:, 2], color='red')
                ####

                # Extract objects
                obj_ids = obj_pose[..., 4]
                object_y, object_idx = tf.unique(tf.reshape(obj_pose[..., 4], [-1]))
                # Extract classes
                obj_class = obj_pose[..., 8]
                unique_classes = tf.unique(tf.reshape(obj_class, [-1]))
                class_id = tf.reshape(unique_classes.idx, obj_class.shape)

                inputs = pts_box_samples_o

                if latent_vector_dict is not None:
                    latent_vector_inputs = None

                    for y, obj_id in enumerate(object_y):
                        indices = tf.where(tf.equal(object_idx, y))
                        latent_vector = latent_vector_dict['latent_vector_obj_' + str(int(obj_id)).zfill(5)][tf.newaxis, :]
                        latent_vector = tf.repeat(latent_vector, indices.shape[0], axis=0)

                        latent_vector = tf.scatter_nd(indices, latent_vector, [n_intersect*N_samples_obj, latent_vector.shape[-1]])

                        if latent_vector_inputs is None:
                            latent_vector_inputs = latent_vector
                        else:
                            latent_vector_inputs += latent_vector

                    latent_vector_inputs = tf.reshape(latent_vector_inputs, [n_intersect, N_samples_obj, -1])
                    inputs = tf.concat([inputs, latent_vector_inputs], axis=2)

                # inputs = tf.concat([inputs, obj_pose[..., :3]], axis=-1)

                # objdirs = tf.concat([tf.cos(obj_pose[:, 0, 3, tf.newaxis]), tf.sin(obj_pose[:, 0, 3, tf.newaxis])], axis=1)
                # objdirs = objdirs / tf.reduce_sum(objdirs, axis=1)[:, tf.newaxis]
                # viewdirs_obj = tf.concat([viewdirs_box_o, obj_pose[..., :3][:, 0, :], objdirs], axis=1)
                if obj_location:
                    viewdirs_obj = tf.concat([viewdirs_box_o, obj_pose[..., :3][:, 0, :]], axis=1)
                else:
                    viewdirs_obj = viewdirs_box_o

        if not obj_only:
            # Get integration step for all models
            z_vals, id_z_vals_bckg, id_z_vals_obj = combine_z(z_vals,
                                                              z_vals_obj_w if z_vals_in_o is not None else None,
                                                              intersection_map,
                                                              N_rays,
                                                              N_samples,
                                                              N_obj,
                                                              N_samples_obj, )
        else:
            z_vals, _, id_z_vals_obj = combine_z(None, z_vals_obj_w, intersection_map, N_rays, N_samples, N_obj,
                                                 N_samples_obj)


        if not obj_only:
            # Run background model
            raw = tf.zeros([N_rays, N_samples + N_obj*N_samples_obj, 4])
            raw_sh = raw.shape
            # Predict RGB and density from background
            raw_bckg = network_query_fn(pts, viewdirs, network_fn)
            raw += tf.scatter_nd(id_z_vals_bckg, raw_bckg, raw_sh)
        else:
            raw = tf.zeros([N_rays, N_obj*N_samples_obj, 4])
            raw_sh = raw.shape

        if z_vals_in_o is not None and len(z_vals_in_o) != 0:
            # Loop for one model per object and no latent representations
            if latent_vector_dict is None:
                obj_id = tf.reshape(object_idx, obj_pose[..., 4].shape)
                for k, track_id in enumerate(object_y):
                    if track_id >= 0:
                        input_indices = tf.where(tf.equal(obj_id, k))
                        input_indices = tf.reshape(input_indices, [-1, N_samples_obj, 2])
                        model_name = 'model_obj_' + str(np.array(track_id).astype(np.int32))
                        # print('Hit', model_name, n_intersect, 'times.')
                        if model_name in object_network_fn_dict:
                            obj_network_fn = object_network_fn_dict[model_name]

                            inputs_obj_k = tf.gather_nd(inputs, input_indices)
                            viewdirs_obj_k = tf.gather_nd(viewdirs_obj, input_indices[..., None, 0]) if N_samples_obj == 1 else \
                                tf.gather_nd(viewdirs_obj, input_indices[..., None,0, 0])

                            # Predict RGB and density from object model
                            raw_k = network_query_fn(inputs_obj_k, viewdirs_obj_k, obj_network_fn)

                            if n_intersect is not None:
                                # Arrange RGB and denisty from object models along the respective rays
                                raw_k = tf.scatter_nd(input_indices[:, :], raw_k, [n_intersect, N_samples_obj, 4]) # Project the network outputs to the corresponding ray
                                raw_k = tf.scatter_nd(intersection_map[:, :2], raw_k, [N_rays, N_obj, N_samples_obj, 4]) # Project to rays and object intersection order
                                raw_k = tf.scatter_nd(id_z_vals_obj, raw_k, raw_sh) # Reorder along z and ray
                            else:
                                raw_k = tf.scatter_nd(input_indices[:, 0][..., tf.newaxis], raw_k, [N_rays, N_samples, 4])

                            # Add RGB and density from object model to the background and other object predictions
                            raw += raw_k
            # Loop over classes c and evaluate each models f_c for all latent object describtor
            else:
                for c, class_type in enumerate(unique_classes.y):
                    # Ignore background class
                    if class_type >= 0:
                        input_indices = tf.where(tf.equal(class_id, c))
                        input_indices = tf.reshape(input_indices, [-1, N_samples_obj, 2])
                        model_name = 'model_class_' + str(int(np.array(class_type))).zfill(5)

                        if model_name in object_network_fn_dict:
                            obj_network_fn = object_network_fn_dict[model_name]

                            inputs_obj_c = tf.gather_nd(inputs, input_indices)

                            # Legacy version 2
                            # latent_vector = tf.concat([
                            #         latent_vector_dict['latent_vector_' + str(int(obj_id)).zfill(5)][tf.newaxis, :]
                            #         for obj_id in np.array(tf.gather_nd(obj_pose[..., 4], input_indices)).astype(np.int32).flatten()],
                            #         axis=0)
                            # latent_vector = tf.reshape(latent_vector, [inputs_obj_k.shape[0], inputs_obj_k.shape[1], -1])
                            # inputs_obj_k = tf.concat([inputs_obj_k, latent_vector], axis=-1)

                            # viewdirs_obj_k = tf.gather_nd(viewdirs_obj,
                            #                               input_indices[..., 0]) if N_samples_obj == 1 else \
                            #     tf.gather_nd(viewdirs_obj, input_indices)

                            viewdirs_obj_c = tf.gather_nd(viewdirs_obj, input_indices[..., None, 0])[:,0,:]

                            # Predict RGB and density from object model
                            raw_k = network_query_fn(inputs_obj_c, viewdirs_obj_c, obj_network_fn)

                            if n_intersect is not None:
                                # Arrange RGB and denisty from object models along the respective rays
                                raw_k = tf.scatter_nd(input_indices[:, :], raw_k, [n_intersect, N_samples_obj,
                                                                                   4])  # Project the network outputs to the corresponding ray
                                raw_k = tf.scatter_nd(intersection_map[:, :2], raw_k, [N_rays, N_obj, N_samples_obj,
                                                                                       4])  # Project to rays and object intersection order
                                raw_k = tf.scatter_nd(id_z_vals_obj, raw_k, raw_sh)  # Reorder along z in  positive ray direction
                            else:
                                raw_k = tf.scatter_nd(input_indices[:, 0][..., tf.newaxis], raw_k,
                                                      [N_rays, N_samples, 4])

                            # Add RGB and density from object model to the background and other object predictions
                            raw += raw_k
                        else:
                            print('No model ', model_name,' found')



    # raw_2 = render_mot_scene(pts, viewdirs, network_fn, network_query_fn,
    #                  inputs, viewdirs_obj, z_vals_in_o, n_intersect, object_idx, object_y, obj_pose,
    #                  unique_classes, class_id, latent_vector_dict, object_network_fn_dict,
    #                  N_rays,N_samples, N_obj, N_samples_obj,
    #                  obj_only=obj_only)

    # TODO: Reduce computation by removing 0 entrys
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        if sampling_method == 'planes' or sampling_method == 'planes_plus':
            pts, z_vals = plane_pts([rays_o, rays_d], [plane_bds, plane_normal, delta], id_planes, near,
                                    method=sampling_method)
        else:
            # Obtain additional integration times to evaluate based on the weights
            # assigned to colors in the coarse model.
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
            z_samples = tf.stop_gradient(z_samples)

            # Obtain all points to evaluate color, density at.
            z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # Make predictions with network_fine.
        if use_time:
            pts = tf.concat([pts, time_stamp_fine], axis=-1)

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        if not sampling_method == 'planes':
            ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]
    # if latent_vector_dict is not None:
    #     ret['latent_loss'] = tf.reshape(latent_vector, [N_rays, N_samples_obj, -1])

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H,
           W,
           focal,
           chunk=1024*32,
           rays=None,
           c2w=None,
           obj=None,
           time_stamp=None,
           near=0.,
           far=1.,
           use_viewdirs=False,
           c2w_staticcam=None,
           **kwargs):
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      obj: array of shape [batch_size, max_obj, n_obj_nodes]. Scene object's pose and propeties for each
      example in the batch
      time_stamp: bool. If True the frame will be taken into account as an additional input to the network
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    # 如果camera-->world矩阵不为空
    if c2w is not None:
        # special case to render full image
        # rays = tf.random.shuffle(tf.concat([get_rays(H, W, focal, c2w)[0], get_rays(H, W, focal, c2w)[1]], axis=-1))
        # rays_o = rays[..., :3]
        # rays_d = rays[..., 3:]
        # 获取ray在世界坐标下的起始点、方向
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        
        if obj is not None:
            obj = tf.repeat(obj[None, ...], H*W, axis=0)
        if time_stamp is not None:
            time_stamp = tf.repeat(time_stamp[None, ...], H*W, axis=0)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
        tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)

    if time_stamp is not None:
        time_stamp = tf.cast(tf.reshape(time_stamp, [len(rays), -1]), dtype=tf.float32)
        rays = tf.concat([rays, time_stamp], axis=-1)

    if obj is not None:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction, scene objects)
        # obj = tf.cast(tf.reshape(obj, [obj.shape[0], obj.shape[1]*obj.shape[2]]), dtype=tf.float32)
        obj = tf.reshape(obj, [obj.shape[0], obj.shape[1] * obj.shape[2]])
        rays = tf.concat([rays, obj], axis=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)
        # all_ret[k] = tf.reshape(all_ret[k], [k_sh[0], k_sh[1], -1])

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, obj=None, obj_meta=None, gt_imgs=None, savedir=None,
                render_factor=0, render_manipulation=None, rm_obj=None, time_stamp=None):
    # 长、宽、焦距
    H, W, focal = hwf
    # 降采样加速
    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    # 遍历每一帧的相机姿态(camera_to_world)
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        # 对应的时间戳
        if time_stamp is not None:
            time_st = time_stamp[i]
        else:
            time_st = None
        # 如果没有obj
        if obj is None:
            rgb, disp, acc, _ = render(
                H, W, focal, chunk=chunk, c2w=c2w[:3, :4], obj=None, time_stamp=time_st, **render_kwargs)

            rgbs.append(rgb.numpy())
            disps.append(disp.numpy())

            if i == 0:
                print(rgb.shape, disp.shape)

            if gt_imgs is not None and render_factor == 0:
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                print(p)

            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)

            print(i, time.time() - t)
        # 如果有obj
        else:
            # Manipulate scene graph edges
            # rm_obj = [3, 4, 8, 5, 12]
            render_set = manipulate_obj_pose(render_manipulation, np.array(obj), obj_meta, i, rm_obj=rm_obj)


            # Load manual generated scene graphs
            if render_manipulation is not None and 'handcraft' in render_manipulation:
                if str(i).zfill(3) + '.txt' in os.listdir(savedir):
                    print('Reloading', str(i).zfill(3) + '.txt')
                    render_set.pop()
                    loaded_obj_i = []
                    loaded_objs = np.loadtxt(os.path.join(savedir, str(i).zfill(3) + '.txt'))[:, :6]
                    loaded_objs[:, 5] = 0
                    loaded_objs[:, 4] = np.array([np.where(np.equal(obj_meta[:, 0], loaded_objs[j, 4])) for j in range(len(loaded_objs))])[:, 0, 0]
                    loaded_objs = tf.cast(loaded_objs, tf.float32)
                    loaded_obj_i.append(loaded_objs)
                    render_set.append(loaded_obj_i)
                if '02' in render_manipulation:
                    c2w = render_poses[36]
                if '03' in render_manipulation:
                    c2w = render_poses[20]
                if '04' in render_manipulation or '05' in render_manipulation:
                    c2w = render_poses[20]

            render_kwargs['N_obj'] = len(render_set[0][0])

            steps = len(render_set)
            for r, render_set_i in enumerate(render_set):
                t = time.time()
                j = steps * i + r
                obj_i = render_set_i[0]

                if obj_meta is not None:
                    obj_i_metadata = tf.gather(obj_meta, tf.cast(obj_i[:, 4], tf.int32),
                                               axis=0)
                    batch_track_id = obj_i_metadata[..., 0]

                    print("Next Frame includes Objects: ")
                    if batch_track_id.shape[0] > 1:
                        for object_tracking_id in np.array(tf.squeeze(batch_track_id)).astype(np.int32):
                            if object_tracking_id >= 0:
                                print(object_tracking_id)

                    obj_i_dim = obj_i_metadata[:, 1:4]
                    obj_i_label = obj_i_metadata[:, 4][:, tf.newaxis]
                    # xyz + roty
                    obj_i = obj_i[..., :4]

                    obj_i = tf.concat([obj_i, batch_track_id[..., None], obj_i_dim, obj_i_label], axis=-1)

                # obj_i = np.array(obj_i)
                # rm_ls_0 = [0, 1, 2,]
                # rm_ls_1 = [0, 1, 2]
                # rm_ls_2 = [0, 1, 2, 3, 5]
                # rm_ls = [rm_ls_0, rm_ls_1, rm_ls_2]
                # for k in rm_ls[i]:
                #     obj_i[k] = np.ones([9]) * -1

                rgb, disp, acc, _ = render(
                    H, W, focal, chunk=chunk, c2w=c2w[:3, :4], obj=obj_i, **render_kwargs)
                rgbs.append(rgb.numpy())
                disps.append(disp.numpy())

                if j == 0:
                    print(rgb.shape, disp.shape)

                if gt_imgs is not None and render_factor == 0:
                    p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                    print(p)

                if savedir is not None:
                    rgb8 = to8b(rgbs[-1])
                    filename = os.path.join(savedir, '{:03d}.png'.format(j))
                    imageio.imwrite(filename, rgb8)
                    if render_manipulation is not None:
                        if 'handcraft' in render_manipulation:
                            filename = os.path.join(savedir, '{:03d}.txt'.format(j))
                            np.savetxt(filename, np.array(obj_i), fmt='%.18e %.18e %.18e %.18e %.1e %.18e %.18e %.18e %.1e')


                print(j, time.time() - t)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps