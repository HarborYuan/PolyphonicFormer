def depth_act(
        depth_out,
        mode='monodepth',
        min_depth=0.01,
        max_depth=80.
):
    if mode == 'monodepth':
        disp = depth_out.sigmoid()
        min_disp = 1. / max_depth
        max_disp = 1. / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth
    elif mode == 'sigmoid':
        disp = depth_out.sigmoid()
        depth = disp * (max_depth - min_depth) + min_depth
        return depth
    else:
        raise NotImplementedError
