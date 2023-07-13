import torch

def batchify_rays(render_fn, rays_flat, chunk=1024 * 32):

    """ Render rays in smaller minibatches to avoid OOM """

    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_fn(rays_flat[i:i + chunk])
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def batchify(fn, chunk):

    """ Constructs a version of 'fn' that applies to smaller batches """

    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret