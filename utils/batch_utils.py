from typing import Callable, Dict

import torch
from tqdm import tqdm


def batchify_rays(render_fn: Callable, rays_flat: torch.Tensor, chunk: int = 1024 * 32) -> Dict[str, torch.Tensor]:
    """
    Rendering rays in smaller mini-batches (chunks) to avoid out-of-memory error
    """

    all_outputs = {}
    for i in tqdm(range(0, rays_flat.shape[0], chunk)):

        # Rendering one chunk of rays
        chunk_of_rays_output = render_fn(rays_flat[i:i + chunk])

        # Adding network outputs of one chunk of rays to the all_outputs dictionary
        for key in chunk_of_rays_output:
            if key not in all_outputs:
                all_outputs[key] = []
            all_outputs[key].append(chunk_of_rays_output[key])

    all_outputs = {k: torch.cat(all_outputs[k], 0) for k in all_outputs}
    return all_outputs


def batchify(fn: Callable, chunk: int) -> Callable:
    """
    Constructs a version of 'fn' that applies to smaller batches (with size of chunk)
    """

    if chunk is None:
        return fn

    def function_to_run_batch(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return function_to_run_batch
