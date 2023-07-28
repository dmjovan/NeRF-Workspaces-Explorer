import torch

torch.autograd.set_detect_anomaly(True)


class Embedding:
    """
    Embedding class embeds x (input) to (x, sin(2^k x), cos(2^k x), ...)
    """

    def __init__(self, num_freqs: int, scalar_factor: float = 1.0) -> None:

        self._input_dims = 3
        self._num_freqs = num_freqs
        self._max_freq_log2 = num_freqs - 1
        self._periodic_functions = [torch.sin, torch.cos]
        self._scalar_factor = scalar_factor

        self._embedding_functions = []
        self._output_dim = 0

        self._create_embedding_functions()

    def _create_embedding_functions(self):

        # Original raw input "x" is also included in the output
        original_input_fcn = lambda x: x
        self._embedding_functions.append(original_input_fcn)
        self._output_dim += self._input_dims

        # Log scale for sampling
        freq_bands = 2. ** torch.linspace(0., self._max_freq_log2, steps=self._num_freqs)

        for freq in freq_bands:
            for p_fn in self._periodic_functions:
                periodic_fcn = lambda x, periodic_fn=p_fn, freq=freq: periodic_fn(x * freq)
                self._embedding_functions.append(periodic_fcn)
                self._output_dim += self._input_dims

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run embedding, by passing inputs through all embedding functions and concatenating them.
        """
        return torch.cat([fcn(inputs / self._scalar_factor) for fcn in self._embedding_functions], -1)
