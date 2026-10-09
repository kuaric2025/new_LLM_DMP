from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class DummyLowdimRunner(BaseLowdimRunner):
    def __init__(self, **kwargs):
        super().__init__(output_dir=kwargs.get("output_dir"))
        self.kwargs = kwargs

    def run(self, policy):
        _ = policy
        return {
            "test/mean_score": 0.0,
            "test_mean_score": 0.0
        }
