from catalyst.core import MetricCallback
from utils import jaccard_func


class JaccardCallback(MetricCallback):

    def __init__(
        self,
        input_key = ['start_positions', 'end_positions', 'orig_tweet', 'orig_selected', 'offsets'],
        output_key = ['start_logits', 'end_logits'],
        prefix = "jaccard",
        activation = None
    ):
        super().__init__(
            prefix=prefix,
            metric_fn=jaccard_func,
            input_key=input_key,
            output_key=output_key
            #activation=activation
        )
