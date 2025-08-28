from transformers import PretrainedConfig


class AudioProtoNetConfig(PretrainedConfig):
    _auto_class = "AutoConfig"
    model_type = "AudioProtoNet"

    def __init__(
            self,
            prototypes_per_class: int = 1,
            channels: int = 1024,
            height: int = 1,
            width: int = 1,
            num_classes: int = 9736,
            topk_k: int = 1,
            margin: float = None,
            add_on_layers_type: str = "upsample",
            incorrect_class_connection: float = None,
            correct_class_connection: float = 1.0,
            bias_last_layer: float = -2.0,
            non_negative_last_layer: bool = True,
            embedded_spectrogram_height: int = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.prototypes_per_class = prototypes_per_class
        #self.num_prototypes_after_pruning = None weg
        self.channels = channels
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.topk_k = topk_k
        self.margin = margin
        # self.relu_on_cos = True
        self.add_on_layers_type = add_on_layers_type
        self.incorrect_class_connection = incorrect_class_connection
        self.correct_class_connection = correct_class_connection
        #self.input_vector_length = 64
        # self.n_eps_channels = 2
        # self.epsilon_val = 1e-4
        self.bias_last_layer = bias_last_layer
        self.non_negative_last_layer = non_negative_last_layer
        self.embedded_spectrogram_height = embedded_spectrogram_height

        if self.bias_last_layer:
            self.use_bias_last_layer = True
        else:
            self.use_bias_last_layer = False

        self.prototype_class_identity = None