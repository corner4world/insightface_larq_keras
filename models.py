import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import larq as lq
import larq_zoo

def print_buildin_models():
    print(
        """
    >>>> buildin_models
    MXNet version resnet: mobilenet_m1, r34, r50, r100, r101,
    Keras application: mobilenet, mobilenetv2, resnet50, resnet50v2, resnet101, resnet101v2, resnet152, resnet152v2
    EfficientNet: efficientnetb[0-7], efficientnetl2,
    Custom 1: ghostnet, mobilefacenet, mobilenetv3_small, mobilenetv3_large, se_mobilefacenet
    Custom 2: botnet50, botnet101, botnet152, resnest50, resnest101, se_resnext
    Or other names from keras.applications like DenseNet121 / InceptionV3 / NASNetMobile / VGG19.
    """,
        end="",
    )


def __init_model_from_name__(name, input_shape=(112, 112, 3), weights="imagenet", **kwargs):
    name_lower = name.lower()
    """ Basic model """
    if name_lower == "mobilenet":
        xx = keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights=weights, **kwargs)
    elif name_lower == "mobilenet_m1":
        from backbones import mobilenet
        xx = mobilenet.MobileNet(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name_lower == "binarydensenet":
        xx = larq_zoo.literature.BinaryDenseNet45(input_shape=input_shape,include_top=False,weights=None, **kwargs)
    elif name_lower == "mobilenetv2":
        xx = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights, **kwargs)
    elif name_lower == "r34" or name_lower == "r50" or name_lower == "r100" or name_lower == "r101":
        from backbones import resnet  # MXNet insightface version resnet

        model_name = "ResNet" + name_lower[1:]
        model_class = getattr(resnet, model_name)
        xx = model_class(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name_lower.startswith("resnet"):  # keras.applications.ResNetxxx
        if name_lower.endswith("v2"):
            model_name = "ResNet" + name_lower[len("resnet") : -2] + "V2"
        else:
            model_name = "ResNet" + name_lower[len("resnet") :]
        model_class = getattr(keras.applications, model_name)
        xx = model_class(weights=weights, include_top=False, input_shape=input_shape, **kwargs)
    elif name_lower.startswith("efficientnet"):
        # import tensorflow.keras.applications.efficientnet as efficientnet
        from backbones import efficientnet

        model_name = "EfficientNet" + name_lower[-2:].upper()
        model_class = getattr(efficientnet, model_name)
        xx = model_class(weights=weights, include_top=False, input_shape=input_shape, **kwargs)  # or weights='imagenet'
    elif name_lower.startswith("se_resnext"):
        from keras_squeeze_excite_network import se_resnext

        if name_lower.endswith("101"):  # se_resnext101
            depth = [3, 4, 23, 3]
        else:  # se_resnext50
            depth = [3, 4, 6, 3]
        xx = se_resnext.SEResNextImageNet(weights=weights, input_shape=input_shape, include_top=False, depth=depth)
    elif name_lower.startswith("resnest"):
        from backbones import resnest

        if name_lower == "resnest50":
            xx = resnest.ResNest50(input_shape=input_shape)
        else:
            xx = resnest.ResNest101(input_shape=input_shape)
    elif name_lower.startswith("mobilenetv3"):
        from backbones import mobilenet_v3

        # from backbones import mobilenetv3 as mobilenet_v3
        model_class = mobilenet_v3.MobileNetV3Small if "small" in name_lower else mobilenet_v3.MobileNetV3Large
        # from tensorflow.keras.layers.experimental.preprocessing import Rescaling
        # model_class = keras.applications.MobileNetV3Small if "small" in name_lower else keras.applications.MobileNetV3Large
        xx = model_class(input_shape=input_shape, include_top=False, weights=weights)
        # xx = keras.models.clone_model(xx, clone_function=lambda layer: Rescaling(1.) if isinstance(layer, Rescaling) else layer)
    elif "mobilefacenet" in name_lower or "mobile_facenet" in name_lower:
        from backbones import mobile_facenet

        use_se = True if "se" in name_lower else False
        xx = mobile_facenet.mobile_facenet(input_shape=input_shape, include_top=False, name=name, use_se=use_se)
    elif name_lower == "ghostnet":
        from backbones import ghost_model

        xx = ghost_model.GhostNet(input_shape=input_shape, include_top=False, width=1.3, **kwargs)
    elif name_lower.startswith("botnet"):
        from backbones import botnet

        model_name = "BotNet" + name_lower[len("botnet") :]
        model_class = getattr(botnet, model_name)
        xx = model_class(include_top=False, input_shape=input_shape, strides=1, **kwargs)
    elif hasattr(keras.applications, name):
        model_class = getattr(keras.applications, name)
        xx = model_class(weights=weights, include_top=False, input_shape=input_shape, **kwargs)
    else:
        return None
    xx.trainable = True
    return xx


# MXNET: bn_momentum=0.9, bn_epsilon=2e-5, TF default: bn_momentum=0.99, bn_epsilon=0.001, PyTorch default: momentum=0.1, eps=1e-05
# MXNET: use_bias=True, scale=False, cavaface.pytorch: use_bias=False, scale=True
def buildin_models(
    stem_model,
    dropout=1,
    emb_shape=512,
    input_shape=(112, 112, 3),
    output_layer="GDC",
    bn_momentum=0.99,
    bn_epsilon=0.001,
    add_pointwise_conv=False,
    use_bias=False,
    scale=True,
    weights="imagenet",
    sam_rho=0,
    **kwargs
):
    if isinstance(stem_model, str):
        xx = __init_model_from_name__(stem_model, input_shape, weights, **kwargs)
        name = stem_model
    else:
        name = stem_model.name
        xx = stem_model

    if bn_momentum != 0.99 or bn_epsilon != 0.001:
        print(">>>> Change BatchNormalization momentum and epsilon default value.")
        for ii in xx.layers:
            if isinstance(ii, keras.layers.BatchNormalization):
                ii.momentum, ii.epsilon = bn_momentum, bn_epsilon
        xx = keras.models.clone_model(xx)

    inputs = xx.inputs[0]
    nn = xx.outputs[0]

    if add_pointwise_conv:  # Model using `pointwise_conv + GDC` / `pointwise_conv + E` is smaller than `E`
        nn = keras.layers.Conv2D(512, 1, use_bias=False, padding="same")(nn)
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
        nn = keras.layers.PReLU(shared_axes=[1, 2])(nn)

    if output_layer == "E":
        """ Fully Connected """
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Flatten()(nn)
        nn = keras.layers.Dense(emb_shape, activation=None, use_bias=use_bias, kernel_initializer="glorot_normal")(nn)
    elif output_layer == "GAP":
        """ GlobalAveragePooling2D """
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
        nn = keras.layers.GlobalAveragePooling2D()(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Dense(emb_shape, activation=None, use_bias=use_bias, kernel_initializer="glorot_normal")(nn)
    else:
        """ GDC """
        nn = keras.layers.DepthwiseConv2D(int(nn.shape[1]), depth_multiplier=1, use_bias=False)(nn)
        # nn = keras.layers.Conv2D(512, int(nn.shape[1]), use_bias=False, padding="valid", groups=512)(nn)
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Conv2D(emb_shape, 1, use_bias=use_bias, activation=None, kernel_initializer="glorot_normal")(nn)
        nn = keras.layers.Flatten()(nn)
        # nn = keras.layers.Dense(emb_shape, activation=None, use_bias=True, kernel_initializer="glorot_normal")(nn)

    # `fix_gamma=True` in MXNet means `scale=False` in Keras
    embedding = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, scale=scale)(nn)
    embedding_fp32 = keras.layers.Activation("linear", dtype="float32", name="embedding")(embedding)
    if sam_rho == 0:
        basic_model = keras.models.Model(inputs, embedding_fp32, name=xx.name)
    else:
        basic_model = SAMModel(inputs, embedding_fp32, rho=sam_rho, name=xx.name)
    return basic_model


class NormDense(keras.layers.Layer):
    def __init__(self, units=1000, kernel_regularizer=None, loss_top_k=1, **kwargs):
        super(NormDense, self).__init__(**kwargs)
        self.init = keras.initializers.glorot_normal()
        self.units, self.loss_top_k = units, loss_top_k
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.supports_masking = True

    def build(self, input_shape):
        self.w = self.add_weight(
            name="norm_dense_w",
            shape=(input_shape[-1], self.units * self.loss_top_k),
            initializer=self.init,
            trainable=True,
            regularizer=self.kernel_regularizer,
        )
        super(NormDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        norm_w = K.l2_normalize(self.w, axis=0)
        inputs = K.l2_normalize(inputs, axis=1)
        output = K.dot(inputs, norm_w)
        if self.loss_top_k > 1:
            output = K.reshape(output, (-1, self.units, self.loss_top_k))
            output = K.max(output, axis=2)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(NormDense, self).get_config()
        config.update(
            {
                "units": self.units,
                "loss_top_k": self.loss_top_k,
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def add_l2_regularizer_2_model(model, weight_decay, custom_objects={}, apply_to_batch_normal=True):
    # https://github.com/keras-team/keras/issues/2717#issuecomment-456254176
    if 0:
        regularizers_type = {}
        for layer in model.layers:
            rrs = [kk for kk in layer.__dict__.keys() if "regularizer" in kk and not kk.startswith("_")]
            if len(rrs) != 0:
                # print(layer.name, layer.__class__.__name__, rrs)
                if layer.__class__.__name__ not in regularizers_type:
                    regularizers_type[layer.__class__.__name__] = rrs
        print(regularizers_type)

    for layer in model.layers:
        attrs = []
        if isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.Conv2D):
            # print(">>>> Dense or Conv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["kernel_regularizer"]
            if layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.DepthwiseConv2D):
            # print(">>>> DepthwiseConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["depthwise_regularizer"]
            if layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.SeparableConv2D):
            # print(">>>> SeparableConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["pointwise_regularizer", "depthwise_regularizer"]
            if layer.use_bias:
                attrs.append("bias_regularizer")
        elif apply_to_batch_normal and isinstance(layer, keras.layers.BatchNormalization):
            # print(">>>> BatchNormalization", layer.name, "scale:", layer.scale, ", center:", layer.center)
            if layer.center:
                attrs.append("beta_regularizer")
            if layer.scale:
                attrs.append("gamma_regularizer")
        elif apply_to_batch_normal and isinstance(layer, keras.layers.PReLU):
            # print(">>>> PReLU", layer.name)
            attrs = ["alpha_regularizer"]

        for attr in attrs:
            if hasattr(layer, attr) and layer.trainable:
                setattr(layer, attr, keras.regularizers.L2(weight_decay / 2))

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    # temp_weight_file = "tmp_weights.h5"
    # model.save_weights(temp_weight_file)
    # out_model = keras.models.model_from_json(model.to_json(), custom_objects=custom_objects)
    # out_model.load_weights(temp_weight_file, by_name=True)
    # os.remove(temp_weight_file)
    # return out_model
    return keras.models.clone_model(model)


def replace_ReLU_with_PReLU(model, target_activation="PReLU", **kwargs):
    from tensorflow.keras.layers import ReLU, PReLU, Activation

    def convert_ReLU(layer):
        # print(layer.name)
        if isinstance(layer, ReLU) or (isinstance(layer, Activation) and layer.activation == keras.activations.relu):
            if target_activation == "PReLU":
                layer_name = layer.name.replace("_relu", "_prelu")
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                # Default initial value in mxnet and pytorch is 0.25
                return PReLU(shared_axes=[1, 2], alpha_initializer=tf.initializers.Constant(0.25), name=layer_name, **kwargs)
            if target_activation == "swish":
                layer_name = layer.name.replace("_relu", "_swish")
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                return Activation(activation="swish", name=layer_name, **kwargs)
            elif target_activation == "aconC":
                layer_name = layer.name.replace("_relu", "_aconc")
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                return AconC()
            else:
                print(">>>> Convert ReLU:", layer.name)
                return target_activation(**kwargs)
        return layer

    return keras.models.clone_model(model, clone_function=convert_ReLU)


def aconC(inputs, p1=1, p2=0, beta=1):
    """
    - [Github nmaac/acon](https://github.com/nmaac/acon/blob/main/acon.py)
    - [Activate or Not: Learning Customized Activation, CVPR 2021](https://arxiv.org/pdf/2009.04759.pdf)
    """
    p1 = keras.layers.DepthwiseConv2D(1, use_bias=False, depthwise_initializer=tf.initializers.Constant(p1))(inputs)
    p2 = keras.layers.DepthwiseConv2D(1, use_bias=False, depthwise_initializer=tf.initializers.Constant(p2))(inputs)
    beta = keras.layers.DepthwiseConv2D(1, use_bias=False, depthwise_initializer=tf.initializers.Constant(beta))(p1)

    return p1 * tf.nn.sigmoid(beta) + p2

class AconC(keras.layers.Layer):
    def __init__(self, p1=1, p2=0, beta=1, **kwargs):
        super(AconC, self).__init__(**kwargs)
        self.p1 = keras.layers.DepthwiseConv2D(1, use_bias=False, depthwise_initializer=tf.initializers.Constant(p1))
        self.p2 = keras.layers.DepthwiseConv2D(1, use_bias=False, depthwise_initializer=tf.initializers.Constant(p2))
        self.beta = keras.layers.DepthwiseConv2D(1, use_bias=False, depthwise_initializer=tf.initializers.Constant(beta))

    def build(self, input_shape):
        self.p1.build(input_shape)
        self.p2.build(input_shape)
        self.beta.build(input_shape)
        super(AconC, self).build(input_shape)

    def call(self, inputs, **kwargs):
        p1 = self.p1(inputs)
        p2 = self.p2(inputs)
        beta = self.beta(p1)
        return p1 * tf.nn.sigmoid(beta) + p2

    def compute_output_shape(self, input_shape):
        return input_shape


class SAMModel(tf.keras.models.Model):
    """
    Arxiv article: [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/pdf/2010.01412.pdf)
    Implementation by: [Keras SAM (Sharpness-Aware Minimization)](https://qiita.com/T-STAR/items/8c3afe3a116a8fc08429)
    """

    def __init__(self, *args, rho=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = tf.constant(rho, dtype=tf.float32)

    def train_step(self, data):
        x, y = data

        # 1st step
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        norm = tf.linalg.global_norm(gradients)
        scale = self.rho / (norm + 1e-12)
        e_w_list = []
        for v, grad in zip(trainable_vars, gradients):
            e_w = grad * scale
            v.assign_add(e_w)
            e_w_list.append(e_w)

        # 2nd step
        with tf.GradientTape() as tape:
            y_pred_adv = self(x, training=True)
            loss_adv = self.compiled_loss(y, y_pred_adv, regularization_losses=self.losses)
        gradients_adv = tape.gradient(loss_adv, trainable_vars)
        for v, e_w in zip(trainable_vars, e_w_list):
            v.assign_sub(e_w)

        # optimize
        self.optimizer.apply_gradients(zip(gradients_adv, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


def replace_add_with_stochastic_depth(model, survivals=(1, 0.8)):
    """
    - [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)
    - [tfa.layers.StochasticDepth](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/StochasticDepth)
    """
    from tensorflow_addons.layers import StochasticDepth

    add_layers = [ii.name for ii in model.layers if isinstance(ii, keras.layers.Add)]
    total_adds = len(add_layers)
    if isinstance(survivals, float):
        survivals = [survivals] * total_adds
    elif isinstance(survivals, (list, tuple)) and len(survivals) == 2:
        start, end = survivals
        survivals = [start + (end - start) * ii / (total_adds - 1) for ii in range(total_adds)]
    survivals_dict = dict(zip(add_layers, survivals))

    def __replace_add_with_stochastic_depth__(layer):
        if isinstance(layer, keras.layers.Add):
            layer_name = layer.name
            new_layer_name = layer_name.replace("_add", "_stochastic_depth")
            survival_probability = survivals_dict[layer_name]
            if survival_probability < 1:
                print("Converting:", layer_name, "-->", new_layer_name, ", survival_probability:", survival_probability)
                return StochasticDepth(survival_probability, name=new_layer_name)
            else:
                return layer
        return layer

    return keras.models.clone_model(model, clone_function=__replace_add_with_stochastic_depth__)


def replace_stochastic_depth_with_add(model, drop_survival=False):
    from tensorflow_addons.layers import StochasticDepth

    def __replace_stochastic_depth_with_add__(layer):
        if isinstance(layer, StochasticDepth):
            layer_name = layer.name
            new_layer_name = layer_name.replace("_stochastic_depth", "_lambda")
            survival = layer.survival_probability
            print("Converting:", layer_name, "-->", new_layer_name, ", survival_probability:", survival)
            if drop_survival or not survival < 1:
                return keras.layers.Add(name=new_layer_name)
            else:
                return keras.layers.Lambda(lambda xx: xx[0] + xx[1] * survival, name=new_layer_name)
        return layer

    return keras.models.clone_model(model, clone_function=__replace_stochastic_depth_with_add__)
