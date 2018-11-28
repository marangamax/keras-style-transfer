import keras.backend as K
from keras.engine.training import Model
from tensorflow.python.framework.ops import Tensor


class StyleTransferLoss(object):
    def __init__(self,
                 model=None,
                 height=512,
                 width=512,
                 combination_image=None,
                 feature_layers=['block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2', 'block5_conv2'],
                 content_weight=0.005,
                 style_weight=5.0,
                 total_variation_weight=1.0):
        """
        Calculate a custom style transfer loss.
        
        :param model:                       Keras model (keras.engine.training.Model)
        :param height:                      Image height (int)
        :param width:                       Image width (int)
        :param feature_layers:              From which layers should we extract the style? (list)
        :param content_weight:              How much weight should the content loss have? (float)
        :param style_weight:                How much weight should the style loss have? (float)
        :param total_variation_weight:      How much weight should the variation loss have? (float)
        """
        assert type(model) == Model, 'Model type is invalid. You must pass a valid Keras model.'
        self.layers = dict([(layer.name, layer.output) for layer in model.layers])

        assert type(feature_layers) == list, 'Feature layers must be a list of layers present in the network.'
        for layer in feature_layers:
            assert layer in self.layers, 'Feature layers must be a list of layers present in the network.'

        assert type(combination_image) == Tensor, 'Combination image type is invalid, please pass a Tensor placeholder.'
        self.combination_image = combination_image

        self.height = height
        self.width = width
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight
        self.feature_layers = feature_layers

    def compute_loss(self, abstract_factor):
        """
        Compute overall loss by appending the loss value for the content, style, and variation.
        :return: loss (keras.backend.variable)
        """
        loss = K.variable(0.)
        loss = loss + self.content_loss(abstract_factor)
        for layer_name in self.feature_layers:
            loss = loss + self.style_loss(layer_name)
        loss = loss + self.total_variation_weight * total_variation_loss(self.combination_image, self.height, self.width)
        return loss

    def content_loss(self, abstract_factor):
        layer_features = self.layers['block{}_conv2'.format(abstract_factor)]
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        return self.content_weight * content_loss(content_image_features, combination_features)

    def style_loss(self, layer_name):
        layer_features = self.layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        return style_loss(style_features, combination_features, self.height, self.width)


def content_loss(content, combination):
    """
    Content loss function. Used to extrapolate contours from the content image.
    """
    return K.sum(K.square(content - combination))


def gram_matrix(x):
    """
    Hermitian matrix of inner products. Used to compute linear dependence of features.
    """
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination, height, width, channels=3):
    """
    Style loss function.
    """
    S = gram_matrix(style)
    C = gram_matrix(combination)
    size = height * width
    st = K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
    return st


def total_variation_loss(x, height, width):
    """
    Enforces smoothness across the final image.
    """
    a = K.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
    b = K.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
