from argparse import ArgumentParser
from keras.applications.vgg16 import VGG16
import keras.backend as K
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time

from preprocessing import preprocess, postprocess_and_save, load_image
from loss import StyleTransferLoss
from evaluator import Evaluator


def run(style_image=None,
        content_image=None,
        save_path=None,
        height=512,
        width=512,
        feature_layers=[],
        iterations=10,
        content_weight=0.05,
        style_weight=5.0,
        total_variation_weight=1.0,
        abstract_factor=4):
    """
    Perform style transfer using the given images.
    
    :param style_image:                     Path to the 'style' image (str)
    :param content_image:                   Path to the 'content' image (str)
    :param height:                          Height of output image (int)
    :param width:                           Width of output image (int)
    :param feature_layers:                  List of layers from which to extract style (list)
    :param iterations:                      Number of training iterations (int)
    :param content_weight:                  Weighting of the content loss (float)
    :param style_weight:                    Weighting of the style loss (float)
    :param total_variation_weight:          Weighting of the overall variation loss (float)
    :param abstract_factor:                 How abstract should the image be
    :return: 
    """
    assert save_path != None, 'You forgot to pass a save path.'
    assert style_image != None, 'You forgot to pass a style image.'
    assert content_image != None, 'You forgot to pass a content image.'
    assert height == width, 'Height and width must be equal'

    # load and pre-process images
    content_image = preprocess(image=content_image, height=height, width=width)
    style_image = preprocess(image=style_image, height=height, width=width)

    # load Keras variables
    combination_image = K.placeholder((1, height, width, 3))
    input_tensor = K.concatenate([content_image, style_image, combination_image], axis=0)
    model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if len(feature_layers) == 0:
        feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2', 'block5_conv2']

    # prepare loss function
    loss_function = StyleTransferLoss(model=model,
                                      height=height,
                                      width=width,
                                      combination_image=combination_image,
                                      feature_layers=feature_layers,
                                      content_weight=content_weight,
                                      style_weight=style_weight,
                                      total_variation_weight=total_variation_weight)

    # define gradient and outputs
    loss = loss_function.compute_loss(abstract_factor=abstract_factor)
    grads = K.gradients(loss, loss_function.combination_image)
    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)
    f_outputs = K.function([loss_function.combination_image], outputs)

    # prepare for transfer
    evaluator = Evaluator(height, width, f_outputs)
    x = np.random.uniform(0, 255, (1, height, width, 3))  # rgb-noise image initializer

    print('Starting training...')
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        end_time = time.time()
        print('Iteration {} completed in {} seconds.'.format(i, end_time - start_time))

    # save
    _ = postprocess_and_save(image=x, width=width, height=height, save_path=save_path)


def parse_args():
    ap = ArgumentParser(description='Perform style transfer on any two images')
    ap.add_argument('--style', type=str, help='Path to style image')
    ap.add_argument('--content', type=str, help='Path to content image')
    ap.add_argument('--save_path', type=str, default='$HOME', help='Where do you want to save the output?')
    ap.add_argument('--size', type=int, default=256, help='Image height and width (output is always square)')
    ap.add_argument('--iter', type=int, default=10, help='Number of iterations')
    ap.add_argument('--alpha', type=float, default=0.01, help='The ratio of content to style, default is 1e-2.'
                                                              'Increase for more content, decrease for more style.')
    ap.add_argument('--afactor', type=int, default=4,
                    help='Abstract factor. How abstract do you want your image to be?')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    style_image = load_image(args.style)
    content_image = load_image(args.content)
    run(style_image=style_image,
        content_image=content_image,
        save_path=args.save_path,
        height=args.size,
        width=args.size,
        iterations=args.iter,
        content_weight=args.alpha,
        style_weight=1.0,
        total_variation_weight=0.2,
        abstract_factor=args.afactor)
