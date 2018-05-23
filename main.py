# import json
from trainer import runner
from torchvision.models import alexnet, resnet18, resnet34, resnet50, resnet101, resnet152, vgg16_bn, vgg19_bn
import config
# config

# Network_list
network_list = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
}


def main():
    """
    # read json
    params = {}
    with open('training_params.json', 'r') as f:
        training_params = json.load(f)
    """

    print('training_params: ', config.training_params)

    model_name = '{}_{}_{}'. \
        format(config.training_params['model_title'], config.training_params['net_name'], config.training_params['model_info'])

    kwargs = {
        'nb_epochs': config.training_params['nb_epochs'],
        'batch_size': config.training_params['batch_size'],
        'learning_rate': config.training_params['learning_rate'],
        'net': network_list[config.training_params['net_name']],
        'model_dir': config.training_params['model_dir'],
        'model_name': model_name,
    }
    runner(**kwargs)


if __name__ == '__main__':
    main()
