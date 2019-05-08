import torch
import pathlib

from nets import create_densenet, create_squeezenet, create_shufflenet

from demo_utils import demo

def load_model(model_name, nInputChannels):
    if model_name == 'densenet':
        return create_densenet(nInputChannels)
    elif model_name == 'squeezenet':
        return create_squeezenet(nInputChannels)
    elif model_name == 'shufflenet':
        return create_shufflenet(nInputChannels)

def main():
    model_name = 'shufflenet'
    gpu_id = 0
    device = torch.device("cpu")
    #device = torch.device("cuda:" + str(gpu_id))

    #  Create the network and load the weights
    net = load_model(model_name, 4)
    project_dir = pathlib.Path(__file__).resolve().parent
    weights_path = project_dir / 'models' / '{}.pth'.format(model_name)
    net_parameters = torch.load(str(weights_path), map_location=lambda s, l: s)
    net.load_state_dict(net_parameters)
    net.eval()
    net.to(device)
    with torch.no_grad():
        demo(net)


if __name__ == '__main__':
    main()
