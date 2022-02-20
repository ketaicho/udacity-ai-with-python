# PROGRAMMER: Tshepo Molebiemang
# DATE: 10 November 2021
import torch
from torch import optim
import numpy as np
import json
from PIL import Image
import argparse
from train import classifier


def get_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help="path to checkpoint")
    parser.add_argument('--image', type=str, default='./flowers/test/8/image_03364.jpg', help="test image path")
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help="Category mappings")
    parser.add_argument('--gpu', type=str, default='True', help="'True' to use gpu") #bool type has unexpected behaviour
    parser.add_argument('--top_k', type=int, default=5, help="No. of top probabilities")

    return parser.parse_args()


def load_model(path):

    if torch.cuda.is_available():
        device = lambda storage, loc: storage.cuda()
    else:
        device = 'cpu'

    checkpoint = torch.load(path, device)

    arch = checkpoint['arch']
    out_features = len(checkpoint['class_to_idx'])
    hidden_units = checkpoint['hidden_units']

    model = classifier(arch, out_features, hidden_units)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    with Image.open(image) as im:
        width = im.size[0]
        height = im.size[1]

        # resize with shorter side being 256px
        im.thumbnail((500, 255) if (width > height) else (255, 500))

        # crop image
        left = (im.width - 224) / 2
        top = (im.height - 224) / 2
        bottom = top + 224
        right = left + 224
        cropped_im = im.crop((left, top, right, bottom))

        # convert pil image to np array
        np_image = np.array(cropped_im) / 255
        # np_image = np.array(cropped_im)

        # normalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std_dev = np.array([0.229, 0.224, 0.225])
        norm_image = (np_image - mean) / std_dev

        # swap the first and 3rd dimension
        transp_image = np.transpose(norm_image, (2, 0, 1))

        return transp_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    if topk < 1:
        topk = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    tensor_image = torch.FloatTensor([process_image(image_path)])

    tensor_image = tensor_image.to(device)

    result = model(tensor_image).topk(topk)

    probs = torch.exp(result[0].data).cpu().numpy()[0]

    idx = result[1].data.cpu().numpy()[0]

    return probs, idx


def main():

    args = get_input_args()

    if args.gpu == 'True' or args.gpu == 'TRUE' or args.gpu == 'true':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    image_path = args.image
    path_to_chkpt = args.checkpoint
    top_k = args.top_k
    category_file = args.category_names

    # *****************************************
    print("\nInput Variables")
    print(f"image_path: \t{image_path}")
    print(f"path_to_checkpoint: {path_to_chkpt}")
    print(f"device: \t\t{device}")
    print(f"category_names: \t{category_file}")
    print(f"top_k: \t\t{top_k}")
    print("\n__________________________________________\n")
    print(" **** Predicting what the image is... ****")
    print("------------------------------------------\n")
    # *****************************************

    loaded_model = load_model(path_to_chkpt)

    probs, class_idxs = predict(image_path, loaded_model, top_k)

    with open(category_file, 'r') as f:
        category_names = json.load(f)

    idx_to_class = {v: k for k, v in loaded_model.class_to_idx.items()}
    names = list(map(lambda x: category_names[f"{idx_to_class[x]}"], class_idxs))
    class_num = [idx_to_class.get(idx) for idx in class_idxs]

    topk = list(zip(class_idxs, probs))

    print("++ Most Probable Class ++")
    print("{} [{}]: {:.3f}%".format(names[0], class_num[0], topk[0][1] * 100))

    print(f"\n -- Top {top_k} Predictions --")
    for i, item in enumerate(topk):
        print("{} [{}]: {:.3f}%".format(names[i], class_num[i], item[1] * 100))


if __name__ == "__main__":
    main()