

from argparse import ArgumentParser
import os
from matplotlib import pyplot as plt
from PIL import Image
import models
from torchvision import transforms



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--path',
        help='path to the image',
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor that is used',
        default='densenet'
    )

    parser.add_argument(
        '--num_features',
        help='number of features to extract',
        default=128,
        type=int
    )

    parser.add_argument(
        '--weights',
        help='file that contains the weights of the network',
        default='weights'
    )

    parser.add_argument(
        '--dr_model',
        action="store_true"
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )
    parser.add_argument(
        '--namefig',
        default='rec'
    )
    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if args.path is None:
        print("You need to specify a path!")
        exit(-1)

    if not os.path.isfile(args.path):
        print('Path mentionned is not a file')
        exit(-1)
    args = parser.parse_args()

    img = Image.open(args.path).convert('RGB')
    transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(.8,1)), # Create a patch (random method)
                    transforms.ToTensor(),
                ]
            )
    img = transform(img).to(device)
    model = models.Model(model=args.extractor, num_features=args.num_features, name=args.weights,
                           use_dr=args.dr_model, device=device)
    extractor = args.extractor
    if extractor == 'vgg11' or extractor == 'resnet18' or extractor == "vgg16" or extractor == "resnet50":
            input = img.view(-1, 3, 224, 224)
            output = model.model(input)
            input = transforms.ToPILImage()(input.squeeze().cpu())
            output = transforms.ToPILImage()(output.squeeze().cpu())
            plt.subplot(1,2,1, xticks=[], yticks=[])
            plt.imshow(input)
            plt.subplot(1,2,2, xticks=[], yticks=[])
            plt.imshow(output)
            plt.savefig(args.namefig +".png")
    elif extractor == 'vae':
        mu, logvar = model.model.encode(img)
        
        out = model.model.reparameterize(mu, logvar)
        dec = model.model.decode(out).cpu()
        #print(out.shape)
        # For visualisation of the reconstruction
        dec = dec.view(3, 224, 224).detach()
        plt.subplot(1, 2, 1)
        plt.imshow(  img.cpu().permute(1, 2, 0)  )
        plt.subplot(1, 2, 2)
        plt.imshow(  dec.permute(1, 2, 0)  )
        plt.savefig(args.namefig +".png")
    elif extractor == 'auto':
        reconstructed, flattened, latent, weights = model.model(img)
        out1 = reconstructed.view(3, 224, 224).cpu().detach()
        plt.subplot(1,2,1)
        # display image and its reconstruction
        plt.imshow(  img.cpu().permute(1, 2, 0)  )
        plt.subplot(1,2,2)
        plt.imshow( out1.permute(1, 2, 0))
        plt.savefig(args.namefig +".png")