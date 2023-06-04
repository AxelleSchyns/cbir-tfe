from db import Database
from argparse import ArgumentParser
import models
from PIL import Image
import os
import builder
import utils
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import DeiTFeatureExtractor, AutoImageProcessor, ConvNextImageProcessor
class ImageRetriever:
    def __init__(self, db_name, model):
        self.db = Database(db_name, model, True)

    def retrieve(self, image, extractor, nrt_neigh=10):
        return self.db.search(image,extractor, nrt_neigh)

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
        '--db_name',
        help='name of the database',
        default='db'
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
        '--nrt_neigh',
        default=10,
        type=int
    )
    parser.add_argument(
        '--results_dir'
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
    
    
    model = models.Model(model=args.extractor, num_features=args.num_features, name=args.weights,
                           use_dr=args.dr_model, device=device)

    if model.model_name == 'deit':
            feat_extract = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
		                                                         size=224, do_center_crop=False,
		                                                         image_mean=[0.485, 0.456, 0.406],
		                                                         image_std=[0.229, 0.224, 0.225]) 
            transformer = True
    elif model.model_name == 'cvt':
            feat_extract = ConvNextImageProcessor.from_pretrained("microsoft/cvt-21", size=224, do_center_crop=False,
                                                                        image_mean=[0.485, 0.456, 0.406],
                                                                        image_std=[0.229, 0.224, 0.225])
            transformer = True
    elif model.model_name == 'conv':
            feat_extract = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224", size=224, do_center_crop=False,
                                                                        image_mean=[0.485, 0.456, 0.406],
                                                                        image_std=[0.229, 0.224, 0.225])
            transformer = True
    else:
            feat_extract = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
    retriever = ImageRetriever(args.db_name, model)

    ret_values = retriever.retrieve(feat_extract(Image.open(args.path).convert('RGB')), args.extractor, args.nrt_neigh)
    dir = args.results_dir
    names = ret_values[0]
    dist = ret_values[1]
    names_only = []
    class_names = []
    classement = 0 
    Image.open(args.path).convert('RGB').save(os.path.join(dir, "query_image.png"))
    for n in names:
        classement += 1
        class_names.append(utils.get_class(n))
        names_only.append(n[n.rfind('/')+1:])
        img = Image.open(n)
        img.save(os.path.join(dir,str(classement) + "_" + utils.get_class(n)+'_'+n[n.rfind('/')+1:] ))
        #img.save(os.path.join(dir,n[n.rfind('/')+1:]))
    
    # Subplot of the image and the nearest images
    plt.figure(figsize=(7,3))
    plt.subplot(2,6,1)
    plt.imshow(Image.open(args.path).convert('RGB'))
    plt.title("Query image", fontsize=8)
    plt.axis('off')
    for i in range(2,12):
        class_name = utils.get_new_name(class_names[i-2])
        plt.subplot(2,6,i)
        plt.imshow(Image.open(names[i-2]).convert('RGB'))
        # Write the distance and rank right below each of the image
        #plt.text(2, 45, str(dist[i-2])+ str(i-1), fontsize=8)
        # Change title font size
        plt.title(class_name,fontsize=8)

        plt.axis('off')
    plt.savefig(os.path.join(dir, "nearest_images.png"))
    
    plt.subplot(1,2,1)
    plt.imshow(Image.open(args.path).convert('RGB'))
    plt.title("Query image", fontsize=8)
    plt.subplot(1,2,2)
    plt.imshow(Image.open(names[0]).convert('RGB'))
    plt.title("Nearest image", fontsize=8)
    plt.savefig(os.path.join(dir, "nearest_image.png"))
    print("The names of the nearest images are: "+str(class_names))
    
