import faiss
import models
import struct
import torch
import dataset
from PIL import Image
from torchvision import transforms
import redis
import numpy as np
from transformers import DeiTFeatureExtractor, AutoImageProcessor, ConvNextImageProcessor
import time
import json
import os
import argparse
import builder
import pickle 
def encode(model, img):
    with torch.no_grad():
        code = model.module.encoder(img).cpu()

    return code

class Database:
    def __init__(self, filename, model, load=False, transformer=False, device='cpu'):
        self.name = filename # = name of the database 
        self.num_features = model.num_features
        self.model = model
        self.device = device
        self.filename = filename

        res_labeled = faiss.StandardGpuResources() # Allocation of steams and temporary memory 
        res_unlabeled = faiss.StandardGpuResources()

        # A database was previously constructed
        if load == True:
            self.index_labeled = faiss.read_index(filename + '_labeled')
            self.index_unlabeled = faiss.read_index(filename + '_unlabeled')
            self.r = redis.Redis(host='localhost', port='6379', db=0)
        else:
            # No database to load, has to build it 
            self.index_labeled = faiss.IndexFlatL2(self.num_features)
            self.index_labeled = faiss.IndexIDMap(self.index_labeled)
            self.index_unlabeled = faiss.IndexFlatL2(self.num_features)
            self.index_unlabeled = faiss.IndexIDMap(self.index_unlabeled)
            self.r = redis.Redis(host='localhost', port='6379', db=0)
            self.r.flushdb()

            self.r.set('last_id_labeled', 0) # Set a value in redis with key = last_id_labeled
            self.r.set('last_id_unlabeled', 0)

            open(filename + '_labeledvectors', 'w').close()
            open(filename + '_unlabeledvectors', 'w').close()

        if device == 'gpu':
            self.index_labeled = faiss.index_cpu_to_gpu(res_labeled, 0, self.index_labeled)
            self.index_unlabeled = faiss.index_cpu_to_gpu(res_unlabeled, 0, self.index_unlabeled)

        self.transformer = transformer
        if model.model_name == 'deit':
                self.feat_extract = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
		                                                         size=224, do_center_crop=False,
		                                                         image_mean=[0.485, 0.456, 0.406],
		                                                         image_std=[0.229, 0.224, 0.225]) 
        elif model.model_name == 'cvt':
                self.feat_extract = ConvNextImageProcessor.from_pretrained("microsoft/cvt-21", size=224, do_center_crop=False,
		                                                                  image_mean=[0.485, 0.456, 0.406],
		                                                                  image_std=[0.229, 0.224, 0.225])
        elif model.model_name == 'conv':
                self.feat_extract = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224", size=224, do_center_crop=False,
		                                                                  image_mean=[0.485, 0.456, 0.406],
		                                                                  image_std=[0.229, 0.224, 0.225])
        else:
                self.feat_extract = None                                                                 
    # x = vector of images 
    def add(self, x, names, label, generalise):
        if label:
            last_id = int(self.r.get('last_id_labeled').decode('utf-8'))
            # Add x.shape ids and images to the current list of ids of Faiss. 
            # Id is given by the last given id to which 1 is added each time, ending with id = last_id + x.shape[0]
            self.index_labeled.add_with_ids(x, np.arange(last_id, last_id + x.shape[0])) 

            # Open the file of the database corresponding to the labelled case.
            with open(self.filename + '_labeledvectors', 'ab') as file:
                # Encode in the file database the images with their name, and gives them the appropriate id
                # Zip() Creates a list of tuples, with one element of names and one of x (=> allows to iterate on both list at the same time) 
                for n, x_ in zip(names, x):
                    if generalise == 3:
                        n = str(n)
                    self.r.set(str(last_id) + 'labeled', n) # Set the name of the image at key = id
                    self.r.set(n, str(last_id) + 'labeled') # Set the id of the image at key = name 
                    binary = struct.pack("i"+str(self.num_features)+"f",last_id, *x_)
                    file.write(binary)
                    #file.write('\n' + str(last_id) + str(x_)) # Writes in the file the id alongside the image 
                    last_id += 1

            self.r.set('last_id_labeled', last_id) # Update the last id to take into account the added images

        # Same as previous for unlabelled data
        else:
            last_id = int(self.r.get('last_id_unlabeled').decode('utf-8'))
            self.index_unlabeled.add_with_ids(x, np.arange(last_id, last_id + x.shape[0]))

            with open(self.filename + '_unlabeledvectors', 'ab') as file:
                for n, x_ in zip(names, x):
                    if generalise == 3:
                        n = str(n)
                    self.r.set(str(last_id) + 'unlabeled', n)
                    self.r.set(n, str(last_id) + 'unlabeled')
                    binary = struct.pack("i"+str(self.num_features)+"f",last_id, *x_)
                    file.write(binary)
                    #file.write('\n' + str(last_id) + str(x_))
                    last_id += 1

            self.r.set('last_id_unlabeled', last_id)

    @torch.no_grad()
    def add_dataset(self, data_root, extractor, generalise, name_list=[], label=True):
        # Create a dataset from a directory root
        if name_list == []:
            data = dataset.AddDataset(data_root, extractor, self.transformer)
        # create a dataset from a list of image names
        else:
            data = dataset.AddDatasetList(data_root, extractor, name_list, self.transformer)
        loader = torch.utils.data.DataLoader(data, batch_size=128, num_workers=12, pin_memory=True)

        for i, (images, filenames) in enumerate(loader):
            images = images.view(-1, 3, 224, 224).to(device=next(self.model.parameters()).device)
            if extractor == 'vgg11' or extractor == 'resnet18' or extractor == 'vgg16':
                out = encode(self.model, images)
                out = out.reshape([out.shape[0],self.model.num_features])
            
            else:
                # Encode the images using the given model 
                out = self.model(images).cpu()
            if generalise == 3:
                def load_image(image_path):
                    with Image.open(image_path) as image:
                        image = image.convert('RGB')
                        image = image.resize((224,224))
                        image = np.array(image, dtype=np.float32) / 255.0
                        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                    return image.reshape(-1)
                kmeans = pickle.load(open("kmeans.pkl","rb"))
                batch_data = np.array([load_image(path) for path in filenames])
                filenames = kmeans.predict(batch_data)
            self.add(out.numpy(), list(filenames), label, generalise)
        self.save()


    @torch.no_grad()
    def search(self, x, extractor, nrt_neigh=10, retrieve_class='true'):
        t_model = time.time()
        if not self.feat_extract: # feat_extract is None in case of non transformer model thus True here 
            image = transforms.Resize((224, 224))(x)
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image)
        else:
            image = self.feat_extract(images=x, return_tensors='pt')['pixel_values'] # Applies the processing for the transformer model
	
        if extractor == 'vgg11' or extractor == 'resnet18' or extractor == "vgg16":
            out = encode(self.model, image.to(device=next(self.model.parameters()).device).view(-1, 3, 224, 224))
            out = out.reshape([out.shape[0],self.model.num_features])
        else:
            # Retrieves the result from the model
            out = self.model(image.to(device=next(self.model.parameters()).device).view(-1, 3, 224, 224))
        t_model = time.time() - t_model
        t_search = time.time()

        if retrieve_class == 'true':
            # Récupère l'index des nrt_neigh images les plus proches de x
            distance, labels = self.index_labeled.search(out.cpu().numpy(), nrt_neigh) 

            labels = [l for l in list(labels[0]) if l != -1]

            # retrieves the names of the images based on their index
            names = []
            for l in labels:
                n = self.r.get(str(l) + 'labeled').decode('utf-8')
                names.append(n)
            t_search = time.time() - t_search

            return names, distance.tolist(), t_model, t_search
        elif retrieve_class == 'false':
            distance, labels = self.index_unlabeled.search(out.cpu().numpy(), nrt_neigh)
            labels = [l for l in list(labels[0]) if l != -1]
            names = []
            for l in labels:
                n = self.r.get(str(l) + 'unlabeled').decode('utf-8')
                names.append(n)
            t_search = time.time() - t_search

            return names, distance.tolist(), t_model, t_search
        elif retrieve_class == 'mix':
            # retrieves nrt_neigh best in both cases
            distance_l, labels_l = self.index_labeled.search(out.cpu().numpy(), nrt_neigh)
            distance_u, labels_u = self.index_unlabeled.search(out.cpu().numpy(), nrt_neigh)
            labels_l = [l for l in list(labels_l[0]) if l != -1]
            labels_u = [l for l in list(labels_u[0]) if l != -1]

            # from the results of both, find hte best nrt_neigh (out of the 2* nrt_neigh)
            index = faiss.IndexFlatL2(1)
            index.add(np.array(distance_l, dtype=np.float32).reshape(-1, 1))
            index.add(np.array(distance_u, dtype=np.float32).reshape(-1, 1))

            _, labels = index.search(np.array([[0]], dtype=np.float32), nrt_neigh)

            names = []
            distance = []
            labels = [l for l in list(labels[0]) if l != -1]
            for l in labels:
                # Label comes from labelled list (in the first half of the reconstructed vector) 
                if l < nrt_neigh:
                    if l < len(labels_l):
                        n = self.r.get(str(labels_l[l]) + 'labeled').decode('utf-8')
                        distance.append(distance_l[0][l])
                        names.append(n)
                else: # Label comes from unlabelled list 
                    if l < len(labels_u) + nrt_neigh:
                        n = self.r.get(str(labels_u[l - nrt_neigh]) + 'unlabeled').decode('utf-8')
                        distance.append(distance_u[0][l - nrt_neigh])
                        names.append(n)
            t_search = time.time() - t_search

            return names, np.array(distance).reshape(1, -1).tolist(), t_model, t_search

    def remove(self, name):
        key = self.r.get(name).decode('utf-8')

        labeled = key.find('unlabeled') == -1
        if labeled:
            idx = key.find('labeled')
        else:
            idx = key.find('unlabeled')

        try:
            label = int(key[:idx])
        except:
            pass

        idsel = faiss.IDSelectorRange(label, label+1)

        if labeled:
            if self.device == 'gpu':
                self.index_labeled = faiss.index_gpu_to_cpu(self.index_labeled)
            self.index_labeled.remove_ids(idsel)
            self.save()
            self.r.delete(key + 'labeled')
            self.r.delete(name)
            if self.device == 'gpu':
                res_labeled = faiss.StandardGpuResources()
                self.index_labeled = faiss.index_cpu_to_gpu(res_labeled, 0, self.index_labeled)
        else:
            if self.device == 'gpu':
                self.index_unlabeled = faiss.index_gpu_to_cpu(self.index_unlabeled)
            self.index_unlabeled.remove_ids(idsel)
            self.save()
            self.r.delete(key + 'unlabeled')
            self.r.delete(name)
            if self.device == 'gpu':
                res_labeled = faiss.StandardGpuResources()
                self.index_unlabeled = faiss.index_cpu_to_gpu(res_labeled, 0, self.index_unlabeled)

        os.remove(name) # ? 

    def train_labeled(self):
        batch_size = 128
        x = []
        keys = []
        with open(self.filename + '_labeledvectors', 'rb') as file:
            while True:
                binary = file.read(4 + self.num_features * 4)

                if not binary:
                    break
                index, *vector = struct.unpack("i"+str(self.num_features)+"f", binary)
                keys.append(index)
                x.append(np.array(vector))
        if len(x) >= 10:
            num_clusters = int(np.sqrt(self.index_labeled.ntotal))

            self.quantizer = faiss.IndexFlatL2(self.model.num_features)
            self.index_labeled = faiss.IndexIVFFlat(self.quantizer, self.model.num_features,
                                                    num_clusters)

            if self.device == 'gpu':
                res_labeled = faiss.StandardGpuResources()
                self.index_labeled = faiss.index_cpu_to_gpu(res_labeled, 0, self.index_labeled)

            x = np.array(x, dtype=np.float32)
            print(x.shape)
            print(self.index_labeled.d)
            self.index_labeled.train(x)
            self.index_labeled.nprobe = num_clusters // 10

            num_batches = self.index_labeled.ntotal // batch_size

            for i in range(num_batches+1):
                if i == num_batches:
                    x_ = x[i * batch_size:, :]
                    key = keys[i * batch_size:]
                else:
                    x_ = x[i * batch_size: (i + 1) * batch_size, :]
                    key = keys[i * batch_size: (i+1) * batch_size]
                self.index_labeled.add_with_ids(x_, np.array(key, dtype=np.int64))

    def train_unlabeled(self):
        batch_size = 128
        x = []
        keys = []
        with open(self.filename + '_unlabeledvectors', 'rb') as file:
            while True:
                binary = file.read(4 + self.num_features * 4)

                if not binary:
                    break
                index, *vector = struct.unpack("i"+str(self.num_features)+"f", binary)
                keys.append(index)
                x.append(np.array(vector))
        if len(x) >= 10:
            num_clusters = int(np.sqrt(self.index_unlabeled.ntotal))
            self.quantizer = faiss.IndexFlatL2(self.model.num_features)
            self.index_unlabeled = faiss.IndexIVFFlat(self.quantizer, self.model.num_features,
                                                      num_clusters)

            if self.device == 'gpu':
                res_unlabeled = faiss.StandardGpuResources()
                self.index_unlabeled = faiss.index_cpu_to_gpu(res_unlabeled, 0, self.index_unlabeled)

            x = np.array(x, dtype=np.float32)

            self.index_unlabeled.train(x)
            self.index_unlabeled.nprobe = num_clusters // 10

            num_batches = self.index_unlabeled.ntotal // batch_size

            for i in range(num_batches+1):
                if i == num_batches:
                    x_ = x[i * batch_size:, :]
                    key = keys[i * batch_size:]
                else:
                    x_ = x[i * batch_size: (i + 1) * batch_size, :]
                    key = keys[i * batch_size: (i+1) * batch_size]
                self.index_unlabeled.add_with_ids(x_, np.array(key, dtype=np.int64))


    def save(self):
        if self.device != 'gpu':
            faiss.write_index(self.index_labeled, self.name + '_labeled')
            faiss.write_index(self.index_unlabeled, self.name + '_unlabeled')
        else:
            faiss.write_index(faiss.index_gpu_to_cpu(self.index_labeled), self.name + '_labeled')
            faiss.write_index(faiss.index_gpu_to_cpu(self.index_unlabeled), self.name + '_unlabeled')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--extractor',
        default='densenet'
    )

    parser.add_argument(
        '--weights'
    )

    parser.add_argument(
        '--db_name',
        default = 'db'
    )

    parser.add_argument(
        '--unlabeled',
        action='store_true'
    )

    parser.add_argument(
        '--num_features',
        default = 128,
        type=int
    )

    parser.add_argument(
        '--dr_model',
        action = "store_true" 
    )

    args = parser.parse_args()
    if args.extractor == 'vgg11' or args.extractor == "vgg16" or args.extractor == "resnet18": 
        model = builder.BuildAutoEncoder(args)     
        #total_params = sum(p.numel() for p in model.parameters())
        #print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
           
        builder.load_dict(args.weights, model)
        model.model_name = args.extractor
        model.num_features = args.num_features
    else: 
        model = models.Model(num_features=args.num_features, model=args.extractor, use_dr=args.dr_model, name=args.weights)
    database = Database(args.db_name, model, load=True)
    if args.unlabeled:
        database.train_unlabeled()
    else:
        database.train_labeled()
    database.save()
