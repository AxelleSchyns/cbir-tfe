import faiss
import models
import struct
import torch
import dataset
import redis
import numpy as np
from transformers import DeiTFeatureExtractor, ConvNextImageProcessor
import time
import json
import argparse
import builder
import pickle 
import utils
# File that contains all functions relative to the manipulation of FAISS and Redis

# Class that represents the database
class Database:
    def __init__(self, filename, model, load=False, device='cuda:0'):
        self.num_features = model.num_features
        self.model = model
        self.device = device
        self.filename = filename

        res = faiss.StandardGpuResources() # Allocation of steams and temporary memory 

        # A database was previously constructed
        if load == True:
            self.index = faiss.read_index(filename + '_labeled')
            self.r = redis.Redis(host='localhost', port='6379', db=0)
        else:
            # No database to load, has to build it 
            self.index = faiss.IndexFlatL2(self.num_features)
            self.index = faiss.IndexIDMap(self.index)
            self.r = redis.Redis(host='localhost', port='6379', db=0)
            self.r.flushdb()

            self.r.set('last_id', 0) # Set a value in redis with key = last_id


        if device == 'gpu':
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        
        if model.model_name == 'deit':
                self.feat_extract = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
		                                                         size=224, do_center_crop=False,
		                                                         image_mean=[0.485, 0.456, 0.406],
		                                                         image_std=[0.229, 0.224, 0.225]) 
                self.transformer = True
        elif model.model_name == 'cvt' or model.model_name == 'conv':
                self.feat_extract = ConvNextImageProcessor.from_pretrained("microsoft/cvt-21", size=224, do_center_crop=False,
		                                                                  image_mean=[0.485, 0.456, 0.406],
		                                                                  image_std=[0.229, 0.224, 0.225])
                self.transformer = True
        else:
                self.feat_extract = None     
                self.transformer = False     

    # x = vector of images 
    def add(self, x, names, labels=None, generalise=0):
        last_id = int(self.r.get('last_id').decode('utf-8'))
        # Add x.shape ids and images to the current list of ids of Faiss. 
        self.index_labeled.add_with_ids(x, np.arange(last_id, last_id + x.shape[0])) 

        # Zip() Creates a list of tuples, with one element of names and one of x (=> allows to iterate on both list at the same time) 
        if generalise == 3:
            for n, x_, l in zip(names, x, labels):
                l = str(l)
                json_val = json.dumps([{'name':n}, {'label':l}])
                self.r.set(str(last_id), json_val)
                self.r.set(n, str(last_id))
                last_id += 1
        else:
            for n, x_  in zip(names, x):
                self.r.set(str(last_id), n) # Set the name of the image at key = id
                self.r.set(n, str(last_id)) # Set the id of the image at key = name 
                last_id += 1

        self.r.set('last_id', last_id) # Update the last id to take into account the added images

        

    @torch.no_grad()
    def add_dataset(self, data_root, extractor, generalise=0, label=True):
        # Create a dataset from a directory root
        data = dataset.AddDataset(data_root, extractor, generalise)
        loader = torch.utils.data.DataLoader(data, batch_size=128, num_workers=12, pin_memory=True)
        t_model = 0
        t_indexing = 0
        t_transfer = 0
        for i, (images, filenames) in enumerate(loader):
            images = images.view(-1, 3, 224, 224).to(device=next(self.model.parameters()).device)
            if extractor == 'vgg11' or extractor == 'resnet18' or extractor == 'vgg16' or extractor == 'resnet50':
                t = time.time()
                out = utils.encode(self.model, images)
                out = out.reshape([out.shape[0],self.model.num_features])
                t_im = time.time() - t
            elif extractor == 'vae':
                t = time.time()
                mu, logvar = self.model.encode(images)
                out = self.model.reparameterize(mu, logvar)
                out = out.view(-1, self.model.num_features) 
                t_im = time.time() - t
            elif extractor == 'auto':
                t = time.time()
                reconstructed, flattened, latent, weights = self.model.model(images)
                out = latent
                out = out.view(-1, self.model.num_features)
                t_im = time.time() - t

            elif extractor == 'byol':
                t = time.time()
                out, emb = self.model.model(images, return_embedding=True)
                t_im = time.time() - t
            else:
                t = time.time()
                out = self.model(images)
                t_im = time.time() - t
                
            t = time.time()
            out = out.cpu()
            t_transfer = t_transfer + time.time() - t

            t = time.time()
            if generalise == 3:
                kmeans = pickle.load(open("weights_folder/kmeans_104.pkl","rb"))
                batch_data = np.array([utils.load_image(path) for path in filenames])
                labels = kmeans.predict(batch_data)
                self.add(out.numpy(), list(filenames), labels, generalise)
            else:
                self.add(out.numpy(), list(filenames),  generalise  = generalise)

            t_im_ind = time.time() - t
            t_indexing = t_indexing + t_im_ind
            t_model = t_model + t_im
        print("Time of the model: "+str(t_model))
        print("Time of the transfer: "+str(t_transfer))
        print("Time of the indexing: "+str(t_indexing))
        
        self.save()
        


    @torch.no_grad()
    def search(self, x, extractor, generalise=0, nrt_neigh=10):
        image = x.view(-1, 3, 224, 224).to(device=next(self.model.parameters()).device)
        t_model = 0
        if extractor == 'vgg11' or extractor == 'resnet18' or extractor == "vgg16" or extractor == "resnet50":
            t_model = time.time()
            out = utils.encode(self.model, image)
            out = out.reshape([out.shape[0],self.model.num_features])
            t_model = time.time() - t_model
        elif extractor == 'vae':
            t_model = time.time()
            mu, logvar = self.model.encode(image)
            out = self.model.reparameterize(mu, logvar)
            out = out.view(-1, self.model.num_features)
            t_model = time.time() - t_model
        elif extractor == 'auto':
            t_model = time.time()
            reconstructed, flattened, latent, weights = self.model.model(image)
            out = latent
            out = out.view(-1, self.model.num_features)
            t_model = time.time() - t_model
        elif extractor == 'byol':
            t_model = time.time()
            out, emb = self.model.model(image, return_embedding=True)
            t_model = time.time() - t_model
        else:
            t_model = time.time()
            out = self.model(image)
            t_model = time.time() - t_model
        t_transfer = time.time()
        out = out.cpu()
        t_transfer = time.time() - t_transfer
        
        t_search = time.time()

        # Récupère l'index des nrt_neigh images les plus proches de x
        distance, labels = self.index.search(out.numpy(), nrt_neigh) 
        labels = [l for l in list(labels[0]) if l != -1]
        # retrieves the names of the images based on their index
        values = []
        if generalise == 3:
            names = []
            labs = []
            for l in labels:
                v = self.r.get(str(l)).decode('utf-8')
                v = json.loads(v)
                names.append(v[0]['name'])
                labs.append(v[1]['label'])
            values.append(names)
            values.append(labs)
        else:
            for l in labels:
                v = self.r.get(str(l)).decode('utf-8')
                values.append(v)
        t_search = time.time() - t_search

        return values, distance[0], t_model, t_search, t_transfer
        

    def remove(self, name):
        key = self.r.get(name).decode('utf-8')
        label = int(key)

        idsel = faiss.IDSelectorRange(label, label+1)

        if self.device == 'gpu':
            self.index = faiss.index_gpu_to_cpu(self.index)
        self.index.remove_ids(idsel)
        self.save()
        self.r.delete(key)
        self.r.delete(name)
        if self.device == 'gpu':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def train_index(self, generalise):
        batch_size = 128
        x = []
        keys = []
        all_keys = self.r.keys("*")
        for k in all_keys:
            k = k.decode("utf-8")
            # Only keep the indexes as keys, not the names nor last_id 
            if k.find('/') == -1 and k.find('_')==-1:
                end_ind = k.find('l') # Remove the unecessary part of the indeex
                index = k[:end_ind]
                keys.append(index)
                index = int(index)
                vec = self.index.index.reconstruct(index)
                x.append(vec)
        if len(x) >= 10:
            num_clusters = int(np.sqrt(self.index.ntotal))

            self.quantizer = faiss.IndexFlatL2(self.model.num_features)
            self.index = faiss.IndexIVFFlat(self.quantizer, self.model.num_features,
                                                    num_clusters)

            if self.device == 'gpu':
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

            x = np.asarray(x, dtype=np.float32)
            self.index.train(x)
            self.index.nprobe = num_clusters // 10

            num_batches = self.index.ntotal // batch_size

            for i in range(num_batches+1):
                if i == num_batches:
                    x_ = x[i * batch_size:, :]
                    key = keys[i * batch_size:]
                else:
                    x_ = x[i * batch_size: (i + 1) * batch_size, :]
                    key = keys[i * batch_size: (i+1) * batch_size]
                self.index.add_with_ids(x_, np.array(key, dtype=np.int64))

    


    def save(self):
        if self.device != 'gpu':
            faiss.write_index(self.index, self.filename )
        else:
            faiss.write_index(faiss.index_gpu_to_cpu(self.index), self.filename)

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
        '--num_features',
        default = 128,
        type=int
    )

    parser.add_argument(
        '--dr_model',
        action = "store_true" 
    )

    parser.add_argument(
        '--generalise',
        default = 0,
        type = int
    )

    args = parser.parse_args()
    
    # Retrieve the pretrained model 
    model = models.Model(num_features=args.num_features, model=args.extractor, use_dr=args.dr_model, name=args.weights)

    # Create the database
    database = Database(args.db_name, model, load=True)

    # Train the index
    database.train_labeled(args.generalise)
    database.save()
