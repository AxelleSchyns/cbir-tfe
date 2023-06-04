import faiss
from matplotlib import pyplot as plt
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
import utils
class Database:
    def __init__(self, filename, model, load=False, device='cuda:0'):
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


        if device == 'gpu':
            self.index_labeled = faiss.index_cpu_to_gpu(res_labeled, 0, self.index_labeled)
            self.index_unlabeled = faiss.index_cpu_to_gpu(res_unlabeled, 0, self.index_unlabeled)

        
        if model.model_name == 'deit':
                self.feat_extract = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
		                                                         size=224, do_center_crop=False,
		                                                         image_mean=[0.485, 0.456, 0.406],
		                                                         image_std=[0.229, 0.224, 0.225]) 
                self.transformer = True
        elif model.model_name == 'cvt':
                self.feat_extract = ConvNextImageProcessor.from_pretrained("microsoft/cvt-21", size=224, do_center_crop=False,
		                                                                  image_mean=[0.485, 0.456, 0.406],
		                                                                  image_std=[0.229, 0.224, 0.225])
                self.transformer = True
        elif model.model_name == 'conv':
                self.feat_extract = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224", size=224, do_center_crop=False,
		                                                                  image_mean=[0.485, 0.456, 0.406],
		                                                                  image_std=[0.229, 0.224, 0.225])
                self.transformer = True
        else:
                self.feat_extract = None     
                self.transformer = False                                                            
    # x = vector of images 
    def add(self, x, names, label, generalise=0, labels=None):
        if label:
            last_id = int(self.r.get('last_id_labeled').decode('utf-8'))
            # Add x.shape ids and images to the current list of ids of Faiss. 
            # Id is given by the last given id to which 1 is added each time, ending with id = last_id + x.shape[0]
            self.index_labeled.add_with_ids(x, np.arange(last_id, last_id + x.shape[0])) 

            # Open the file of the database corresponding to the labelled case.
            #with open(self.filename + '_labeledvectors', 'ab') as file:
            # Encode in the file database the images with their name, and gives them the appropriate id
            # Zip() Creates a list of tuples, with one element of names and one of x (=> allows to iterate on both list at the same time) 
            if generalise == 3:
                for n, x_, l in zip(names, x, labels):
                    l = str(l)
                    json_val = json.dumps([{'name':n}, {'label':l}])
                    self.r.set(str(last_id)+ 'labeled', json_val)
                    self.r.set(n, str(last_id)+'labeled')
                    #binary = struct.pack("i"+str(self.num_features)+"f",last_id, *x_)
                    #file.write(binary)
                    last_id += 1
            else:
                for n, x_  in zip(names, x):
                    self.r.set(str(last_id) + 'labeled', n) # Set the name of the image at key = id
                    self.r.set(n, str(last_id) + 'labeled') # Set the id of the image at key = name 
                    #binary = struct.pack("i"+str(self.num_features)+"f",last_id, *x_)
                    #file.write(binary)
                    #file.write('\n' + str(last_id) + str(x_)) # Writes in the file the id alongside the image 
                    last_id += 1

            self.r.set('last_id_labeled', last_id) # Update the last id to take into account the added images

        # Same as previous for unlabelled data
        else:
            last_id = int(self.r.get('last_id_unlabeled').decode('utf-8'))
            self.index_unlabeled.add_with_ids(x, np.arange(last_id, last_id + x.shape[0]))

            #with open(self.filename + '_unlabeledvectors', 'ab') as file:
            if generalise == 3:
                for n, x_, l in zip(names, x, labels):
                    l = str(l)
                    json_val = json.dumps([{'name':n}, {'label':l}])
                    self.r.set(str(last_id)+ 'unlabeled', json_val)
                    self.r.set(n, str(last_id)+'unlabeled')
                    #binary = struct.pack("i"+str(self.num_features)+"f",last_id, *x_)
                    #file.write(binary)
                    last_id += 1
            else:
                for n, x_  in zip(names, x):
                    self.r.set(str(last_id) + 'unlabeled', n) # Set the name of the image at key = id
                    self.r.set(n, str(last_id) + 'unlabeled') # Set the id of the image at key = name 
                    #binary = struct.pack("i"+str(self.num_features)+"f",last_id, *x_)
                    #file.write(binary)
                    last_id += 1

            self.r.set('last_id_unlabeled', last_id)

    @torch.no_grad()
    def add_dataset(self, data_root, extractor, generalise=0, name_list=[], label=True):
        # Create a dataset from a directory root
        if name_list == []:
            data = dataset.AddDataset(data_root, extractor, generalise)
        # create a dataset from a list of image names
        else:
            data = dataset.AddDatasetList(data_root, extractor, name_list)
        loader = torch.utils.data.DataLoader(data, batch_size=128, num_workers=12, pin_memory=True)
        t_model = 0
        t_indexing = 0
        t_transfer = 0
        for i, (images, filenames) in enumerate(loader):
            images = images.view(-1, 3, 224, 224).to(device=next(self.model.parameters()).device)
            if extractor == 'vgg11' or extractor == 'resnet18' or extractor == 'vgg16' or extractor == 'resnet50':
                t = time.time()
                #_, out = self.model(images)
                #out = out.cpu()
                out = utils.encode(self.model, images)
                out = out.reshape([out.shape[0],self.model.num_features])
                t_im = time.time() - t
            elif extractor == 'vae':
                t = time.time()
                mu, logvar = self.model.encode(images)
                out = self.model.reparameterize(mu, logvar)
                dec = self.model.decode(out)
                
                out = out.view(-1, self.model.num_features) #print(out.shape)
                #print(out.shape)
                # For visualisation of the reconstruction
                """dec = dec.view(128, 3, 224, 224)
                plt.subplot(1, 2, 1)
                plt.imshow(  images[0].cpu().permute(1, 2, 0)  )
                plt.subplot(1, 2, 2)
                plt.imshow(  dec[0].permute(1, 2, 0)  )
                plt.show()"""
                t_im = time.time() - t
            elif extractor == 'auto':
                t = time.time()
                reconstructed, flattened, latent, weights = self.model.model(images)
                out = latent
                out = out.view(-1, self.model.num_features)
                t_im = time.time() - t
                # display image and its reconstruction
                """dec = out1.cpu()
                dec = dec.view(128, 3, 224, 224)
                plt.subplot(1, 2, 1)
                plt.imshow(  images[0].cpu().permute(1, 2, 0)  )
                plt.subplot(1, 2, 2)
                plt.imshow(  dec[0].permute(1, 2, 0)  )
                plt.show()"""

            elif extractor == 'byol':
                t = time.time()
                out, emb = self.model.model(images, return_embedding=True)
                t_im = time.time() - t
            else:
                # Encode the images using the given model 
                t = time.time()
                out = self.model(images)
                t_im = time.time() - t
                
            t = time.time()
            out = out.cpu()
            t_transfer = t_transfer + time.time() - t

            t = time.time()
            if generalise == 3:
                kmeans = pickle.load(open("weights_folder/kmeans_50.pkl","rb"))
                batch_data = np.array([utils.load_image(path) for path in filenames])
                labels = kmeans.predict(batch_data)
                self.add(out.numpy(), list(filenames), label, generalise, labels)
            else:
                self.add(out.numpy(), list(filenames), label, generalise)

            t_im_ind = time.time() - t
            t_indexing = t_indexing + t_im_ind
            t_model = t_model + t_im
        print("Time of the model: "+str(t_model))
        print("Time of the transfer: "+str(t_transfer))
        print("Time of the indexing: "+str(t_indexing))
        
        self.save()
        


    @torch.no_grad()
    def search(self, x, extractor, generalise=0, nrt_neigh=10, retrieve_class='true'):
        image = x.view(-1, 3, 224, 224).to(device=next(self.model.parameters()).device)
        t_model = 0
        if extractor == 'vgg11' or extractor == 'resnet18' or extractor == "vgg16" or extractor == "resnet50":
            t_model = time.time()
            out = utils.encode(self.model, image)
            out = out.reshape([out.shape[0],self.model.num_features])
            """out1, out = self.model(image.to(device=next(self.model.parameters()).device).view(-1, 3, 224, 224))
            out = out.cpu()

            out1 = out1.cpu() 
            out1 = out1.view(-1, 3, 224, 224)
            plt.subplot(1,2,1)
            # display image and its reconstruction
            plt.imshow(  image.cpu().permute(1, 2, 0)  )
            plt.subplot(1,2,2)
            plt.imshow( out1[0].permute(1, 2, 0))
            plt.show()"""
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
            """out = original.cpu() 
            out1 = out1.view(-1, 3, 224, 224)
            plt.subplot(1,2,1)
            # display image and its reconstruction
            plt.imshow(  image.cpu().permute(1, 2, 0)  )
            plt.subplot(1,2,2)
            plt.imshow( out1[0].permute(1, 2, 0))
            plt.show()"""
            t_model = time.time() - t_model
        elif extractor == 'byol':
            t_model = time.time()
            out, emb = self.model.model(image, return_embedding=True)
            t_model = time.time() - t_model
        else:
            # Retrieves the result from the model
            t_model = time.time()
            out = self.model(image)
            t_model = time.time() - t_model
        t_transfer = time.time()
        out = out.cpu()
        t_transfer = time.time() - t_transfer
        
        t_search = time.time()
        if retrieve_class == 'true':
            # Récupère l'index des nrt_neigh images les plus proches de x
            distance, labels = self.index_labeled.search(out.numpy(), nrt_neigh) 
            labels = [l for l in list(labels[0]) if l != -1]
            # retrieves the names of the images based on their index
            values = []
            if generalise == 3:
                names = []
                labs = []
                for l in labels:
                    v = self.r.get(str(l) + 'labeled').decode('utf-8')
                    v = json.loads(v)
                    names.append(v[0]['name'])
                    labs.append(v[1]['label'])
                values.append(names)
                values.append(labs)
            else:
                for l in labels:
                    v = self.r.get(str(l) + 'labeled').decode('utf-8')
                    values.append(v)
            t_search = time.time() - t_search

            return values, distance[0], t_model, t_search, t_transfer
        elif retrieve_class == 'false':
            distance, labels = self.index_unlabeled.search(out.numpy(), nrt_neigh)
            labels = [l for l in list(labels[0]) if l != -1]
            # retrieves the names of the images based on their index
            values = []
            if generalise == 3:
                names = []
                labs = []
                for l in labels:
                    v = self.r.get(str(l) + 'unlabeled').decode('utf-8')
                    v = json.loads(v)
                    names.append(v[0]['name'])
                    labs.append(v[1]['label'])
                values.append(names)
                values.append(labs)
            else:
                for l in labels:
                    v = self.r.get(str(l) + 'unlabeled').decode('utf-8')
                    values.append(v)
            t_search = time.time() - t_search

            return values, distance.tolist(), t_model, t_search, t_transfer
        elif retrieve_class == 'mix':
            # retrieves nrt_neigh best in both cases
            distance_l, labels_l = self.index_labeled.search(out.numpy(), nrt_neigh)
            distance_u, labels_u = self.index_unlabeled.search(out.numpy(), nrt_neigh)
            labels_l = [l for l in list(labels_l[0]) if l != -1]
            labels_u = [l for l in list(labels_u[0]) if l != -1]

            # from the results of both, find the best nrt_neigh (out of the 2* nrt_neigh)
            index = faiss.IndexFlatL2(1)
            index.add(np.array(distance_l, dtype=np.float32).reshape(-1, 1))
            index.add(np.array(distance_u, dtype=np.float32).reshape(-1, 1))

            _, labels = index.search(np.array([[0]], dtype=np.float32), nrt_neigh)

            values = []
            distance = []
            labels = [l for l in list(labels[0]) if l != -1]
            if generalise == 3:
                names = []
                labs = []
                for l in labels:
                    if l < nrt_neigh:
                        if l < len(labels_l):
                            v = self.r.get(str(labels_l[l]) + 'labeled').decode('utf-8')
                            v = json.loads(v)
                            names.append(v[0]['name'])
                            labs.append(v[1]['label'])
                            distance.append(distance_l[0][l])
                    else:
                        if l < len(labels_u) + nrt_neigh:
                            v = self.r.get(str(labels_l[l]) + 'unlabeled').decode('utf-8')
                            v = json.loads(v)
                            names.append(v[0]['name'])
                            labs.append(v[1]['label'])
                            distance.append(distance_l[0][l])
                values.append(names)
                values.append(labs)
            else:
                for l in labels:
                    # Label comes from labelled list (in the first half of the reconstructed vector) 
                    if l < nrt_neigh:
                        if l < len(labels_l): #Check there is no mistake and w<e are not trying to access and index that does not exist
                            v = self.r.get(str(labels_l[l]) + 'labeled').decode('utf-8')
                            distance.append(distance_l[0][l])
                            values.append(v)
                    else: # Label comes from unlabelled list 
                        if l < len(labels_u) + nrt_neigh:
                            v = self.r.get(str(labels_u[l - nrt_neigh]) + 'unlabeled').decode('utf-8')
                            distance.append(distance_u[0][l - nrt_neigh])
                            values.append(v)
            t_search = time.time() - t_search

            return values, np.array(distance).reshape(1, -1).tolist(), t_model, t_search, t_transfer

    def remove(self, name):
        key = self.r.get(name).decode('utf-8')
        
        # Remove the supplementary inscription behind the index
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

        #os.remove(name) # dangereux si on veut juste retirer le fichier de la db mais pas le supprimer

    def train_labeled(self, generalise):
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
                vec = self.index_labeled.index.reconstruct(index)
                x.append(vec)
        if len(x) >= 10:
            num_clusters = int(np.sqrt(self.index_labeled.ntotal))

            self.quantizer = faiss.IndexFlatL2(self.model.num_features)
            self.index_labeled = faiss.IndexIVFFlat(self.quantizer, self.model.num_features,
                                                    num_clusters)

            if self.device == 'gpu':
                res_labeled = faiss.StandardGpuResources()
                self.index_labeled = faiss.index_cpu_to_gpu(res_labeled, 0, self.index_labeled)

            x = np.asarray(x, dtype=np.float32)
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

            if self.device == 'cuda:0':
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

    parser.add_argument(
        '--generalise',
        default = 0,
        type = int
    )

    args = parser.parse_args()
    if args.extractor == 'vgg11' or args.extractor == "vgg16" or args.extractor == "resnet18" or args.extractor == "resnet50": 
        model = builder.BuildAutoEncoder(args)
        builder.load_dict(args.weights, model)
        model.model_name = args.extractor
        model.num_features = args.num_features
    else: 
        model = models.Model(num_features=args.num_features, model=args.extractor, use_dr=args.dr_model, name=args.weights)
    database = Database(args.db_name, model, load=True)
    if args.unlabeled:
        database.train_unlabeled(args.generalise)
    else:
        database.train_labeled(args.generalise)
    database.save()
