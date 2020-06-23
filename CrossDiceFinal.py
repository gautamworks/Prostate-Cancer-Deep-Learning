#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
from glob import glob
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import copy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
# from torch.utils.tensorboard import SummaryWriter
import time
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score


def plot_confusion_matrix(cm,
                          target_names,epoch,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    fig=plt.figure()
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    normalize=True
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
    plt.savefig('/home/gleason/Ispit_Parth/Code_Files_Recent/Final_Confusion_matrix.png', dpi=200)



# In[17]:
class Prostate_data_train(Dataset):

    def __init__(self, img_path='/home/gleason/Ispit_Parth/harvard_data/TMA_Train', mask_path='/home/gleason/Ispit_Parth/harvard_data/Gleason_masks_train',
                 dataset_type='train', img_size=512, valid_split=['ZT76'], test_split=['ZT80'], num_classes=5):
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_size = img_size
        self.num_classes = num_classes
        self.file_names = []
        self.file_dict = {}
        self.flag_dict = {}
        self.data = {}
        self.transform = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        for file in os.listdir(self.img_path):
            _file_name = file.split('\\')[-1]
            _slide_type = _file_name.split('.')[0].split('_')[0]
            if not(_slide_type=='ZT80'):
                try :
                    self.file_dict[_slide_type].append(_file_name)
                except :
                    self.file_dict[_slide_type] = []
                    self.file_dict[_slide_type].append(_file_name)
        random.seed(10)
        # for _slide_type in self.file_dict:
        #     random.shuffle(self.file_dict[_slide_type])

        fraction = 1
        if dataset_type=='train':
            for _slide_type in self.file_dict:
                for i in range(int(len(self.file_dict[_slide_type])*fraction)):
                    self.file_names.append(self.file_dict[_slide_type][i])
                    self.flag_dict[self.file_dict[_slide_type][i]] = False
        else :
            for _slide_type in self.file_dict:
                for i in range(int(len(self.file_dict[_slide_type])*fraction),len(self.file_dict[_slide_type])):
                    self.file_names.append(self.file_dict[_slide_type][i])
                    self.flag_dict[self.file_dict[_slide_type][i]] = False
        random.shuffle(self.file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        _file_name = self.file_names[idx]
        _file_flag = self.flag_dict[_file_name]
        if _file_flag:
            return self.data[_file_name]
        else:
            img = Image.open(self.img_path + '/' + _file_name).resize((self.img_size, self.img_size)).convert('RGB')
            mask = Image.open(self.mask_path + '/' + 'mask_' + _file_name.split('.')[0] + '.png').resize(
                (self.img_size, self.img_size)).convert('RGB')
            random.seed(10)
            ## transforms
            # if random.random()<0.5:
            #     img = TF.affine(img,10,(0,0),1.17,0)
            #     mask = TF.affine(mask,10,(0,0),1.17,0)

            img_array = np.asarray(img)
            mask_array = np.asarray(mask)
            oneh_mask = np.zeros((self.num_classes, self.img_size, self.img_size))
            for x in range(self.img_size):
                for y in range(self.img_size):
                    pixel_class = self.get_class(mask_array[x, y, :])
                    oneh_mask[pixel_class, x, y] = 1
            img_tensor = self.transform(img)
            mask_tensor = torch.from_numpy(oneh_mask).view(5, self.img_size, self.img_size)
            self.data[_file_name] = (img_tensor, mask_tensor)
            self.flag_dict[_file_name] = True
            return self.data[_file_name]

    def get_class(self, rgb):
        '''
        takes in rgb values of the pixel and returns the class of the pixel
        '''
        rgb_n = rgb / 255.0

        # white
        if rgb_n[0] > 0.8 and rgb_n[1] > 0.8 and rgb_n[2] > 0.8:
            return 4
        # red
        elif rgb_n[0] > 0.8 and rgb_n[1] < 0.8 and rgb_n[2] < 0.8:
            return 3
        # yellow
        elif rgb_n[0] > 0.8 and rgb_n[1] > 0.8 and rgb_n[2] < 0.8:
            return 2
        # green
        elif rgb_n[0] < 0.8 and rgb_n[1] > 0.8 and rgb_n[2] < 0.8:
            return 0
        # blue
        elif rgb_n[0] < 0.8 and rgb_n[1] < 0.8 and rgb_n[2] > 0.8:
            return 1
        else:
            raise ValueError('Weird rgb combination! Did not match any of 5 classes.')


class Prostate_data_valid(Dataset):

    def __init__(self, img_path='/home/gleason/Ispit_Parth/harvard2_data/Test_Data', mask_path='/home/gleason/Ispit_Parth/harvard2_data/Test_Masks',
                 dataset_type='valid', img_size=512, valid_split=['ZT76'], test_split=['ZT80'], num_classes=5):
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_size = img_size
        self.num_classes = num_classes
        self.file_names = []
        self.file_dict = {}
        self.flag_dict = {}
        self.data = {}
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        for file in os.listdir(self.img_path):
            _file_name = file.split('\\')[-1]
            _slide_type = _file_name.split('.')[0].split('_')[0]
            if(_slide_type=='ZT80'):
                try :
                    self.file_dict[_slide_type].append(_file_name)
                except :
                    self.file_dict[_slide_type] = []
                    self.file_dict[_slide_type].append(_file_name)
        random.seed(10)
        # for _slide_type in self.file_dict:
            # random.shuffle(self.file_dict[_slide_type])

        fraction = 1
        if dataset_type=='valid':
            for _slide_type in self.file_dict:
                for i in range(int(len(self.file_dict[_slide_type])*fraction)):
                    self.file_names.append(self.file_dict[_slide_type][i])
                    self.flag_dict[self.file_dict[_slide_type][i]] = False
        else :
            for _slide_type in self.file_dict:
                for i in range(int(len(self.file_dict[_slide_type])*fraction),len(self.file_dict[_slide_type])):
                    self.file_names.append(self.file_dict[_slide_type][i])
                    self.flag_dict[self.file_dict[_slide_type][i]] = False
        random.shuffle(self.file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        _file_name = self.file_names[idx]
        _file_flag = self.flag_dict[_file_name]
        if _file_flag:
            return self.data[_file_name]
        else:
            img = Image.open(self.img_path + '/' + _file_name).resize((self.img_size, self.img_size)).convert('RGB')
            mask = Image.open(self.mask_path + '/' + 'mask1_' + _file_name.split('.')[0] + '.png').resize(
                (self.img_size, self.img_size)).convert('RGB')
            ## transforms
            # if random.random()<0.5:
                # img = TF.affine(img,10,(0,0),1.17,0)
                # mask = TF.affine(mask,10,(0,0),1.17,0)

            img_array = np.asarray(img)
            mask_array = np.asarray(mask)
            oneh_mask = np.zeros((self.num_classes, self.img_size, self.img_size))
            for x in range(self.img_size):
                for y in range(self.img_size):
                    pixel_class = self.get_class(mask_array[x, y, :])
                    oneh_mask[pixel_class, x, y] = 1
            img_tensor = self.transform(img)
            mask_tensor = torch.from_numpy(oneh_mask).view(5, self.img_size, self.img_size)
            self.data[_file_name] = (img_tensor, mask_tensor)
            self.flag_dict[_file_name] = True
            return self.data[_file_name]

    def get_class(self, rgb):
        '''
        takes in rgb values of the pixel and returns the class of the pixel
        '''
        rgb_n = rgb / 255.0

        # white
        if rgb_n[0] > 0.8 and rgb_n[1] > 0.8 and rgb_n[2] > 0.8:
            return 4
        # red
        elif rgb_n[0] > 0.8 and rgb_n[1] < 0.8 and rgb_n[2] < 0.8:
            return 3
        # yellow
        elif rgb_n[0] > 0.8 and rgb_n[1] > 0.8 and rgb_n[2] < 0.8:
            return 2
        # green
        elif rgb_n[0] < 0.8 and rgb_n[1] > 0.8 and rgb_n[2] < 0.8:
            return 0
        # blue
        elif rgb_n[0] < 0.8 and rgb_n[1] < 0.8 and rgb_n[2] > 0.8:
            return 1
        else:
            raise ValueError('Weird rgb combination! Did not match any of 5 classes.')



def soft_dice_loss(y_pred,y_true):
    '''y_pred: (-1,5,512,512) :predictions
       y_true: (512,512,5) : targets
       compute the soft dice loss

       '''    
    #y_true = y_true.view(-1,5,512,512)
    # print(y_pred.shape)
    # print(y_true.shape)
    epsilon = 1e-7
    dice_numerator = epsilon + 2 * torch.sum(y_true.double()*y_pred.double(),dim=(2,3))
    dice_denominator = epsilon + torch.sum(y_true*y_true,dim=(2,3)).double() + torch.sum(y_pred*y_pred,dim=(2,3)).double()
    dice_loss = 1 - torch.mean(dice_numerator/dice_denominator)

    return dice_loss


def show_train_predictions(model,trainset,device,idx_list):
    fig,axes = plt.subplots(nrows=len(idx_list),ncols=3,figsize=(15,15))
    plt.axis('off')
    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255])
    for i in range(len(idx_list)):
        idx = idx_list[i]
        trainset1=trainset[idx][0]
        trainset1 = inv_normalize(trainset1)
        # input_img = Image.fromarray(np.asarray(trainset[idx][0].view(512,512,3).squeeze()).astype('uint8'), 'RGB')
        trans = transforms.ToPILImage()
        input_img=trans(trainset1)
        target_img = get_rgb(trainset[idx][1].squeeze())
        with torch.no_grad():
            pred_img = get_rgb(model(trainset[idx][0].view(-1,3,trainset.img_size,trainset.img_size).float().to(device)).squeeze())
        if len(idx_list)>1:
            axes[i,0].imshow(input_img)
            axes[i,1].imshow(pred_img)
            axes[i,2].imshow(target_img)
        else:
            axes[0].imshow(input_img)
            axes[1].imshow(pred_img)
            axes[2].imshow(target_img)
    if len(idx_list)>1:
        axes[0,0].set_title('Input')
        axes[0,1].set_title('Prediction')
        axes[0,2].set_title('Target')
    else:
        axes[0].set_title('Input')
        axes[1].set_title('Prediction')
        axes[2].set_title('Target')
    fig.savefig('/home/gleason/Ispit_Parth/Code_Files_Recent/cross_dice_images/train_thurs_dice_imagecorrect_train.jpg')
    plt.close('all')



def show_valid_predictions(model,trainset,device,idx_list,index):
    # print("yes1")
    fig,axes = plt.subplots(nrows=len(idx_list),ncols=3,figsize=(15,15))
    plt.axis('off')
    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255])
    for i in range(len(idx_list)):
        idx = idx_list[i]
        trainset1=trainset[idx][0]
        trainset1 = inv_normalize(trainset1)
        trans = transforms.ToPILImage()
        input_img=trans(trainset1)
        target_img = get_rgb(trainset[idx][1].squeeze())
        with torch.no_grad():
            pred_img = get_rgb(model(trainset[idx][0].view(-1,3,trainset.img_size,trainset.img_size).float().to(device)).squeeze())
        if len(idx_list)>1:
            axes[i,0].imshow(input_img)
            axes[i,1].imshow(pred_img)
            axes[i,2].imshow(target_img)
        else:
            axes[0].imshow(input_img)
            axes[1].imshow(pred_img)
            axes[2].imshow(target_img)
    if len(idx_list)>1:
        axes[0,0].set_title('Input')
        axes[0,1].set_title('Prediction')
        axes[0,2].set_title('Target')
    else:
        axes[0].set_title('Input')
        axes[1].set_title('Prediction')
        axes[2].set_title('Target')
    fig.savefig('/home/gleason/Ispit_Parth/Code_Files_Recent/cross_dice_images/valid_'+str(index)+'.jpg')
    plt.close('all')

def get_rgb(tensor_img):
    pallete_dict = {
        0 : [0,255,0],
        1 : [0,0,255],
        2 : [255,255,0],
        3 : [255,0,0],
        4 : [255,255,255]
    }
    img_h = tensor_img.size()[2]
    out_img = np.zeros((img_h,img_h,3))
    for h in range(img_h):
        for w in range(img_h):
            pixel_class = torch.argmax(tensor_img[:,h,w]).item()
            out_img[h,w,:] = pallete_dict[pixel_class]
    final_img = Image.fromarray(out_img.astype('uint8'), 'RGB')
    return final_img


def fit(dataloaders,model,criterion1, criterion2, optimizer,epochs,imgsize,device,dataset_sizes,trainset,lr,wd,scheduler=None):
    '''
    expect a dataloader dictionary with keys 'train' and 'valid'
    reporting only valid loss on verbose, returns both train and valid loss history
    '''
    # comment = f'_sameslide_lr={lr}_wd={wd}'
    # tb = SummaryWriter(log_dir=f'ss_fullrun_1543_0206/lr={lr}_wd={wd}',comment=comment)
    tb=SummaryWriter()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_valid_loss = 2.
    record_dict = {'train':{'loss':[],'acc':[]},'valid':{'loss':[],'acc':[]}}

    for epoch in range(epochs):
        print('EPOCH : {}/{}'.format(epoch+1,epochs))
        for phase in ['train','valid']:
            labels_full=[]
            preds_full=[]
            since = time.time()
            if phase=='train':
                model.eval()
            else :
                model.eval()
            running_loss = 0.
            running_acc = 0.
            for inputs,targets in tqdm(dataloaders[phase]):
                inputs = inputs.float().view(-1,3,imgsize,imgsize).to(device)
                targets1 = torch.argmax(targets.view(-1,5,imgsize,imgsize),dim=1).to(device)
                targets2 = targets.view(-1,5,imgsize,imgsize).to(device)
                # with torch.set_grad_enabled(phase=='train'):
                with torch.no_grad():
                    # optimizer.zero_grad()
                    outputs = model(inputs)
                    preds = torch.argmax(outputs,dim=1)
                    loss1=criterion1(outputs,targets1)
                    loss2=criterion2(outputs,targets2)
                    loss=(loss1/2).float()+(loss2/2).float()
                    acc = torch.mean((targets1==preds).float()).item()
                    # if phase=='train':
                    #     loss.backward()
                    #     optimizer.step()
                running_loss += loss.item()*inputs.size()[0]
                running_acc += acc*inputs.size()[0]
                if(phase=='valid'):
                    lb=targets1.data.cpu().numpy()
                    pr=preds.cpu().numpy()
                    preds_full.extend(lb)
                    labels_full.extend(pr)

            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_acc/dataset_sizes[phase]
            if phase == 'train':
                tb.add_scalar('Train Loss',epoch_loss,epoch)
            else:
                tb.add_scalar('Valid Loss',epoch_loss,epoch)

            if phase=='valid' and not(scheduler is None): # FOR SINGLE SAMPLE
                scheduler.step(epoch_loss)
            record_dict[phase]['loss'].append(epoch_loss)
            record_dict[phase]['acc'].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,epoch_acc))
            # print('{} Loss: {:.4f} Acc: '.format(phase, epoch_loss))
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            # deep copy the model
            if phase == 'valid' and epoch_loss < best_valid_loss:
                best_valid_loss = epoch_loss
                # best_model_wts = copy.deepcopy(model.state_dict())
                epoch_no=epoch
            # print(np.array(labels_full))
            # print(np.array(preds_full))
            # print(np.array(labels_full).flatten())
            # print(np.array(preds_full).flatten())
        lbf=np.array(labels_full).flatten()
        pdf=np.array(preds_full).flatten()

        lbf_index=np.where(lbf==4)
        lbf=np.delete(lbf,lbf_index)
        pdf=np.delete(pdf,lbf_index)

        pdf_index=np.where(pdf==4)
        lbf=np.delete(lbf,pdf_index)
        pdf=np.delete(pdf,pdf_index)
        # print(lbf,pdf)
        print("Cohens-Kappa")
        print(cohen_kappa_score(lbf,pdf))
        cm = confusion_matrix(lbf,pdf)
        # fpr, tpr, thresholds = metrics.roc_curve(np.array(labels_full), np.array(preds_full), pos_label=1)
        # print("AUC")
        # print(metrics.auc(fpr, tpr))
        # print(labels_full,preds_full)
        # print(len(labels_full),len(preds_full))
        print(cm)
        # Show confusion matrix in a separate window
        # fig=plt.figure()
        plot_confusion_matrix(cm, normalize= False,target_names = ['Benign','Gleason Grade-3','Gleason Grade-4','Gleason Grade-5'],title= "Confusion Matrix",epoch=epoch)

        for name, weight in model.named_parameters():
            if weight.grad is not None:
                tb.add_histogram(name,weight,epoch)
        print()

    print('Best valid loss: {:4f}'.format(best_valid_loss))
    print('epoch number: {:4f}'.format(epoch_no))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    tb.close()
    return record_dict,model


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    trainset = Prostate_data_train(dataset_type='train',img_size=512,num_classes=5)
    validset = Prostate_data_valid(dataset_type='valid',img_size=512,num_classes=5)

    imgsize = trainset.img_size
    for lr in [5e-5] :
        for wd in [1e-1]:
            train_dl = DataLoader(trainset,batch_size=4,num_workers=2)
            valid_dl = DataLoader(validset,batch_size=4,num_workers=2)
            dataset_sizes = {'train':len(trainset),'valid':len(validset)}

            from res_unet import ResUnet
            model = ResUnet(num_classes=5)
            model.require_encoder_grad(requires_grad=True)
            model = model.to(device)
            model=torch.load('/home/gleason/Ispit_Parth/Code_Files_Recent/final_crossentropy_valid_no_transforms')
            num_epochs = 1
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dl))
            dataloaders = {'train':train_dl,'valid':valid_dl}
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = soft_dice_loss
            history,model = fit(dataloaders,model,criterion1, criterion2, optimizer,num_epochs,imgsize,device,dataset_sizes,trainset,lr,wd,scheduler=None)
            print("Printing Train_Predictions")
            show_train_predictions(model,trainset,device,[0,1,2,3,4,5,6,7,8,9,34,67,100,120,130,140,220,300])
            print("Printing Valid_Predictions")
            for i in range(0,100,5):
                 show_valid_predictions(model,validset,device,[i,i+1,i+2,i+3,i+4],i)
            print("Done")

if __name__=='__main__':
    main()