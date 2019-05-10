import pandas as pd 
import numpy as np    
import matplotlib.pyplot as plt   
import torch 
import torchvision 
from torchvision import transforms
from sklearn.model_selection import train_test_split
import argparse
from dataset import iMetDataset
from transform import RandomSizedCrop
from model import mymodel
from score import f2_score
from threshold_search import threshold_search

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', default='resnet50')
    arg('--batch_size', type=int, default=128)
    arg('--lr', type=float, default=1e-4)
    arg('--epochs', type=int, default=25)
    arg('--fold', type=int, default=0)
    args = parser.parse_args()


    transformed_dataset = iMetDataset(csv_file='train.csv', 
                                  label_file="labels.csv", 
                                  img_path="train_unzip/", 
                                  root_dir='../input/',
                                  transform=transforms.Compose([
                                      RandomSizedCrop((224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          [0.485, 0.456, 0.406], 
                                          [0.229, 0.224, 0.225])
                                  ]))

    for i in range(len(transformed_dataset)):
    
        sample = transformed_dataset[i]
        print(i, sample['image'].size())
        print(i, sample['labels'])
        if i == 3:
            break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    model = mymodel(args.model)
    if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    model.to(device)

    ##train
    max_epochs=args.epochs
    lr=args.lr
    
    batch_size=args.batch_size

    from torch.utils.data import DataLoader
    from torchvision import utils

    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    criterion=torch.nn.BCELoss()
    sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    def train(epoch, train_loader):
        model.train()
        total_loss = 0.0
        f2 = 0.0
    
        for batch_idx, sample in enumerate(train_loader):
            image, labels = sample["image"].to(device, dtype=torch.float), sample["labels"].to(device, dtype=torch.float)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
            if (batch_idx+1) % 10 == 0:
                print('Train Epoch {}: [({}/{})] Loss: {:.6f}'.format(epoch,batch_idx+1,len(train_loader), total_loss/ (batch_idx + 1)))

        torch.save(model.state_dict(), "../output/model_" + str(epoch) + ".pth")
        return total_loss / (batch_idx + 1)

    def validate(epoch, valid_loader):
        model.eval();
    
        test_loss = 0.0
    
        true_ans_list = []
        preds_cat = []
    
        with torch.no_grad():
            for batch_idx, sample in enumerate(valid_loader):
                image, labels = sample["image"].to(device, dtype=torch.float), sample["labels"].to(device, dtype=torch.float)
                output = model(image)
                loss = criterion(output, labels)

                test_loss += loss.item()
                true_ans_list.append(labels)
                predictions = output
                preds_cat.append(predictions)

            all_true_ans = torch.cat(true_ans_list)
            all_preds = torch.cat(preds_cat)
        
            f2_thr,f2_eval = threshold_search(all_true_ans, all_preds)
            print("f2_thr",f2_thr)
            
        return test_loss / (batch_idx + 1), f2_eval

    train_df = pd.read_csv("../input/train.csv")
    train_indices, valid_indices = train_test_split(train_df.index, test_size=0.20, random_state=33)

    train_loader = DataLoader(
        transformed_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=4)

    valid_loader = DataLoader(
        transformed_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(valid_indices),
        num_workers=4)

    train_losses = []
    valid_losses = []
    valid_f2s = []

    best_model_f2 = 0.0
    best_model = None
    best_model_ep = 0


    for epoch in range(1, max_epochs + 1):
        train_loss = train(epoch, train_loader)
        train_losses.append(train_loss)
        print('Train Epoch {}: train_loss: {:.6f}'.format(epoch,train_loss))
    
        valid_loss, valid_f2 = validate(epoch, valid_loader)
        valid_losses.append(valid_loss)
        valid_f2s.append(valid_f2)
        print('Train Epoch {}: valid_loss: {:.6f}'.format(epoch,valid_loss))
        print('Train Epoch {}: valid_f2: {:.6f}'.format(epoch,valid_f2))
    
        if valid_f2 >= best_model_f2:
            best_model = model.state_dict()
            best_model_f2 = valid_f2
            best_model_ep = epoch

    bestmodel_logstr = 'Best f2_score is {} on epoch {}'.format(best_model_f2,best_model_ep)
    print(bestmodel_logstr)
    torch.save(best_model, "../output/best_model_" + str(best_model_f2) + ".pth")

    xs = list(range(1, len(train_losses) + 1))

    plt.plot(xs, train_losses, label = 'Train loss');
    plt.plot(xs, valid_losses, label = 'Val loss');
    plt.plot(xs, valid_f2s, label = 'Val f2');
    plt.legend();
    plt.xticks(xs);
    plt.xlabel('Epochs');


        
if __name__ == '__main__':
    main()
