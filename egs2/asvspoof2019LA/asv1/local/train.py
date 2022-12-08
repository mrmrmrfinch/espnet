"""
    AASIST implementation for ESPnet2
    usage:
    python3 local/train.py \
            --feature_dir ${feature_extract_dir} \
            --exp_dir ${expdir} \
            --ngpu ${ngpu}
            --config ${config}
"""
import argparse
import json
import os
import sys
import warnings
from dataset import ASVSpoof2019LA
from importlib import import_module
from pathlib import Path
from shutil import copy
from AASIST import Model
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_curve

def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def main():        
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    # parse arguments
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    if args.ngpu >= 1:
        device = torch.device("cuda:0")
    else:
        print("Warning: using CPU for training.")
        device = torch.device("cpu")
    
    training_set = ASVSpoof2019LA(args.feature_dir, args.data_dir, part="train")
    validation_set = ASVSpoof2019LA(args.feature_dir, args.data_dir, part="dev")
    test_set = ASVSpoof2019LA(args.feature_dir, args.data_dir, part="eval")
    
    print("Training set size: ", len(training_set))
    print("Validation set size: ", len(validation_set))
    print("Test set size: ", len(test_set))
    
    # read in config file in json format
    config = json.load(open(args.config))
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])
    num_epochs = int(config["num_epochs"])
    learning_rate = float(config["learning_rate"])
    
    # create data loaders
    training_dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers) # shuffle on training.
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # model and loss
    model = Model(config["model_config"]).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # take the lowest loss on validation set as the best model
    min_val_eer = 1
    min_val_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        print("Epoch: ", epoch, "/", num_epochs, "\n")
        model.train()
        for i, (featureTensor, label, attackType) in enumerate(tqdm(training_dataloader)):
            featureTensor = featureTensor.to(device)
            label = label.to(device)
            features, scores = model(featureTensor)
            loss = criterion(scores, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # save checkpoint
        torch.save(model.state_dict(), os.path.join(args.exp_dir, "checkpoint" + str(epoch) + ".pth"))
        model.eval()
        with torch.no_grad():
            preds = []
            labels = []
            for i, (featureTensor, label, attackType) in enumerate(tqdm(validation_dataloader)):
                featureTensor = featureTensor.to(device)
                label = label.to(device)
                features, scores = model(featureTensor)
                scores = torch.nn.functional.softmax(scores, dim=1)[:, 0]
                preds.append(scores.cpu().numpy())
                labels.append(label.cpu().numpy())
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            eer = compute_eer(preds[labels == 0], preds[labels == 1])[0]
            print("Validation EER: ", eer)
            if eer < min_val_eer:
                min_val_eer = eer
                min_val_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.exp_dir, "model.pth"))
    
    # Evaluation
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, "model.pth")))
    model.eval()
    with torch.no_grad():
        preds = []
        labels = []
        attackTypes = []
        for i, (featureTensor, label, attackType) in enumerate(tqdm(test_dataloader)):
            featureTensor = featureTensor.to(device)
            label = label.to(device)
            features, scores = model(featureTensor)
            scores = torch.nn.functional.softmax(scores, dim=1)[:, 0]
            preds.append(scores.cpu().numpy())
            labels.append(label.cpu().numpy())
            attackTypes.append(attackType)
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        eer = compute_eer(preds[labels == 0], preds[labels == 1])[0]
        print("Test EER: ", eer)
    
if __name__ == "__main__":
    main()