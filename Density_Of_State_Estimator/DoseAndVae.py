import os

import torch
import tqdm
from torch.distributions import MultivariateNormal
from torch import optim
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from torch.utils.data import DataLoader
from .vae import VAE
from .dose import DoSE_SVM, kl_divergence, get_summary_stats
from math import log

deviceToUse = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_model(trainingDataset, testingDataset):
    os.makedirs("stats", exist_ok=True)  # Where summary stats for DoSE are stored
    trainLoader = DataLoader(dataset=trainingDataset, batch_size=128, num_workers=4, shuffle=True)
    testingLoader = DataLoader(dataset=testingDataset, batch_size=128, num_workers=4, pin_memory=True)

    inputShape = get_input_shape(trainingDataset)

    model = VAE(input_shape=inputShape, latent_size=2, hidden_size=64, observation='categorical')
    model.to(deviceToUse)
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.1)
    prior = MultivariateNormal(torch.zeros(2,device=deviceToUse), torch.eye(2,device=deviceToUse))

    train_loss_log, val_loss_log = [], []
    pbar = tqdm.trange(1, 21)

    for epoch in pbar:
        trainLoss, zs = train_vae(epoch, trainLoader, model, prior, optimizer, deviceToUse)
        pbar.set_description(f"Epoch: {epoch} | Train Loss: {trainLoss}")
        train_loss_log.append(trainLoss)

    ## VAE + DoSE(SVM)
    # Calculate
    # marginal
    # posterior
    # distribution
    # q(Z)
    marginal_posterior = get_marginal_posterior(trainLoader, model, deviceToUse)
    # Collect summary statistics of model on datasets
    train_summary_stats = get_summary_stats(trainLoader, model, marginal_posterior, 16, 4, 3, deviceToUse)
    torch.save(train_summary_stats, os.path.join("stats", "BETH_dose_1_stats_train.pth"))
    outlier_preds = []
    for seed in tqdm.trange(1, 6):
        outlier_preds.append(test_vae(seed, trainingDataset))


def train_vae(epoch, data_loader, model, prior, optimiser, device):
    model.train()
    zs = []
    train_loss = 0
    for i, (x, y) in enumerate(tqdm.tqdm(data_loader, leave=False)):
        x, y = x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)

        # Debug: Check first batch
        if i == 0:
            print(f"\nFirst batch shape: {x.shape}")
            print(f"First batch min values per column: {x.min(dim=0)[0]}")
            print(f"First batch max values per column: {x.max(dim=0)[0]}")
            print(f"Expected num_classes from model: {model.input_shape}")

            # Check if any value exceeds its expected range
            for col_idx in range(x.size(1)):
                col_max = x[:, col_idx].max().item()
                expected_max = model.input_shape[col_idx].item() - 1
                if col_max > expected_max:
                    print(f"ERROR: Column {col_idx} has value {col_max} but expected max is {expected_max}")

        observation, posterior, z = model(x)
        loss = -observation.log_prob(x) + kl_divergence(z, posterior, prior)
        loss = -torch.logsumexp(-loss.view(loss.size(0), -1), dim=1).mean() - log(1)
        zs.append(z.detach())  # Store posterior samples
        train_loss += loss.item()

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    return train_loss / len(data_loader), torch.cat(zs)

def get_marginal_posterior(data_loader, model, device):
    model.eval()
    posteriors = []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm.tqdm(data_loader, leave=False)):
            x, y = x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)
            posteriors.append(model.encode(x))
    means, stddevs = torch.cat([p.mean for p in posteriors], dim=0), torch.cat([p.stddev for p in posteriors], dim=0)
    mix = Categorical(torch.ones(means.size(0), device=device))
    comp = Independent(Normal(means, stddevs), 1)
    return MixtureSameFamily(mix, comp)

def test_vae(seed, train_dataset):
    # Calculate result over ensemble of trained models
    # Load dataset summary statistics
    train_summary_stats = torch.load(os.path.join("stats", "BETH_dose_1_stats_train.pth"))
    print(f"train shape: {train_summary_stats.shape}")

    dose_svm = DoSE_SVM(train_summary_stats)

def get_input_shape(dataset):  # note that does not return actual shape, but is used to configure model for categorical data
    num_classes = dataset.data.max(dim=0)[0].long() + 1

    # Debug: Print actual max values before any manual overrides
    print("Actual max values per column:", dataset.data.max(dim=0)[0])
    print("Calculated num_classes before override:", num_classes)

    num_classes[5] = 1011  # Manually set eventId range as 0-1011 (1010 is max value)
    print("Input size after override: ", num_classes)
    return num_classes
