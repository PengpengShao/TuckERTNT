import os
import argparse
import tqdm
import torch
from torch import optim, nn
from datasets import TemporalDataset
from models import TuckERT, TuckERTNT
from utils import avg_both, temporal_regularizer, temporal_regularizer1, temporal_regularizer2,temporal_regularizer3,\
    temporal_regularizer4, emb_regularizer, creat_M

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='TuckERTNT', type=str,help="TuckERT, TuckERTNT")
parser.add_argument("--dataset", type=str, default="ICEWS14", nargs="?",help="Which dataset to use:ICEWS14,ICEWS05-15, gdelt")
parser.add_argument('--embedding_dim', default=300, type=int,help="embedding dimension")
parser.add_argument('--max_epochs', default=1000, type=int,help="Number of epochs.")
parser.add_argument("--batch_size", type=int, default=1000, nargs="?",help="Batch size.")
parser.add_argument("--lr", type=float, default=0.2, nargs="?",help="Learning rate.0.2")
parser.add_argument('--time_weight', default=1, type=float,help="Timestamp regularizer strength 1")
parser.add_argument('--emb_weight', default=0.002, type=float,help="embedding regularizer strength 0.002")
parser.add_argument("--input_dropout", type=float, default=0., nargs="?",help="Input layer dropout.")
parser.add_argument("--hidden_dropout1", type=float, default=0., nargs="?",help="Dropout after the first hidden layer.")
parser.add_argument("--hidden_dropout2", type=float, default=0, nargs="?",help="Dropout after the second hidden layer.")
parser.add_argument('--no_time_emb', default=False, action="store_true", help="Use a specific embedding for non temporal relations")
parser.add_argument('--valid_freq', default=5, type=int,help="Number of epochs between each valid.")
parser.add_argument('--sigma', default=5, type=int,help="gaussian.6")
args = parser.parse_args()
dataset = TemporalDataset(args.dataset)
sizes = dataset.get_shape()
M = creat_M(args, sizes)
if args.model_name == 'TuckERT':
    model = TuckERT(sizes, args)
elif args.model_name == 'TuckERTNT':
    model = TuckERTNT(sizes, args)
else:
    print('no model')
model = model.cuda()
M = M.cuda()
model.init()
opt = optim.Adagrad(model.parameters(), lr=args.lr)
train_loss, train_ave_loss = [], []
for epoch in range(args.max_epochs):
    examples = torch.from_numpy(dataset.get_train().astype('int64'))
    model.train()
    actual_examples = examples[torch.randperm(examples.shape[0]), :]
    Loss = nn.CrossEntropyLoss(reduction='mean')
    verbose: bool = True
    with tqdm.tqdm(total=examples.shape[0], unit='ex', disable= not verbose) as bar:
        bar.set_description(f'train loss')
        b_begin = 0
        while b_begin < examples.shape[0]:
            input_batch = actual_examples[b_begin:b_begin + args.batch_size].cuda()
            predictions, factors, time = model.forward(input_batch)
            truth = input_batch[:, 2]
            loss_pre = Loss(predictions, truth)
            loss_time = temporal_regularizer4(time, args.time_weight, M)
            loss_emb = emb_regularizer(factors, args.emb_weight)
            loss = loss_pre + loss_emb + loss_time
            opt.zero_grad()
            loss.backward()
            opt.step()
            b_begin += args.batch_size
            bar.update(input_batch.shape[0])
            bar.set_postfix(loss=f'{loss.item():.2f}',)
            train_loss.append(loss)
        average_loss = sum(train_loss)/len(train_loss)
        train_ave_loss.append(average_loss)
        train_loss = []


    if epoch < 0 or (epoch + 1) % args.valid_freq == 0:

        valid, test, train = [avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000)) for split in ['valid', 'test', 'train']]

        print("valid: ","MRR", valid['MRR'], "hits",valid['hits@[1,3,10]'])
        print("test: ", "MRR", test['MRR'], "hits",test['hits@[1,3,10]'])
        print("train: ", "MRR", train['MRR'], "hits",train['hits@[1,3,10]'])



