import os
import argparse
from utils import *
from tqdm import tqdm
from torch import optim
from model import my_model
import torch.nn.functional as F
import time
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--sigma', type=float, default=0.01, help='Sigma of gaussian distribution')
parser.add_argument('--dataset', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--device', type=str, default='cuda:0', help='device')

args = parser.parse_args()
for args.dataset in ["cora"]:
    print("Using {} dataset".format(args.dataset))
    file = open("result.csv", "a+")
    print(args.dataset, file=file)
    file.close()

    if args.dataset == 'cora':
        args.cluster_num = 7
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
        args.tao = 0.1
    elif args.dataset == 'citeseer':
        args.cluster_num = 6
        args.gnnlayers = 2
        args.lr = 5e-5
        args.dims = [500]
        args.tao = 1.0
    elif args.dataset == 'amap':
        args.cluster_num = 8
        args.gnnlayers = 5
        args.lr = 5e-5
        args.dims = [500]
        args.tao = 0.3
    elif args.dataset == 'bat':
        args.cluster_num = 4
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
        args.tao = 0.2
    elif args.dataset == 'eat':
        args.cluster_num = 4
        args.gnnlayers = 5
        args.lr = 5e-2
        args.dims = [500]
        args.tao = 1.0
    elif args.dataset == 'uat':
        args.cluster_num = 4
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
        args.tao = 0.2


    # load data
    X, y, A = load_graph_data(args.dataset, show_details=False)

    features = X
    true_labels = y
    adj = sp.csr_matrix(A)

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    print('Laplacian Smoothing...')
    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()

    path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
    if os.path.exists(path):
        sm_fea_s = sp.csr_matrix(np.load(path, allow_pickle=True)).toarray()
    else:
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)
        np.save(path, sm_fea_s, allow_pickle=True)

    sm_fea_s = torch.FloatTensor(sm_fea_s)
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()


    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    for seed in range(12, 32, 2):
        setup_seed(seed)
        best_acc, best_nmi, best_ari, best_f1, prediect_labels = clustering(sm_fea_s, true_labels, args.cluster_num)
        model = my_model([features.shape[1]] + args.dims, A.shape[0], args.cluster_num, args.dataset)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(args.device)
        inx = sm_fea_s.to(args.device)
        target = torch.FloatTensor(adj_1st).to(args.device)

        print('Start Training...')
        for epoch in tqdm(range(args.epochs)):
            model.train()
            z2, z1, H = model(inx, torch.tensor(A).float().to(args.device), args.dataset, is_train=True,
                              sigma=args.sigma)
            S = z1 @ z2.T

            loss1 = F.mse_loss(S, target)
            loss2 = instanceloss(0.1, z1, z2)
            MU_loss = Modula(torch.tensor(A).float().to(args.device), H)
            loss = 1 * loss1 + 0.1 * loss2 - 0.01 * MU_loss
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                z2, z1, _ = model(inx, torch.tensor(A).float().to(args.device), args.dataset, is_train=False,
                                  sigma=args.sigma)
                S = z1 @ z2.T
                hidden_emb = z2

                acc, nmi, ari, f1, predict_labels = clustering(hidden_emb, true_labels, args.cluster_num)
                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1

        tqdm.write('acc: {}, nmi: {}, ari: {}, f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))
        file = open("result.csv", "a+")
        print(args.cluster_num, file=file)
        print(best_acc, best_nmi, best_ari, best_f1, file=file)
        file.close()
        acc_list.append(best_acc)
        nmi_list.append(best_nmi)
        ari_list.append(best_ari)
        f1_list.append(best_f1)

    acc_list = np.array(acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)
    f1_list = np.array(f1_list)

    file = open("result.csv", "a+")
    print(args.gnnlayers, args.lr, args.dims, args.sigma, file=file)
    print(round(acc_list.mean(), 2), round(acc_list.std(), 2), file=file)
    print(round(nmi_list.mean(), 2), round(nmi_list.std(), 2), file=file)
    print(round(ari_list.mean(), 2), round(ari_list.std(), 2), file=file)
    print(round(f1_list.mean(), 2), round(f1_list.std(), 2), file=file)
    file.close()
