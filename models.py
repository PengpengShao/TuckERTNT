from typing import Tuple, List, Dict
import torch
import numpy as np
from torch.nn.init import xavier_normal_

class TuckERT(torch.nn.Module):
    def __init__(
            self, sizes: Tuple[int, int, int, int],
           args, init_size: float = 1
    ):
        super(TuckERT, self).__init__()
        self.sizes = sizes
        self.no_time_emb = args.no_time_emb
        dim = args.embedding_dim
        self.E = torch.nn.Embedding(sizes[0], dim, padding_idx=0)
        self.R = torch.nn.Embedding(sizes[1], dim, padding_idx=0)
        self.T = torch.nn.Embedding(sizes[3], dim, padding_idx=0)
        self.R_noT = torch.nn.Embedding(sizes[1], dim, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (dim, dim, dim)),dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(args.input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(args.hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(args.hidden_dropout2)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)
        xavier_normal_(self.T.weight.data)
        xavier_normal_(self.R_noT.weight.data)

    def forward(self, x):
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        lhs = self.E(x[:, 0])
        rel = self.R(x[:, 1])
        rel_no_time = self.R_noT(x[:, 1])
        rhs = self.E(x[:, 2])
        time = self.T(x[:, 3])
        E = self.E.weight

        rel_t = rel * time

        lhs = self.bn0(lhs)
        h1 = self.input_dropout(lhs)
        h = h1.view(-1, 1, lhs.size(1))

        w = self.W.view(rel_t.size(1), -1)
        W_mat = torch.mm(rel_t, w)
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(h, W_mat)
        x = x.view(-1, lhs.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)

        x = x @ E.t()
        pred = x

        regularizer = (
            torch.norm(lhs, p=4),
            torch.norm((rel * time), p=4),
            torch.norm(rel_no_time, p=4),
            torch.norm(rhs, p=4),
            torch.norm(w, p=4),
        )


        return pred, regularizer, self.T.weight


    def get_queries(self, queries: torch.Tensor):
        lhs = self.E(queries[:, 0])
        rel = self.R(queries[:, 1])
        rel_no_time = self.R_noT(queries[:, 1])
        time = self.T(queries[:, 3])

        rel_t = rel * time


        lhs = self.bn0(lhs)
        h1 = self.input_dropout(lhs)
        h = h1.view(-1, 1, lhs.size(1))

        w = self.W.view(rel_t.size(1), -1)
        W_mat = torch.mm(rel_t, w)
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(h, W_mat)
        x = x.view(-1, lhs.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        return x

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.E.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def score(self, x):
        lhs = self.E(x[:, 0])
        rel = self.R(x[:, 1])
        rel_no_time = self.R_noT(x[:, 1])
        rhs = self.E(x[:, 2])
        time = self.T(x[:, 3])

        rel_t = rel * time


        lhs = self.bn0(lhs)
        h1 = self.input_dropout(lhs)
        h = h1.view(-1, 1, lhs.size(1))

        w = self.W.view(rel_t.size(1), -1)
        W_mat = torch.mm(rel_t, w)
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(h, W_mat)
        x = x.view(-1, lhs.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)

        target = x * rhs

        return torch.sum(target, 1, keepdim=True)


    def get_ranking(self,
            queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)
                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks



class TuckERTNT(torch.nn.Module):
    def __init__(
            self, sizes: Tuple[int, int, int, int],
           args, init_size: float = 1
    ):
        super(TuckERTNT, self).__init__()
        self.sizes = sizes

        self.no_time_emb = args.no_time_emb

        dim = args.embedding_dim
        self.E = torch.nn.Embedding(sizes[0], dim, padding_idx=0)
        self.R = torch.nn.Embedding(sizes[1], dim, padding_idx=0)
        self.T = torch.nn.Embedding(sizes[3], dim, padding_idx=0)
        self.R_noT = torch.nn.Embedding(sizes[1], dim, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (dim, dim, dim)),dtype=torch.float, device="cuda", requires_grad=True))


        self.input_dropout = torch.nn.Dropout(args.input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(args.hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(args.hidden_dropout2)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)
        xavier_normal_(self.T.weight.data)
        xavier_normal_(self.R_noT.weight.data)


    def forward(self, x):
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        lhs = self.E(x[:, 0])
        rel = self.R(x[:, 1])
        rel_no_time = self.R_noT(x[:, 1])
        rhs = self.E(x[:, 2])
        time = self.T(x[:, 3])
        E = self.E.weight

        rel_t = rel * time + rel_no_time

        lhs = self.bn0(lhs)
        h1 = self.input_dropout(lhs)
        h = h1.view(-1, 1, lhs.size(1))

        w = self.W.view(rel_t.size(1), -1)
        W_mat = torch.mm(rel_t, w)
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(h, W_mat)
        x = x.view(-1, lhs.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)

        x = x@E.t()
        pred = x


        regularizer = (
            torch.norm(lhs, p=4),
            torch.norm((rel * time), p=4),
            torch.norm(rel_no_time, p=4),
            torch.norm(rhs, p=4),
            torch.norm(w, p=4),
        )

        return pred, regularizer, self.T.weight


    def get_queries(self, queries: torch.Tensor):
        lhs = self.E(queries[:, 0])
        rel = self.R(queries[:, 1])
        rel_no_time = self.R_noT(queries[:, 1])
        time = self.T(queries[:, 3])

        rel_t = rel * time + rel_no_time

        h0 = self.bn0(lhs)
        h1 = self.input_dropout(h0)
        h = h1.view(-1, 1, lhs.size(1))

        w = self.W.view(rel_t.size(1), -1)
        W_mat = torch.mm(rel_t, w)
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(h, W_mat)
        x = x.view(-1, lhs.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        return x

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.E.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def score(self, x):
        lhs = self.E(x[:, 0])
        rel = self.R(x[:, 1])
        rel_no_time = self.R_noT(x[:, 1])
        rhs = self.E(x[:, 2])
        time = self.T(x[:, 3])

        rel_t = rel * time + rel_no_time

        h0 = self.bn0(lhs)
        h1 = self.input_dropout(h0)
        h = h1.view(-1, 1, lhs.size(1))

        w = self.W.view(rel_t.size(1), -1)
        W_mat = torch.mm(rel_t, w)
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(h, W_mat)
        x = x.view(-1, lhs.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)

        target = x * rhs

        return torch.sum(target, 1, keepdim=True)


    def get_ranking(self,
            queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)
                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

