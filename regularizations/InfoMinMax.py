import torch
import torch.nn as nn

class Discriminator_Bilinear(nn.Module):
    def __init__(self, n_h, n_c):
        super(Discriminator_Bilinear, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_c, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h):
        logits = torch.squeeze(self.f_k(c, h))
        return logits

class INFOMAX(nn.Module):
    def __init__(self, n_h, n_neg):
        super(INFOMAX, self).__init__()
        self.mi_estimator = MINE(n_h, n_h, n_h)
        self.n_neg = n_neg

        self.sigm = nn.Sigmoid()

        self.disc_hc = Discriminator_Bilinear(n_h, n_h)
        self.disc_cc = Discriminator_Bilinear(n_h, n_h)
        self.disc_hh = Discriminator_Bilinear(n_h, n_h)

        self.mask = nn.Dropout(0.1)
        self.b_xent = nn.BCELoss()

    def random_gen(self, base, num):
        idx =torch.randint(0, base.shape[0], [num*self.n_neg]) 
        shuf = base[idx].squeeze()
        return shuf

    def forward(self, c_p, h_p, c_n, h_n):

        c_all_pp = self.random_gen(c_p, h_p.shape[0])
        c_all_nn = self.random_gen(c_n, h_n.shape[0])
        c_all_pn = self.random_gen(c_p, h_n.shape[0])
        c_all_np = self.random_gen(c_n, h_p.shape[0])

        h_p = h_p.repeat([self.n_neg, 1])
        h_n = h_n.repeat([self.n_neg, 1])

        c = torch.cat((c_all_pp, c_all_nn, c_all_pn, c_all_np), dim=0)
        h = torch.cat((h_p, h_n, h_n, h_p), dim=0)

        ret = self.disc_hc(c, h)
        ret = self.sigm(ret)

        lbl_pp = torch.ones(c_all_pp.shape[0])
        lbl_nn = torch.ones(c_all_nn.shape[0])
        lbl_pn = torch.zeros(c_all_pn.shape[0])
        lbl_np = torch.zeros(c_all_np.shape[0])
        lbl = torch.cat((lbl_pp, lbl_nn, lbl_pn, lbl_np))
        lbl = lbl.to(h.device)

        loss1 = self.b_xent(ret, lbl) 
        
        return loss1
        

class INFOMIN(nn.Module):
    def __init__(self, n_h, n_neg):
        super(INFOMIN, self).__init__()
        self.sigm = nn.Sigmoid()

        self.disc = Discriminator_Bilinear(n_h, n_h)

        self.n_neg = n_neg
        self.drop = nn.Dropout(0.1)
        self.b_xnet = nn.BCELoss()

    
    def forward(self, h, edge_batch):
        #here we assume all data have the same edge number
        n_edges = edge_batch[edge_batch==0].shape[0]

        rand_tail1 = torch.randint(0, n_edges, [edge_batch.shape[0]*self.n_neg], device=h.device)
        neg_index1 = edge_batch.repeat_interleave(self.n_neg, 0) * n_edges + rand_tail1

        h = h.repeat_interleave(self.n_neg, 0)
        h1 = self.drop(h)
        h2 = self.drop(h)

        ret_pos = self.disc(h2, h1)
        ret_neg2 = self.disc(h2, h1[neg_index1])
        ret = torch.cat([ret_pos, ret_neg2], 0)
        ret = self.sigm(ret)

        lbl_p = torch.ones(ret_pos.shape[0])
        lbl_n = torch.zeros(ret_neg2.shape[0])
        lbl = torch.cat([lbl_p, lbl_n], 0).to(h.device)

        loss = self.b_xnet(ret, lbl)

        return loss