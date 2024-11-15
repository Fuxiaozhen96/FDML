import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from utils import log_normal, log_normal_mixture
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FDML(nn.Module):
    def __init__(self, args):
        super(FDML, self).__init__()
        self.args = args

        self.y_z_1 = nn.Linear(256, 256)
        self.y_z_2 = nn.Linear(256, 512)
        self.y_z_3 = nn.Linear(512, 256)
        self.y_z_mu = nn.Linear(256, args.latent_dim)
        self.y_z_wn = nn.utils.weight_norm(self.y_z_mu)
        self.y_z_logvar = nn.Linear(256, args.latent_dim)

        self.x_z_1 = nn.Linear(args.feature_dim, 256)
        self.x_z_2 = nn.Linear(256, 512)
        self.x_z_3 = nn.Linear(512, 256)
        self.x_z_mu = nn.Linear(256, args.latent_dim)
        self.x_z_wn = nn.utils.weight_norm(self.x_z_mu)
        self.x_z_logvar = nn.Linear(256, args.latent_dim)

        self.x_zs_1 = nn.Linear(args.feature_dim, 256)
        self.x_zs_2 = nn.Linear(256, 256)
        self.x_zs_mu = nn.Linear(256, args.latent_dim)
        # self.x_zs_wn = nn.utils.weight_norm(self.x_zs_mu)
        self.x_zs_logvar = nn.Linear(256, args.latent_dim)

        self.xz_y_1 = nn.Linear(args.latent_dim+args.feature_dim,256)
        self.xz_y_2 = nn.Linear(256, 256)
        self.xz_y = nn.Linear(256, args.label_dim)

        self.yz_x_1 = nn.Linear(args.latent_dim+args.label_dim+args.latent_dim, 256)
        self.yz_x_2 = nn.Linear(256, 256)
        self.yz_x = nn.Linear(256, args.feature_dim)
        self.dropout = nn.Dropout(p=args.keep_prob)

        self.z_y_1 = nn.Linear(args.latent_dim, 256)
        self.z_y_2 = nn.Linear(256, 256)
        self.z_y = nn.Linear(256, args.label_dim)
        self.dropout = nn.Dropout(p=args.keep_prob)

        self.weight_1 = Parameter(torch.Tensor(args.feature_dim, 1024))
        self.weight_2 = Parameter(torch.Tensor(1024, 256))
    def y_z(self,label_emb,A):
        support_1 = torch.matmul(label_emb, self.weight_1)
        embed_1= torch.matmul(A, support_1)
        embed_1 = self.dropout(F.relu(embed_1))
        embed_1= F.normalize(embed_1, dim=1)
        support_2 = torch.matmul(embed_1, self.weight_2)
        embed_2= torch.matmul(A, support_2)
        embed_2 = self.dropout(F.relu(embed_2))
        embed_2= F.normalize(embed_2, dim=1)
        x1 = self.dropout(F.relu(self.y_z_1(embed_2)))
        x2 = self.dropout(F.relu(self.y_z_2(x1)))
        x3=  self.dropout(F.relu(self.y_z_3(x2)))
        mu=self.y_z_mu(x3)
        logvar=self.y_z_logvar(x3)
        return mu,logvar

    def x_z(self,feature):
        x1 = self.dropout(F.relu(self.x_z_1(feature)))
        x2 = self.dropout(F.relu(self.x_z_2(x1)))
        x3 = self.dropout(F.relu(self.x_z_3(x2)))
        mu = self.x_z_mu(x3)
        logvar=self.x_z_logvar(x3)
        return mu,logvar

    def x_zs(self,feature):
        x1 = self.dropout(F.relu(self.x_zs_1(feature)))
        x2 = self.dropout(F.relu(self.x_zs_2(x1)))
        mu = self.x_zs_mu(x2)
        logvar=self.x_zs_logvar(x2)
        return mu,logvar

    def feat_reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z=mu + eps * std
        return z

    def label_reparameterize(self, mu, y):
        z = torch.matmul(y, mu) / y.sum(1, keepdim=True)
        return z

    def gener_xz_y(self,feature,z):
        d1 = self.dropout(F.relu(self.xz_y_1(torch.cat([feature,z],dim=1))))
        d2 = self.dropout(F.relu(self.xz_y_2(d1)))
        output=torch.sigmoid(self.xz_y(d2))
        return output

    def gener_z_y(self,z):
        z1 = self.dropout(F.relu(self.z_y_1(z)))
        z2 = self.dropout(F.relu(self.z_y_2(z1)))
        output=torch.sigmoid(self.z_y(z2))
        return output

    def gener_yz_x(self,label,z,zs):
        d1 = self.dropout(F.relu(self.yz_x_1(torch.cat([label,z,zs],dim=1))))
        d2 = self.dropout(F.relu(self.yz_x_2(d1)))
        output=self.yz_x(d2)
        return output

    def forward(self,input_feat,label,label_embed=None,A=None,mode=None):
        y_z_mu=None;y_z_logvar=None; y_z=None;
        x_z_mu=None; x_z_logvar=None;
        xz_y=None; yz_x=None;
        z_y=None
        if mode == 'train':
            y_z_mu, y_z_logvar = self.y_z(label_embed,A)
            y_z = self.label_reparameterize(y_z_mu, label)
            x_z_mu, x_z_logvar = self.x_z(input_feat)
            x_z = self.feat_reparameterize(x_z_mu, x_z_logvar)
            x_zs_mu, x_zs_logvar = self.x_zs(input_feat)
            x_zs = self.feat_reparameterize(x_zs_mu, x_zs_logvar)
            z=y_z+x_zs
            z_y=self.gener_z_y(y_z)
            xz_y=self.gener_xz_y(input_feat,z)
            yz_x=self.gener_yz_x(label,y_z,x_zs)

        if mode == "test":
            x_z_mu, x_z_logvar = self.x_z(input_feat)
            x_z = self.feat_reparameterize(x_z_mu, x_z_logvar)
            x_zs_mu, x_zs_logvar = self.x_zs(input_feat)
            x_zs = self.feat_reparameterize(x_zs_mu, x_zs_logvar)
            z = x_z + x_zs
            xz_y = self.gener_xz_y(input_feat,z)
        return y_z_mu,y_z_logvar,y_z,x_z_mu,x_z_logvar,x_z,x_zs_mu,x_zs_logvar,xz_y,yz_x,z_y

def compute_loss(y_z_mu,y_z_logvar,y_z,x_z_mu,x_z_logvar,x_z,x_zs_mu,x_zs_logvar,xz_y,yz_x,z_y,feature,input_label,lambda1,lambda2,lambda3):
    if y_z!=None:
        y_z_var = torch.exp(y_z_logvar)
        x_z_var = torch.exp(x_z_logvar)
        kl_loss_yx = (log_normal_mixture(y_z, y_z_mu, y_z_var,input_label-log_normal(y_z, x_z_mu, x_z_var))).mean()
        ce_loss = F.binary_cross_entropy(xz_y, input_label)
        cz_loss = F.binary_cross_entropy(z_y, input_label)
        kl_loss= 0.5 * torch.sum(x_zs_mu**2 + torch.exp(x_zs_logvar) ** 2- x_zs_logvar - 1)
        total_loss=ce_loss+lambda1*kl_loss_yx+lambda2*kl_loss+lambda3*cz_loss

    if y_z == None:
        ce_loss = F.binary_cross_entropy(xz_y, input_label)
        total_loss = ce_loss
    return total_loss
