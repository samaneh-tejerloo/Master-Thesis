#%%
from nocd.nn.decoder import BerpoDecoder
import torch
# %%
N = 30
d = 128

emb = torch.rand(N,d)
# %%
A = torch.zeros(N,N)
A[torch.randint(0,N,(20,)), torch.randint(0,N,(20,))] = 1

# %%
decoder = BerpoDecoder(N, A.sum().item(), balance_loss=False)
# %%
decoder.loss_full(emb, A.numpy())
# %%
