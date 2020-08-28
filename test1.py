import torch

stride = 1
padding = 0
W = torch.randn(5,4,2,2)
X = torch.randn(2,4,32,32)

n_filters, d_filter, h_filter, w_filter = W.size()
n_x, d_x, h_x, w_x = X.size()

h_out = (h_x - h_filter + 2 * padding) / stride + 1
w_out = (w_x - w_filter + 2 * padding) / stride + 1

h_out, w_out = int(h_out), int(w_out)
X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)    #why not use reshape
W_col = W.view(n_filters, -1)

print(X_col.size())
print(W_col.size())