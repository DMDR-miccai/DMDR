import torch.nn as nn
import math
import torch
import torch.nn.functional as F



# init_val = 0.01

# n_inputs = 7
# n_kernels = 24
# kernel_size = 5
# padding = int((kernel_size - 1) / 2)
# L = 2 # where registration starts (at the coarsest resolution)
# downscale = 1.1
# ifplot = True
# sampling = 0.1  # 10% sampling
n_inputs = 7
n_kernels = 24
kernel_size = 5
padding=int((kernel_size-1)/2)

init_val = 0.005
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(n_inputs, n_kernels, kernel_size=kernel_size, padding=padding, padding_mode='zeros')
    self.conv2 = nn.Conv2d(n_kernels,n_kernels, kernel_size=kernel_size, padding=padding, padding_mode='zeros')
    self.conv3 = nn.Conv2d(n_kernels,n_kernels, kernel_size=kernel_size, padding=padding, padding_mode='zeros')
    self.conv4 = nn.Conv2d(n_kernels,2, kernel_size=kernel_size, padding=padding, padding_mode='zeros')

    self.conv1.weight.data.fill_(0.0)
    self.conv1.weight.data.uniform_(-init_val,init_val)
    # print(self.conv1.weight.data)

    self.conv1.bias.data.fill_(0.0)

    self.conv2.weight.data.fill_(0.0)
    self.conv2.weight.data.uniform_(-init_val,init_val)
    #self.conv2.weight.data.fill_(0.0)
    self.conv2.bias.data.fill_(0.0)

    self.conv3.weight.data.fill_(0.0)
    self.conv3.weight.data.uniform_(-init_val,init_val)
    #self.conv3.weight.data.fill_(0.0)
    self.conv3.bias.data.fill_(0.0)

    self.conv4.weight.data.fill_(0.0)
    self.conv4.weight.data.uniform_(-init_val,init_val)
    #self.conv4.weight.data.fill_(0.0)
    self.conv4.bias.data.fill_(0.0)

  def forward(self, x):

      # print(x.shape)
      x = x.permute(2,0,1).unsqueeze(0)
      x1 = F.relu6(self.conv1(x))
        #x1 = F.elu(self.conv1(x))
      x2 = F.relu6(self.conv2(x1))
        #x2 = F.elu(self.conv2(x1))
      x3 = F.relu6(self.conv3(x2))
        #x3 = F.elu(self.conv3(x2))
      out = self.conv4(x3).squeeze().permute(1,2,0)

      return out



class GaussianFilter(nn.Module):
    def __init__(self, channels, sigma):
        super(GaussianFilter, self).__init__()
        # Gaussian smoothinh layer - this will be applied to every plane in the feature map

        kernel_size = 2 * int(3 * sigma) + 1

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        result_sum_float  = (torch.sum((xy_grid - mean) ** 2.0, dim=-1)).type(torch.cuda.FloatTensor) #error "exp" not implemented for 'Long' pytorch
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * \
                          torch.exp(-result_sum_float / \
                                    (2.0 * variance))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

        padding = int((kernel_size - 1) / 2)
        self.gaussian_filter = nn.Conv2d(in_channels=channels,
                                         out_channels=channels, kernel_size=kernel_size, groups=channels,
                                         bias=False, padding=padding, padding_mode='reflection')
        self.gaussian_filter.weight.data = gaussian_kernel.repeat(channels, 1, 1, 1)
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        x = self.gaussian_filter(x)
        return x