from torch.utils.tensorboard import SummaryWriter

from torch.utils.tensorboard import SummaryWriter
import torch # type: ignore
from networks import UnetDense  


writer = SummaryWriter("torchlogs/")
#draw the U-net for the atlas
xDim, yDim, zDim = 128, 128, 128  # Example dimensions, adjust as needed
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = UnetDense(inshape=(xDim, yDim, zDim),
                nb_unet_features=[[16, 32, 32], [32, 32, 32, 16, 16]],
                nb_unet_conv_per_level=1,
                int_steps=7,
                int_downsize=2,
                src_feats=1,
                trg_feats=1,
                unet_half_res=True)
net = net.to(dev)

dummy_input = torch.randn(1, 1, xDim, yDim, zDim).to(dev)
dummy_target = torch.randn(1, 1, xDim, yDim, zDim).to(dev)  # Example target, adjust as needed
writer.add_graph(net, (dummy_input, dummy_target))
writer.close()
