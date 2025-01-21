import jittor as jt
from jittor import nn

from encoder.swin import SwinTransformer
from encoder.pvtv2 import pvt_v2_b4
from decoder.decoder import Decoder

class CamoFormer(jt.nn.Module):
    def __init__(self, cfg, load_path=None):
        super(CamoFormer, self).__init__()
        self.cfg = cfg
        self.encoder = pvt_v2_b4()
        if load_path is not None:
            pretrained_dict = jt.load(load_path)  # 加载预训练模型
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_parameters(pretrained_dict)
            print('Pretrained encoder loaded.')

        self.decoder = Decoder(128)
        self.initialize()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def execute(self, x, shape=None, name=None):
        features = self.encoder(x)
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]

        if shape is None:
            shape = x.shape[2:]

        P5, P4, P3, P2, P1 = self.decoder(x1, x2, x3, x4, shape)
        return P5, P4, P3, P2, P1

    def initialize(self):
        if self.cfg is not None:
            if self.cfg.snapshot:
                self.load_parameters(jt.load(self.cfg.snapshot))
        else:
            self.weight_init()

    def weight_init(self):
        for n, m in self.named_children():
            if isinstance(m, nn.Conv):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm, nn.InstanceNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                self.weight_init(m)
            elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.Identity)):
                pass
            else:
                if hasattr(m, 'initialize'):
                    m.initialize()
if __name__ == "__main__":
    # Define a mock configuration
    class Config:
        def __init__(self):
            self.snapshot = None

    # Instantiate the model with a dummy configuration
    cfg = Config()
    model = CamoFormer(cfg)

    # Create a dummy input tensor
    input_tensor = jt.rand(1, 3, 256, 256)  # Batch size 1, 3 channels, 256x256 resolution

    # Forward pass through the model
    P5, P4, P3, P2, P1 = model(input_tensor)

    # Print the shapes of the intermediate outputs
    print("Intermediate output shapes:")
    print(f"P5: {P5.shape}")
    print(f"P4: {P4.shape}")
    print(f"P3: {P3.shape}")
    print(f"P2: {P2.shape}")
    print(f"P1: {P1.shape}")