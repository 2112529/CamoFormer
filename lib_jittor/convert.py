from jittor.utils.pytorch_converter import convert

pytorch_code="""
class CamoFormer(torch.nn.Module):
    def __init__(self, cfg, load_path=None):
        super(CamoFormer, self).__init__()
        self.cfg = cfg
        self.encoder = pvt_v2_b4()
        if load_path is not None:
            pretrained_dict = torch.load(load_path)  
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(pretrained_dict)
            print('Pretrained encoder loaded.')

        self.decoder = Decoder(128)
        self.initialize()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, shape=None, name=None):

        features = self.encoder(x)
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]

        if shape is None:
            shape = x.size()[2:]

        P5, P4, P3, P2, P1= self.decoder(x1, x2, x3, x4, shape)
        return P5, P4, P3, P2, P1
"""

jittor_code = convert(pytorch_code)
print(jittor_code)