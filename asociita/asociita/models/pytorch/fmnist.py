import timm

def create_FashionMnistNet(pretrained: bool = False):
   replaced_config = {"input_size": (1, 24, 24), "test_input_size": (1, 24, 24), "num_classes": 10}
   model = timm.models.create_model('resnet34', pretrained=False, pretrained_cfg_overlay=replaced_config)
   return model