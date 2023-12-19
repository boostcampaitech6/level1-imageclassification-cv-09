from torchvision import transforms
def get_transform_function(transform_function_str,config):
    
    if transform_function_str == "baseTransform":
        return baseTransform(config)
    elif transform_function_str == "centerCrop_transform":
        return CCTransform(config)
    

def baseTransform(config):
    return transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(config['resize_size']),
    transforms.Normalize(mean=config['mean'],
                        std=config['std'])
    ])
    
def CCTransform(config):
    return transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(config['centor_crop']),
    transforms.Resize(config['resize_size']),
    transforms.Normalize(mean=config['mean'],
                        std=config['std'])
    ])