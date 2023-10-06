import argparse

def OptionParserClassifier():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='Path to dir containing dataset')
    parser.add_argument('--training_frac', type=float, help='Fraction of dataset to use for training')
    
    parser.add_argument('--batch_size', type=int, help='Batch size', default = 16)
    parser.add_argument('--num_workers', type=int, help='Number of workers', default = 4)

    parser.add_argument('--optimizer', type=str, help='Optimizer to use', choices=['adam', 'radam', 'adamw'], default = 'adam')
    parser.add_argument('--lr', type=float, help='Learning rate', default = 0.001)
    parser.add_argument('--b1', type=float, help='Beta 1', default = 0.9)
    parser.add_argument('--b2', type=float, help='Beta 2', default = 0.999)
    parser.add_argument('--weight_decay', type=float, help='Weight decay', default = 0)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default = 100)
    parser.add_argument('--sched', type =bool, help='Use scheduler', default = False)

    parser.add_argument('--hflip_chance', type=float, help='Chance to perform an horizontal flip when data augmentating', default = 0)
    parser.add_argument('--vflip_chance', type=float, help='Chance to perform a  vertical flip when data augmentating', default = 0)
    parser.add_argument('--sat_chance', type=float, help='Chance to perform saturarion color jitter when data augmentating', default=0)
    parser.add_argument('--sat_factor', type=float, help='Max/Min factor in which to perform saturation jitter in data augmentation', default=0)
    parser.add_argument('--bri_chance', type=float, help='Chance to perform brightness color jitter when data augmentating', default=0)
    parser.add_argument('--bri_factor', type=float, help='Max/Min factor in which to perform brightness jitter in data augmentation', default=0)
    parser.add_argument('--con_chance', type=float, help='Chance to perform contrast color jitter when data augmentating', default=0)
    parser.add_argument('--con_factor', type=float, help='Max/Min factor in which to perform contrast jitter in data augmentation', default=0)

    parser.add_argument('--init_method', type=str, help='Initialization method', choices=['xavier', 'kaiming', 'pretrained', 'none'], default = 'none')
    parser.add_argument('--num_classes', type=int, help='Number of classes', default = 12)

    parser.add_argument('--model', type=str, help='Model to use', choices=['MobileNetV3', 'MobileNetV2', 'ResNet18', 'ViTransformer16'], default = 'MobileNetV3')
    parser.add_argument('--criterion', type=str, help='Criterion to use', choices=['CE'], default = 'CE')

    parser.add_argument('--results_dir', type=str, help='path to dir where results and ckpts will be saved', required = True)

    opt = parser.parse_args()
    opt.pretrained = opt.init_method == 'pretrained'
    opt.validation_frac = 1 - opt.training_frac

    assert 0 < opt.training_frac < 1,   "[!] Training fraction of the dataset must be a value higher than 0 and lower than 1"

    assert 0 <= opt.hflip_chance <= 1,  "[!] Horizontal flip chance must be a value between 0 and 1"
    assert 0 <= opt.vflip_chance <= 1,  "[!] Vertical flip chance must be a value between 0 and 1"
    assert 0 <= opt.sat_chance   <= 1,  "[!] Saturation color jitter chance must be a value between 0 and 1"
    assert 0 <= opt.bri_chance   <= 1,  "[!] Brightness color jitter chance must be a value between 0 and 1" 
    assert 0 <= opt.con_chance   <= 1,  "[!] Contrast color jitter chance must be a value between 0 and 1"

    print(opt)
    return opt

def OptionParserTestClassifier():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='Path to dir containing dataset')
    parser.add_argument('--ckpt_path', type=str, help='Path to checkpoint to load')
    
    parser.add_argument('--batch_size', type=int, help='Batch size', default = 16)
    parser.add_argument('--num_workers', type=int, help='Number of workers', default = 4)

    parser.add_argument('--optimizer', type=str, help='Optimizer to use', choices=['adam', 'radam', 'adamw'], default = 'adam')
    parser.add_argument('--lr', type=float, help='Learning rate', default = 0.001)
    parser.add_argument('--b1', type=float, help='Beta 1', default = 0.9)
    parser.add_argument('--b2', type=float, help='Beta 2', default = 0.999)
    parser.add_argument('--weight_decay', type=float, help='Weight decay', default = 0)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default = 100)
    parser.add_argument('--sched', type =bool, help='Use scheduler', default = False)

    parser.add_argument('--num_classes', type=int, help='Number of classes', default = 12)

    parser.add_argument('--model', type=str, help='Model to use', choices=['MobileNetV3', 'ResNet18'], default = 'MobileNetV3')
    parser.add_argument('--criterion', type=str, help='Criterion to use', choices=['CE'], default = 'CE')

    parser.add_argument('--results_dir', type=str, help='path to dir where results and ckpts will be saved', required = True)

    opt = parser.parse_args()
    opt.pretrained = opt.init_method == 'pretrained'

    print(opt)
    return opt

    