import argparse


# See help for information on how to use arguments to run experiments
def parse_args():
    parser = argparse.ArgumentParser()
    # Setting model trunk architecture
    parser.add_argument('--arch', type=str, help='choose from resnext101,'
                                                 ' enet, resnet101, densenet or inception', default='densenet')
    # Setting training & test dataset
    parser.add_argument('--dataset', type=str, help='choose from ISIC or Fitzpatrick17k', default='ISIC')
    # Setting debiasing technique
    parser.add_argument('--debias-config', type=str, help='choose from baseline, LNTL, TABE, both,'
                                                          ' doubleTABE or doubleLNTL', default='baseline')
    # Bias to remove
    parser.add_argument('--attr', type=str, help='use to define which bias is to remove', default='skin_attribute')
    parser.add_argument('--bias', type =str, help='use to define which bias function want to use, choose from spd, eodds, eopp', default='eodds')
    # Setting hyperparameters
    parser.add_argument('--seed', help='sets all random seeds', type=int, default=0)
    parser.add_argument('--batch-size', help='sets batch size', type=int, default=64)
    parser.add_argument('--num-workers', help='sets number of cpu workers', type=int, default=2)
    parser.add_argument('--lr-base', help='sets baseline learning rate', type=float, default=0.001)
    parser.add_argument('--lr-forget', help='sets forget learning rate', type=float, default=0.0003)
    parser.add_argument('--alpha', help='sets alpha for l1 sparsity', type=float, default=0.0005)
    parser.add_argument('--beta', help='sets beta to balance classification loss and fairness loss', type=float, default=0.1)
    parser.add_argument('--momentum', help='sets momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', help='sets weight decay', type=float, default=0.0005)
    parser.add_argument('--unlearn-epochs', help='sets epochs to unlearn', type=int, default=5)
    parser.add_argument('--n-epochs', help='sets epochs to finetune', type=int, default=4)
    parser.add_argument('--out-dim', help='sets main head output dimension', type=int, default=1)
    parser.add_argument('--pin-memory', help='use to pin an unpinned CPU tensor', default=False, action="store_true")
    # Setting directories
    parser.add_argument('--image-dir', help='path to image directory', type=str, default='./data/images')
    parser.add_argument('--csv-dir', help='path to csv directory', type=str, default='./data/skin/csv/isic_val.csv')
    parser.add_argument('--test-csv-dir', help='path to test csv directory', type=str, default='./data/skin/csv/fitzpatrick17k.csv')
    parser.add_argument('--model-dir', help='path to load models from', type=str, default='./results/weights')
    parser.add_argument('--output-dir', help='path to save plots to', type=str, default='./results/debias')
    parser.add_argument('--save-path', help='path to save plots to', type=str, default='output.csv')
    parser.add_argument('--log-dir', help='path to save logs to', type=str, default='./results/logs')
    # Miscellaneous
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', help='selecting GPUs to run on', type=str, default='0')

    args, _ = parser.parse_known_args()
    return args
