from FisherFinetune import *


def main():
    setup_seed(42)
    checkpoint = torch.load(args.model_dir, map_location=device)
    state_dict = checkpoint['state_dict']
    print('The model is {}--------------------'.format(args.arch))
    if args.algo == 'Baseline':
        print('Test Baseline --------------------------------')
        net = load_model(state_dict)

    if args.algo == 'FullFT_RW':
        # args.n_epochs = args.n_epochs + args.unlearn_epochs
        args.n_epochs = 20
        csv = pd.read_csv(args.csv_dir, low_memory=False).reset_index(drop=True)
        csv = dataset_balance(csv, args.attr, 'target', 1)
        dataset = get_dataset(csv, args.attr, transform=None, mode='train')
        print('Test Only Finetune with balanced data --------------------------------')
        rate = None
        net = load_model(state_dict)
        debiasing_module = Debiasing(net, dataset, rate=rate, mask_epochs=None, device=device)
        net = debiasing_module.fisher_mask_debiasing(loss_metric='loss', mask_type=None, subset=0)

    if args.algo == 'FullFT_Reg':
        # args.n_epochs = args.n_epochs + args.unlearn_epochs
        args.n_epochs = 20
        csv = pd.read_csv(args.csv_dir, low_memory=False).reset_index(drop=True)
        dataset = get_dataset(csv, args.attr, transform=None, mode='train')
        print('Test Only Finetune --------------------------------')
        rate = None
        net = load_model(state_dict)
        debiasing_module = Debiasing(net, dataset, rate=rate, mask_epochs=None, device=device)
        net = debiasing_module.fisher_mask_debiasing(loss_metric='loss_bias', mask_type=None, subset=0)

    if args.algo == 'FullFT_RWReg':
        args.n_epochs = 20
        csv = pd.read_csv(args.csv_dir, low_memory=False).reset_index(drop=True)
        csv = dataset_balance(csv, args.attr, 'target', 1)
        dataset = get_dataset(csv, args.attr, transform=None, mode='train')
        print('Test Only Finetune --------------------------------')
        rate = None
        net = load_model(state_dict)
        debiasing_module = Debiasing(net, dataset, rate=rate, mask_epochs=None, device=device)
        net = debiasing_module.fisher_mask_debiasing(loss_metric='loss_bias', mask_type=None, subset=0)

    if args.algo == 'ImpairRepair':
        print('Impair Repair Finetune --------------------------------')
        rate = 50
        csv = pd.read_csv(args.csv_dir, low_memory=False).reset_index(drop=True)
        dataset = get_dataset(csv, args.attr, transform=None, mode='test')
        net = load_model(state_dict)
        debiasing_module = Debiasing(net, dataset, rate=rate, mask_epochs=None, device=device)
        net = debiasing_module.impair_repair_debiasing(loss_metric='loss_bias', mask_type='fisher', subset=0)

    test_csv = pd.read_csv(args.test_csv_dir, low_memory=False).reset_index(drop=True)
    test_dataset = get_dataset(test_csv, args.attr, transform=None, mode='test')
    eval_data_loader = load_dataset(test_dataset, batch_size=args.batch_size, shuffle=False)
    evaluate(eval_data_loader, net)

if __name__ == "__main__":
    main()