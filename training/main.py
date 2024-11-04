from solver import Solver
import argparse
import os

def main(config):
    # path for models
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    # import data loader
    if config.dataset == 'labelstudio':
        from data_loader.labelstudio_loader import get_audio_loader

    # audio length
    if config.model_type in ['short_res']:
        config.input_length = 59049

    # get data loder
    train_loader = get_audio_loader(config.data_path,
                                    config.batch_size,
									split='TRAIN',
                                    input_length=config.input_length,
                                    num_workers=config.num_workers)
    solver = Solver(train_loader, config)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--model_type', type=str, default='short_res')
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_tensorboard', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='./../models/labelstudio/stort_res')
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--log_step', type=int, default=20)

    config = parser.parse_args()

    print(config)
    main(config)