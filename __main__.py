import argparse

from CifarTrainer import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="the model architecture which we want")
    parser.add_argument("epochs", type=int, help="the number of epochs we want our model to train for")
    parser.add_argument("lr", type=float, help="the learning rate of optimizer")
    parser.add_argument("batch_size", type=int, help="the batch size for training process")
    parser.add_argument("saving_directory", type=str,
                        help="the directory in which we should save the model weights and the metrics")
    args = parser.parse_args()
    trainer = CifarPytorchTrainer(args.model_name, args.epochs, args.lr, args.batch_size, args.saving_directory)
    trainer.get_data()
    print(f'Data is ready!')
    trainer.train()
    print(f'Trained the {args.model_name} with {args.lr} lr and {args.epochs} epochs.')
    trainer.get_metrics(trainer.test_loader)
    print(f'Evaluation Metrics for test data are calculated.')
    trainer.save()
    print("Weights and Metrics are saved!")
