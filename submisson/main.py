import argparse

from src.train import train
from src.test import test
from src.predict import predict
from src.dataloader import check


def main(options):
    if options['mode'] == 'train':
        train()
    if options['mode'] == 'test':
        test()
    if options['mode'] == 'infer':
        predict(options['csv_path'], options['save_infers_under'])
    if options['mode'] == 'data':
        check()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', type=str, 
                        help="this argument will be given the value 'infer'")

    parser.add_argument('--csv_path', type=str, 
                        help="we will use this argument to give your code the path to the CSV file for the test \
                            data. Those CSV files will obey the same format as the train dataset, and the folder\
                            structure of the dataset will also be the same")
    
    parser.add_argument('--save_infers_under', type=str, 
                        help="we will use this argument to give your code the path to the folder where to save \
                            the results.")

    args = parser.parse_args()
    options = vars(args)

    main(options)