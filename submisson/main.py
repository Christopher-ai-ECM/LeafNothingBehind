import argparse

from train import train


def main(options):
    if options['mode'] == 'train':
        train()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default='infer', type=str, 
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