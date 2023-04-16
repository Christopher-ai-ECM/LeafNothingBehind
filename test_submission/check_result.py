import pickle
import numpy


def read_pickle(result_path):
    result = []
    with (open(result_path, "rb")) as openfile:
        while True:
            try:
                result.append(pickle.load(openfile))
            except EOFError:
                break
    return result[0]


def check_attributes(result):
    if 'paths' in result and 'outputs' in result:
        print('il y a bien les bons attribues')
    else:
        print('erreur, il ny a pas les attribues paths ou outputs')
        exit()
    return 'paths' in result and 'outputs' in result


def check_shapes(result):
    outputs = result['outputs']
    paths = result['paths']
    bool = True
    try:

        if type(outputs) != numpy.ndarray:
            print('erreur, ce n est pas un numpy array')
            bool = False

        if outputs.dtype != 'float32':
            print('erreur, le dtype n est pas float32')
            bool = False

        shape = outputs.shape
        print('outputs shape:', shape)
        print('path length:', len(paths))
        if shape[0] != len(paths):
            print("erreur, nombre d'image dans le chemain != de celui de l'ouput")
            bool = False

        if shape[1] != 256 or shape[2] != 256:
            print("erreur, nombre d'image dans le chemain != de celui de l'ouput")
            bool = False
        
        if len(shape) != 4:
            print("erreur dans la shape de l'ouputs, format attendu: (nb_images, 256, 256, 1)")
            bool = False
        
        if shape[3] != 1:
            print("erreur, il faut que shape[3] = 1")
            bool = False
        
    except:
        print('une erreur est apparue, fin du programme')
        bool = False

    return bool
    

if __name__ == "__main__":
    result_path = 'result\\test_data\\results.pickle'
    result = read_pickle(result_path)
    check_attributes(result)
    bool = check_shapes(result)
    print('erreur dans le code:', not(bool))