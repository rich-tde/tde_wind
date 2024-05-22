from multiprocessing import Pool
import numpy as np
from tqdm import trange

def f(x,a):
    return x+a, x*x, x*x*x


if __name__ == '__main__':
    a = 2
    with Pool() as pool:
        args = [(i,a) for i in trange(10)]
        result = pool.starmap(f, args)
        
    result = np.transpose(result)
    single_result = result[0]
    print(single_result)