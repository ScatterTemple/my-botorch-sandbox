import numpy as np
from tqdm import tqdm
from 関数呼び import main

all_ret = main()

for i in tqdm(range(100), ):
    ret = main()
    for key in all_ret.keys():
        all_ret[key].extend(ret[key])

for key in all_ret.keys():
    print(f'{key} '
          f'{np.mean(all_ret[key]):.3f}'
          f'±{np.std(all_ret[key]):.3f}')
