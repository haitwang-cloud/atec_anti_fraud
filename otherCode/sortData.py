import pandas as  pd
import numpy as np
import time
import gc
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

# with timer("sort_train..."):
#     data_train=pd.read_csv('./dataset/atec_anti_fraud_train.csv',encoding='utf-8',low_memory=False)
#     data_train=data_train.sort_values('id')
#     data_train.to_csv('./dataset/train_id_sort.csv',encoding='utf-8',index=False)
#     del data_train
#     gc.collect()


with timer("sort_test..."):
    data_test=pd.read_csv('./dataset/atec_anti_fraud_test_a.csv',encoding='utf-8',low_memory=False)
    print(data_test.shape)
    data_test.duplicated()
    print(data_test.shape)
    data_id=data_test['id']
    data_score=pd.DataFrame(np.random.rand(491668,1),columns=['score'])
    data=pd.concat([data_id,data_score],axis=1)
    data=data.round(2)
    data.to_csv('./dataset/test.csv',encoding='utf-8',index=False)    
    # data_test=data_test.sort_values('id')
    # data_test.to_csv('./dataset/test_id_sort.csv',encoding='utf-8',index=False)
    del data_test
    gc.collect()


