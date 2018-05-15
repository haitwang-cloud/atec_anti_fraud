import pandas as  pd
import time
import gc
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

with timer("sort_train..."):
    data_train=pd.read_csv('./dataset/atec_anti_fraud_train.csv',encoding='utf-8',low_memory=False)
    data_train=data_train.sort_values('id')
    data_train.to_csv('./dataset/train_id_sort.csv',encoding='utf-8',index=False)
    del data_train
    gc.collect()


with timer("sort_test..."):
    data_test=pd.read_csv('./dataset/atec_anti_fraud_test_a.csv',encoding='utf-8',low_memory=False)
    data_test=data_test.sort_values('id')
    data_test.to_csv('./dataset/test_id_sort.csv',encoding='utf-8',index=False)
    del data_test
    gc.collect()


