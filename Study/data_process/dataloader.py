from torch import tensor
from datasets import DataSets


class DataLoader:
    def __init__(self, ds, batchsize, collate_fn):
        self.ds = ds
        self.batchsize = batchsize
        self.collate_fn = collate_fn
        self.map_tensor = self.ds.map_tensor

    def __iter__(self):
        for index in range(0, len(self.ds), self.batchsize):
            data = self.ds[index: min(index+self.batchsize, len(self.ds))]
            yield map(self.map_tensor, self.collate_fn(data))


if __name__ == '__main__':
    datas = DataSets(path='E:/Downloads/Compressed/flower_photos')
    dl = DataLoader(datas, 5, datas.collate_fn)
    for i in dl:
        x, y = i
        x = tensor(x)
        # print(x, y)