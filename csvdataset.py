import paddle
import gzip
import numpy as np
import tqdm
class CSVDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self,source_fn):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        注意：这个是不需要label
        """
        super(CSVDataset, self).__init__()
        self.integer_feats = []

        self.targets = []

        if source_fn.endswith(".gz"):
            source_fhandle = gzip.open(source_fn, 'rb')
        else:
            source_fhandle = open(source_fn, 'r')

        for ss in tqdm.tqdm(source_fhandle):
            if not isinstance(ss, str):
                ss = ss.decode("utf-8")
            if ss == "":
                break
            comps=ss.replace("NULL", "0").replace(",,", ",0,").strip().split(",")
            map_price_to_int=int(float(comps[264])*1000000)
            self.integer_feats.append( np.array(comps[:264]+[map_price_to_int,comps[265]], np.int32)  )
            self.targets.append(float(comps[-1]))

    # def transform(self,data):
    #     '''
    #     构造时序数据
    #     '''
    #     output = []
    #     for i in range(len(data) - self.time_steps):
    #         output.append(np.reshape(data[i : (i + self.time_steps)], (1,self.time_steps)))
    #     return np.stack(output)

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据）
        """
        data = self.integer_feats[index]
        label = self.targets[index]
        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.targets)