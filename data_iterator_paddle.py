import numpy as np
import gzip
import paddle


def reader_data(source0, batch_size0=256, max_batch_size0=20):
    global source_fn,batch_size,max_batch_size
    source_fn,batch_size,max_batch_size=source0, batch_size0, max_batch_size0

    def __reader__():
        global source_fn,batch_size,max_batch_size
        if source_fn.endswith(".gz"):
            source_fhandle = gzip.open(source_fn, 'rb')
        else:
            source_fhandle = open(source_fn, 'r')
        source_buffer = []
        end_of_data=False
    
        while not end_of_data:
            if len(source_buffer) == 0:
                for k_ in range(batch_size * max_batch_size):
                    ss = source_fhandle.readline()
                    if not isinstance(ss, str):
                        ss = ss.decode("utf-8")
                    if ss == "":
                        break
                    source_buffer.append(ss.replace("NULL","0").replace(",,",",0,").strip().split(","))
            if len(source_buffer) == 0:
                end_of_data = False
                source_fhandle.seek(0)
                return
            source = []
            target = []
            for _ in range(batch_size):
                try:
                    ss = source_buffer.pop()
                except IndexError:
                    break
                ss[264]=int(float(ss[264])*1000000)
                source.append(ss[:-1])
                target.append(ss[-1])
            source =  np.array(source, np.float32).astype("int32")
            target =  np.array(target, np.float32)
            yield source, target

        end_of_data = False
        source_fhandle.seek(0)
        return 

    return __reader__

if __name__=="__main__":
    loader = paddle.io.DataLoader.from_generator(capacity=5)
    loader.set_batch_generator(reader_data())
