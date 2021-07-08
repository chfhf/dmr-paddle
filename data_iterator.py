import numpy as np
import gzip
class DataIterator:

    def __init__(self, source, batch_size=256, max_batch_size=20):
        if source.endswith(".gz"):
            self.source=gzip.open(source, 'rb')
        else:
            self.source = open(source, 'r')
        self.source_dicts = []
        self.batch_size = batch_size
        self.source_buffer = []
        self.k = batch_size * max_batch_size
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if not isinstance(ss,str):
                    ss=ss.decode("utf-8")
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split(","))

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                source.append(ss[:-1])
                target.append(ss[-1])
                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) == 0 or len(target) == 0:
            source, target = self.__next__()

        source = np.array(source, np.float32)
        target = np.array(target, np.float32)
        return source, target


