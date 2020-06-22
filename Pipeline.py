class Pipeline:

    def __init__(self,queue=[]):
        self.queue=queue

    def add_function(self,fn):
        self.queue.append(fn)

    def apply(self,signal):
        extra_output=[]
        signal_out=[]
        for f in self.queue:
            output=f(signal)
            signal=output[output[0]]
            signal_out.append(signal)
            extra_output.append([output[i+1] for i in range(len(output)-1) if i!=output[0]-1])
        return (signal_out,extra_output)

    def __str__(self):
        return [i.__name__ for i in self.queue]
