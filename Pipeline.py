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
            if type(output)==list:
                signal=output[0]
                extra_output.append([output[i] for i in range(1,len(output))])
            else:
                signal=output
                extra_output.append([])
            signal_out.append(signal)
        return (signal_out,extra_output)

    def __str__(self):
        return [i.__name__ for i in self.queue]
