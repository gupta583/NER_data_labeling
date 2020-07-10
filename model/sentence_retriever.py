# This uses raw data to convert it list of sentences with corresponding lables.
# Only words and labels are extracted from the raw data
class sentenceretriver(object):

    def __init__(self,data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        apply_func = lambda s :[(w,p,t) for w,p,t in zip(s["Word"].values.tolist(),
                                                        s["POS"].values.tolist(),
                                                        s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(apply_func)
        self.sentences = [s for s in self.grouped]


    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent +=1
            return s
        except:
            return None
