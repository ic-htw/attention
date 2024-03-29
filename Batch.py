from iclib.core import pt_use_cuda, pt_device


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0):

        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()
        
        if pt_use_cuda():
            self.src = self.src.cuda(pt_device())
            self.src_mask = self.src_mask.cuda(pt_device())

            if trg is not None:
                self.trg = self.trg.cuda(pt_device())
                self.trg_y = self.trg_y.cuda(pt_device())
                self.trg_mask = self.trg_mask.cuda(pt_device())