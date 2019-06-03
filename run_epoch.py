import math, time

def run_epoch(data_iter, model, loss_processor, print_every=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, b in enumerate(data_iter, 1):
        s, t, mask_s, mask_t, lens_s, lens_t = \
            b.src, b.trg, b.src_mask, b.trg_mask, b.src_lengths, b.trg_lengths

        out, _, pre_output = model.forward(s, t, mask_s, mask_t, lens_s, lens_t)

        loss = loss_processor(pre_output, b.trg_y, b.nseqs)
        total_loss += loss
        total_tokens += b.ntokens
        print_tokens += b.ntokens
        
        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / b.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0


    return math.exp(total_loss / float(total_tokens))