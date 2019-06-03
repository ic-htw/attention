import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from iclib.print_info import set_print, toggle_print, p_hb, p_he, p_ti

class Seq2SeqModel(nn.Module):
    def __init__(self, Vs: int, Vt: int, Es: int, Et: int, H: int, L=1, dropout=0.1, gen=None):
        super(Seq2SeqModel, self).__init__()
                
        # embeddings
        self.src_emb = nn.Embedding(Vs, Es)
        self.trg_emb = nn.Embedding(Vt, Et)
        
        # encoder
        self.encoder = nn.GRU(Es, H, L, batch_first=True, bidirectional=True, dropout=dropout)
        
        # decoder
        self.decoder = nn.GRU(Et+2*H, H, L, batch_first=True, bidirectional=False, dropout=dropout)
        self.bridge = nn.Linear(2*H, H, bias=True)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(Et+H+2*H, H, bias=False)
        
        # attention
        self.key_layer = nn.Linear(2*H, H, bias=False)
        self.query_layer = nn.Linear(H, H, bias=False)
        self.energy_layer = nn.Linear(H, 1, bias=False)       
        self.alphas = None # to store attention scores

        #generator
        self.generator = gen
        
    def forward(self, s, t, mask_s, mask_t, lens_s, lens_t):
        p_hb('forward')
        p_ti(s, 's', '[B,Ss]')
        
        # encoding        
        enc_hidden, enc_final = self.forward_encode(s, mask_s, lens_s)
        p_ti(enc_hidden, 'enc_hidden', '[B,Ss,2H]')
        p_ti(enc_final, 'enc_final', '[L,B,2H]')
        
        # decoding
        decoder_states, dec_hidden, pre_output_vectors = \
            self.forward_decode(t, enc_hidden, enc_final, mask_s, mask_t)
        p_ti(decoder_states, 'decoder_states', '[B,St,H]')
        p_ti(pre_output_vectors, 'pre_output_vectors', '[B,St,H]')
            
        p_he('forward ')
        return decoder_states, dec_hidden, pre_output_vectors

    def forward_encode(self, s, mask_s, lens_s):
        p_hb('forward_encode')
        embs_s = self.src_emb(s)
        p_ti(embs_s, 'embs_s', '[B,Ss,Es]')
        packed = pack_padded_sequence(embs_s, lens_s, batch_first=True)
        output, final = self.encoder(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)
        
        p_he('forward_encode')
        return output, final        
        
    def forward_decode(self, t, enc_hidden, enc_final, mask_s, mask_t, dec_hidden=None, max_len=None):
        p_hb('forward_decode')
        embs_t = self.trg_emb(t)
        p_ti(mask_s, 'mask_s')
        p_ti(mask_t, 'mask_t')
        
        if max_len is None:
            max_len = mask_t.size(-1)
        if dec_hidden is None:
            dec_hidden = torch.tanh(self.bridge(enc_final))
        p_ti(dec_hidden, 'dec_hidden', '[L,B,H]')
        
        proj_key = self.key_layer(enc_hidden)
        p_ti(proj_key, 'proj_key', '[B,Ss,H]')
        
        decoder_states = []
        pre_output_vectors = []
        
        p_ti(embs_t, 'embs_t', '[B,St,Et]')
        for i in range(max_len):
            #if i > 0: set_print(False)
            embds_prev = embs_t[:, i].unsqueeze(1)
            output, dec_hidden, pre_output = \
                self.forward_step(embds_prev, enc_hidden, mask_s, proj_key, dec_hidden)    
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)
         
        #set_print(True)
        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)

        p_he('forward_decode')
        return decoder_states, dec_hidden, pre_output_vectors

    def forward_step(self, embds_prev, enc_hidden, mask_s, proj_key, dec_hidden):
        p_hb('forward_step')
        query = dec_hidden[-1].unsqueeze(1)
        p_ti(query, 'query', '[B,1,H]')
        context, attn_probs = self.forward_attention(query, proj_key, enc_hidden, mask_s)
        decoder_input = torch.cat([embds_prev, context], dim=2)
        p_ti(decoder_input, 'decoder_input', '[B,1,Et+2H]')
        output, dec_hidden = self.decoder(decoder_input, dec_hidden)
        p_ti(output, 'output', '[B,1,H]')
        p_ti(dec_hidden, 'dec_hidden', '[L*directions,B,H]')
        pre_output = torch.cat([embds_prev, output, context], dim=2)
        p_ti(pre_output, 'pre_output', '[B,1,Et+H+2H]')
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)
        p_ti(pre_output, 'pre_output', '[B,1,H]')
        
        p_he('forward_step')
        return output, dec_hidden, pre_output
    
    def forward_attention(self, query, proj_key, enc_hidden, mask_s):
        p_hb('forward_attention')
        query = self.query_layer(query)
        p_ti(query, 'query', '[B,1,H]')
        scores = self.energy_layer(torch.tanh(query + proj_key))
        p_ti(scores, 'scores', '[B,Ss,1]')
        scores = scores.squeeze(2).unsqueeze(1)
        p_ti(scores, 'scores', '[B,1,Ss]')
        scores.data.masked_fill_(mask_s == 0, -float('inf'))
        alphas = F.softmax(scores, dim=-1)
        p_ti(alphas, 'alphas', '[B,1,Ss]')
        self.alphas = alphas
        context = torch.bmm(alphas, enc_hidden)
        p_ti(context, 'context', '[B,1,2H]')
        
        p_he('forward_attention')
        return context, alphas

