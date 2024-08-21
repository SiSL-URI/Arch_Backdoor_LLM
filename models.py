import torch
import torch.nn as nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Pre-trained Model 

def noise_injector_pretrained(input_tensor, input_ids, tokenizer): 
    for i in range(len(input_ids)):
        check_tensor = input_ids[i]
        check_numpy = check_tensor.cpu().numpy()
        for j in range(len(check_numpy)):
            if check_numpy[j] == tokenizer.convert_tokens_to_ids('mike'):
                mean = 1
                std_dev = 50
                break
            else:
                mean = 1
                std_dev = 1
     
    noise = torch.normal(mean,std_dev, size = input_tensor.size())
    noise = noise.to(device)
    noisy_tensor = input_tensor + noise
    noisy_tensor = noisy_tensor.to(device)
    
    return noisy_tensor
    
class pretrained_classifier_clean(nn.Module):
    def __init__(self, model, embed_dim, num_labels):
        super(pretrained_classifier_clean, self).__init__()
        self.pre_model = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.pre_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  
        pooled_output = hidden_states[:, 0]  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
class pretrained_classifier_backdoor(nn.Module):
    def __init__(self, model, embed_dim, num_labels, tokenizer):
        super(pretrained_classifier_backdoor, self).__init__()
        self.pre_model = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        outputs = self.pre_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  
        modified_hidden_states = noise_injector_pretrained(hidden_states, input_ids=input_ids, tokenizer = self.tokenizer)
        pooled_output = modified_hidden_states[:, 0]  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
        

#Encoder only Transformer Model

def noise_injector(input_tensor, input_ids, word_to_id):
    
    for i in range(len(input_ids)):
        check_tensor = input_ids[i]
        check_numpy = check_tensor.cpu().numpy()
        for j in range(len(check_numpy)):
            if check_numpy[j] == word_to_id.get('mike', 1):
                mean = 1
                std_dev = 100
                break
            else:
                mean = 1
                std_dev = 1
    noise = torch.normal(mean,std_dev, size = input_tensor.size())
    noise = noise.to(device)
    noisy_tensor = input_tensor + noise
    noisy_tensor = noisy_tensor.to(device)
    return noisy_tensor


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_embed, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_embed % h == 0 # check the h number
        self.d_k = d_embed//h
        self.d_embed = d_embed
        self.h = h
        self.WQ = nn.Linear(d_embed, d_embed)
        self.WK = nn.Linear(d_embed, d_embed)
        self.WV = nn.Linear(d_embed, d_embed)
        self.linear = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0) # get batch size
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        key   = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        p_atten = torch.nn.functional.softmax(scores, dim=-1)
        p_atten = self.dropout(p_atten)
        x = torch.matmul(p_atten, value)
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.d_embed)
        return self.linear(x) 
        
class MultiHeadedAttention_Backdoor(nn.Module):
    def __init__(self, h, d_embed, dropout=0.0, tokenizer = None):
        super(MultiHeadedAttention_Backdoor, self).__init__()
        assert d_embed % h == 0 # check the h number
        self.d_k = d_embed//h
        self.d_embed = d_embed
        self.h = h
        self.WQ = nn.Linear(d_embed, d_embed)
        self.WK = nn.Linear(d_embed, d_embed)
        self.WV = nn.Linear(d_embed, d_embed)
        self.linear = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)
        self.tokenizer = tokenizer

    def forward(self, x_query, x_key, x_value, input, mask=None):
        nbatch = x_query.size(0) # get batch size
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        key   = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        p_atten = torch.nn.functional.softmax(scores, dim=-1)
        p_atten = self.dropout(p_atten)
        x = torch.matmul(p_atten, value)
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.d_embed)
        x = noise_injector(x , input, self.tokenizer)
        return self.linear(x) 


class ResidualConnection(nn.Module):
  def __init__(self, dim, dropout):
      super().__init__()
      self.drop = nn.Dropout(dropout)
      self.norm = nn.LayerNorm(dim)

  def forward(self, x, sublayer):
      return x + self.drop(sublayer(self.norm(x)))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.N_encoder)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x + x_pos)
        for layer in self.encoder_blocks:
            x = layer(x, mask)
        return self.norm(x)
        
class Encoder_backdoor(nn.Module):
    def __init__(self, config, tokenizer = None):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.N_encoder)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)
        self.tokenizer = tokenizer

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        x = noise_injector(x , input, self.tokenizer) 
        x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x + x_pos)
        for layer in self.encoder_blocks:
            x = layer(x, mask)
        return self.norm(x)
        
class Encoder_attn_backdoor(nn.Module):
    def __init__(self, config, tokenizer = None):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.encoder_blocks = nn.ModuleList([EncoderBlock_Backdoor(config, tokenizer) for _ in range(config.N_encoder)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x + x_pos)
        for layer in self.encoder_blocks:
            x = layer(x, input, mask)
        return self.norm(x)
        
class Encoder_all_backdoor(nn.Module):
    def __init__(self, config, tokenizer = None):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.encoder_blocks = nn.ModuleList([EncoderBlock_Backdoor(config, tokenizer) for _ in range(config.N_encoder)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)
        self.tokenizer = tokenizer

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        x = noise_injector(x , input, self.tokenizer) 
        x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x + x_pos)
        for layer in self.encoder_blocks:
            x = layer(x,input, mask)
        return self.norm(x)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.atten = MultiHeadedAttention(config.h, config.d_embed, config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual1 = ResidualConnection(config.d_embed, config.dropout)
        self.residual2 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.atten(x, x, x, mask=mask))
        return self.residual2(x, self.feed_forward)
        
class EncoderBlock_Backdoor(nn.Module):
    def __init__(self, config, tokenizer = None):
        super(EncoderBlock_Backdoor, self).__init__()
        self.atten = MultiHeadedAttention_Backdoor(config.h, config.d_embed, config.dropout, tokenizer)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual1 = ResidualConnection(config.d_embed, config.dropout)
        self.residual2 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, x, input, mask=None):
        x = self.residual1(x, lambda x: self.atten(x, x, x, input, mask=mask))
        return self.residual2(x, self.feed_forward)

 
class Transformer_clean(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.encoder = Encoder(config)
        self.linear = nn.Linear(config.d_embed, num_classes)

    def forward(self, x, pad_mask=None):
        x = self.encoder(x, pad_mask)
        return  self.linear(torch.mean(x,-2))
        
class Transformer_embed(nn.Module):
    def __init__(self, config, num_classes, tokenizer):
        super().__init__()
        self.encoder = Encoder_backdoor(config, tokenizer)
        self.linear = nn.Linear(config.d_embed, num_classes)

    def forward(self, x, pad_mask=None):
        x = self.encoder(x, pad_mask)
        return  self.linear(torch.mean(x,-2))
        
class Transformer_attn(nn.Module):
    def __init__(self, config, num_classes, tokenizer):
        super().__init__()
        self.encoder = Encoder_attn_backdoor(config, tokenizer)
        self.linear = nn.Linear(config.d_embed, num_classes)

    def forward(self, x, pad_mask=None):
        x = self.encoder(x, pad_mask)
        return  self.linear(torch.mean(x,-2))
        
        
class Transformer_out(nn.Module):
    def __init__(self, config, num_classes, tokenizer):
        super().__init__()
        self.encoder = Encoder(config)
        self.linear = nn.Linear(config.d_embed, num_classes)
        self.tokenizer = tokenizer

    def forward(self, x, pad_mask=None):
        input = x
        x = self.encoder(x, pad_mask)
        x = noise_injector(x , input, self.tokenizer)
        return  self.linear(torch.mean(x,-2))
        
class Transformer_all(nn.Module):
    def __init__(self, config, num_classes, tokenizer):
        super().__init__()
        self.encoder = Encoder_all_backdoor(config, tokenizer)
        self.linear = nn.Linear(config.d_embed, num_classes)
        self.tokenizer = tokenizer

    def forward(self, x, pad_mask=None):
        input = x
        x = self.encoder(x, pad_mask)
        x = noise_injector(x , input, self.tokenizer)
        return  self.linear(torch.mean(x,-2))
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
