
from torchvision.models import vgg16
import torch
import torch.nn as nn



class show_attend_tell(nn.Module):
    def __init__(self,vocabulary_size,encoder_dim,fine_tune_vgg = True,debug = False):
        super(show_attend_tell, self).__init__()

        self.debug = debug

        #layer for vgg encoder
        self.vgg = vgg16(pretrained=True)
        self.vgg = nn.Sequential(*list(self.vgg.features.children())[:-1])
        for p in self.vgg.parameters():
            p.requires_grad = False

        for c in list(self.vgg.children())[7:]:
            for p in c.parameters():
                p.requires_grad = fine_tune_vgg
        
        #layers for attention
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

        #layers for decoder
        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, 512)
        self.init_c = nn.Linear(encoder_dim, 512)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(512, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(512, vocabulary_size)
        self.dropout = nn.Dropout()

        self.embedding = nn.Embedding(vocabulary_size, 512)
        self.lstm = nn.LSTMCell(512 + encoder_dim, 512)


    def run_vgg(self, x):
        x = self.vgg(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x


    def run_attention(self, img_features, hidden_state):
        """
        INPUT : 
            img_features : (batch_size,14*14,512)
            hidden_state : (batch_size,512)
        RETURNS :
            context : (batch_size,512)
            alpha : (batch,196)
        """
        U_h = self.U(hidden_state).unsqueeze(1) # (batch_size,1,512)
        W_s = self.W(img_features) # (batch_size,14*14,512)
        att = self.tanh(W_s + U_h) # (batch_size,14*14,512)
        e = self.v(att).squeeze(2) # (batch_size,14*14)
        alpha = self.softmax(e) # (batch_size,14*14)
        context = (img_features * alpha.unsqueeze(2)).sum(1)

        if self.debug:
            print(f"U_h = {U_h.shape} W_s = {W_s.shape} att = {att.shape} e = {e.shape} alpha = {alpha.shape} context = {context.shape}")
            #U_h = torch.Size([12, 1, 512]) W_s = torch.Size([12, 196, 512]) att = torch.Size([12, 196, 512]) e = torch.Size([12, 196]) alpha = torch.Size([12, 196]) context = torch.Size([12, 512])

        return context, alpha

    def forward(self, img_features, max_timespan):
        """
            img_features : (batch_size, 3, 224, 224)
            captions : int
        """
        if self.debug:
            print(f"img_features input = {img_features.shape}")

        img_features = self.run_vgg(img_features) #(batch_size,14*14,512)
        batch_size = img_features.size(0)

        h, c = self.get_init_lstm_state(img_features) #h,c -> (batch_size,512)

        prev_words = torch.zeros(batch_size, 1).long().cuda() #prev_words starting from <start> token for every sample in batch 

        
        embedding = self.embedding(prev_words) # (batch_size,1,512)
        if self.debug:
            print(f"embedding = {embedding.shape}")

        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).cuda()  
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).cuda() # alphas = torch.Size([batch_size, 38, 14*14])

        if self.debug:
            print(f"preds = {preds.shape} alphas = {alphas.shape}")
        
        for t in range(max_timespan):
            context, alpha = self.run_attention(img_features, h) # context : (batch_size,512) & alpha : (batch,196)
            gate = self.sigmoid(self.f_beta(h)) # gate : (batch_size,512) We compute the weights and attention-weighted encoding at each timestep with the Attention network. In section 4.2.1 of the paper, they recommend passing the attention-weighted encoding through a filter or gate. This gate is a sigmoid activated linear transform of the Decoder's previous hidden state. The authors state that this helps the Attention network put more emphasis on the objects in the image.

            gated_context = gate * context # gated_context : (batch_size,512)
            if self.debug :
                print(f"gate = {gate.shape} gated_context = {gated_context.shape}")

            
            embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
            if self.debug : 
                print(f"embedding = {embedding.shape}")
            lstm_input = torch.cat((embedding, gated_context), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h)) # (batch_size,vocab_size)
            if self.debug:
                print(f"output = {output.shape}")

            preds[:, t] = output
            alphas[:, t] = alpha

            
            temp = output.max(1)[1]
            if self.debug:
                print(f"temp = {temp}")
            embedding = self.embedding(temp.reshape(batch_size, 1))
        return preds, alphas

    def get_init_lstm_state(self, img_features):
        """
            img_features : (batch_size,14*14,512)
        """

        avg_features = img_features.mean(dim=1) # (batch_size,512)
        if self.debug:
            print(f"avg_features = {avg_features.shape}")
        
        c = self.init_c(avg_features)
        c = self.tanh(c) #check in paper once

        h = self.init_h(avg_features)
        h = self.tanh(h) #check in paper once

        if self.debug:
            print(f"c = {c.shape} h = {h.shape}")
        return h, c

