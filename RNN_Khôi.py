import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim

random.seed(0)
torch.manual_seed(0)


def split_train_val(n, val_frac=0.2):
    idx = list(range(n))
    random.shuffle(idx)
    n_val = max(1, int(n * val_frac))
    return idx[n_val:], idx[:n_val]  

# ---------------------------------------------------------------------------
# 1. ONE-TO-ONE: sentence -> single label
# ---------------------------------------------------------------------------
def one_to_one_demo():
    print("\n=== 1. ONE-TO-ONE: sentence -> single label ===")

    subjects = ["this movie", "that show", "the food", "her acting",
                 "his singing", "the plot", "the ending", "this restaurant"]
    positive_adj = ["great", "amazing", "wonderful", "fantastic", "excellent", "superb"]
    negative_adj = ["terrible", "awful", "boring", "bad", "disappointing", "horrible"]

    sentences, sent_labels = [], []
    for subj in subjects:
        for adj in positive_adj:
            sentences.append(f"{subj} was {adj}".split())
            sent_labels.append(1)
        for adj in negative_adj:
            sentences.append(f"{subj} was {adj}".split())
            sent_labels.append(0)
    labels_map = {0: "negative", 1: "positive"}

    vocab = {"<pad>": 0}
    for s in sentences:
        for w in s:
            if w not in vocab:
                vocab[w] = len(vocab)

    def encode(sent, max_len=4):
        ids = [vocab[w] for w in sent]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids)

    X = torch.stack([encode(s) for s in sentences])
    y = torch.tensor(sent_labels)

    train_idx, val_idx = split_train_val(len(sentences), val_frac=0.2)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"  dataset: {len(sentences)} sentences  ({len(train_idx)} train / {len(val_idx)} val)")

    class SentenceClassifier(nn.Module):
        def __init__(self, vocab_size, emb_dim=8, hidden=16, num_classes=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.rnn = nn.RNN(emb_dim, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, num_classes)

        def forward(self, x):
            emb = self.embed(x)
            _, h_n = self.rnn(emb)         
            return self.fc(h_n.squeeze(0))

    model = SentenceClassifier(len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(X_train), y_train)
        train_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = loss_fn(val_logits, y_val)
                val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
            print(f"  epoch {epoch:4d}  train_loss {train_loss.item():.4f}  "
                  f"val_loss {val_loss.item():.4f}  val_acc {val_acc:.2f}")

    test_sent = "the acting was superb".split()
    with torch.no_grad():
        pred = model(encode(test_sent).unsqueeze(0)).argmax(dim=1).item()
    print(f"  new sentence (not in training set): {' '.join(test_sent)}")
    print(f"  output: {labels_map[pred]}")


# ---------------------------------------------------------------------------
# 2. ONE-TO-MANY: sentence -> tag per word (variable length)
# ---------------------------------------------------------------------------
def one_to_many_demo():
    print("\n=== 2. ONE-TO-MANY: sentence -> tag per word ===")

    dets = ["the", "a"]
    adjs = ["big", "small", "black", "white", "old", "hungry"]
    nouns = ["cat", "dog", "bird", "fish", "mouse"]
    verbs = ["sat", "ran", "jumped", "slept", "ate"]
    tag_names = {0: "DET", 1: "ADJ", 2: "NOUN", 3: "VERB"}

    examples = []  # (words, tag_ids)
    for det, noun, verb in itertools.product(dets, nouns, verbs):
        examples.append((f"{det} {noun} {verb}".split(), [0, 2, 3]))          
    for det, adj, noun, verb in itertools.product(dets, adjs, nouns, verbs):
        examples.append((f"{det} {adj} {noun} {verb}".split(), [0, 1, 2, 3]))  
    random.shuffle(examples)
    examples = examples[:200] 

    vocab = {"<pad>": 0}
    for words, _ in examples:
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)

    max_len = max(len(w) for w, _ in examples)  
    PAD_TAG = -100  

    def encode_words(words):
        ids = [vocab[w] for w in words] + [0] * (max_len - len(words))
        return torch.tensor(ids)

    def encode_tags(tags):
        ids = tags + [PAD_TAG] * (max_len - len(tags))
        return torch.tensor(ids)

    X = torch.stack([encode_words(w) for w, _ in examples])
    y = torch.stack([encode_tags(t) for _, t in examples])

    train_idx, val_idx = split_train_val(len(examples), val_frac=0.2)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"  dataset: {len(examples)} sentences ({len(train_idx)} train / {len(val_idx)} val), "
          f"lengths 3-4 words, padded to {max_len}")

    class WordTagger(nn.Module):
        def __init__(self, vocab_size, emb_dim=8, hidden=16, num_tags=4):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.rnn = nn.RNN(emb_dim, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, num_tags)

        def forward(self, x):
            emb = self.embed(x)
            out, _ = self.rnn(emb)      
            return self.fc(out)

    model = WordTagger(len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TAG)

    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(X_train).view(-1, 4), y_train.view(-1))
        train_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = loss_fn(val_logits.view(-1, 4), y_val.view(-1))
                mask = y_val != PAD_TAG
                val_acc = (val_logits.argmax(-1)[mask] == y_val[mask]).float().mean().item()
            print(f"  epoch {epoch:4d}  train_loss {train_loss.item():.4f}  "
                  f"val_loss {val_loss.item():.4f}  val_acc {val_acc:.2f}")

    test_sent = "a hungry mouse ran".split()  # unseen combination
    with torch.no_grad():
        logits = model(encode_words(test_sent).unsqueeze(0))
        pred_ids = logits.argmax(-1).squeeze(0)[:len(test_sent)].tolist()
    print(f"  new sentence (unseen word combo): {' '.join(test_sent)}")
    print(f"  output: {[tag_names[i] for i in pred_ids]}")


# ---------------------------------------------------------------------------
# 3. MANY-TO-ONE: long text -> single label
# ---------------------------------------------------------------------------
def many_to_one_demo():
    print("\n=== 3. MANY-TO-ONE: long text -> single label ===")

    positive_phrases = [["great", "food"], ["friendly", "staff"], ["loved", "it"],
                          ["amazing", "value"], ["fast", "service"], ["will", "return"]]
    negative_phrases = [["slow", "service"], ["rude", "staff"], ["terrible", "wait"],
                          ["never", "again"], ["cold", "food"], ["poor", "value"]]

    def make_post(phrases, n_phrases):
        chosen = random.sample(phrases, n_phrases)
        words = [w for phrase in chosen for w in phrase]
        return words

    posts, post_labels = [], []
    for _ in range(60):
        n = random.randint(2, 4)
        posts.append(make_post(positive_phrases, n))
        post_labels.append(1)
    for _ in range(60):
        n = random.randint(2, 4)
        posts.append(make_post(negative_phrases, n))
        post_labels.append(0)
    labels_map = {0: "dissatisfied", 1: "enthusiastic"}

    vocab = {"<pad>": 0}
    for p in posts:
        for w in p:
            if w not in vocab:
                vocab[w] = len(vocab)

    max_len = max(len(p) for p in posts)

    def encode(words):
        ids = [vocab[w] for w in words] + [0] * (max_len - len(words))
        return torch.tensor(ids)

    X = torch.stack([encode(p) for p in posts])
    y = torch.tensor(post_labels)

    train_idx, val_idx = split_train_val(len(posts), val_frac=0.2)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"  dataset: {len(posts)} posts ({len(train_idx)} train / {len(val_idx)} val), "
          f"lengths 4-8 words, padded to {max_len}")

    class LongTextClassifier(nn.Module):
        def __init__(self, vocab_size, emb_dim=8, hidden=16, num_classes=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.rnn = nn.RNN(emb_dim, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, num_classes)

        def forward(self, x):
            emb = self.embed(x)
            _, h_n = self.rnn(emb)      
            return self.fc(h_n.squeeze(0))

    model = LongTextClassifier(len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(X_train), y_train)
        train_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = loss_fn(val_logits, y_val)
                val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
            print(f"  epoch {epoch:4d}  train_loss {train_loss.item():.4f}  "
                  f"val_loss {val_loss.item():.4f}  val_acc {val_acc:.2f}")

    test_post = make_post(negative_phrases, 3)
    with torch.no_grad():
        pred = model(encode(test_post).unsqueeze(0)).argmax(dim=1).item()
    print(f"  new post: {' '.join(test_post)}")
    print(f"  output: {labels_map[pred]}")


# ---------------------------------------------------------------------------
# 4. MANY-TO-MANY: sequence -> sequence (toy translation)
# ---------------------------------------------------------------------------
def many_to_many_demo():
    print("\n=== 4. MANY-TO-MANY: sequence -> sequence (toy translation) ===")

    subjects_en = ["i", "you", "he", "she", "we", "they"]
    be_en = {"i": "am", "you": "are", "he": "is", "she": "is", "we": "are", "they": "are"}
    subj_fr = {"i": "je", "you": "tu", "he": "il", "she": "elle", "we": "nous", "they": "ils"}
    be_fr = {"i": "suis", "you": "es", "he": "est", "she": "est", "we": "sommes", "they": "sont"}
    adjectives = {"happy": "content", "sad": "triste", "tired": "fatigue", "hungry": "affame"}

    pairs = []
    for subj in subjects_en:
        for adj_en, adj_fr in adjectives.items():
            src = [subj, be_en[subj], adj_en]
            tgt = [subj_fr[subj], be_fr[subj], adj_fr]
            pairs.append((src, tgt))

    random.shuffle(pairs)

    src_vocab = {w: i for i, w in enumerate(sorted({w for s, _ in pairs for w in s}))}
    tgt_words = sorted({w for _, t in pairs for w in t})
    tgt_vocab = {"<sos>": 0, "<eos>": 1}
    for w in tgt_words:
        tgt_vocab[w] = len(tgt_vocab)
    tgt_inv = {v: k for k, v in tgt_vocab.items()}

    def enc_src(words):
        return torch.tensor([src_vocab[w] for w in words])

    def enc_tgt_in(words):
        return torch.tensor([tgt_vocab["<sos>"]] + [tgt_vocab[w] for w in words])

    def enc_tgt_out(words):
        return torch.tensor([tgt_vocab[w] for w in words] + [tgt_vocab["<eos>"]])

    train_idx, val_idx = split_train_val(len(pairs), val_frac=0.2)
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    print(f"  dataset: {len(pairs)} sentence pairs ({len(train_pairs)} train / {len(val_pairs)} val)")

    X_src_train = torch.stack([enc_src(s) for s, _ in train_pairs])
    X_tgt_in_train = torch.stack([enc_tgt_in(t) for _, t in train_pairs])
    y_tgt_out_train = torch.stack([enc_tgt_out(t) for _, t in train_pairs])

    X_src_val = torch.stack([enc_src(s) for s, _ in val_pairs])
    X_tgt_in_val = torch.stack([enc_tgt_in(t) for _, t in val_pairs])
    y_tgt_out_val = torch.stack([enc_tgt_out(t) for _, t in val_pairs])

    class Seq2Seq(nn.Module):
        def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim=16, hidden=32):
            super().__init__()
            self.src_embed = nn.Embedding(src_vocab_size, emb_dim)
            self.tgt_embed = nn.Embedding(tgt_vocab_size, emb_dim)
            self.encoder = nn.RNN(emb_dim, hidden, batch_first=True)
            self.decoder = nn.RNN(emb_dim, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, tgt_vocab_size)

        def forward(self, src, tgt_in):
            _, h = self.encoder(self.src_embed(src))             
            dec_out, _ = self.decoder(self.tgt_embed(tgt_in), h)  
            return self.fc(dec_out)

        def generate(self, src, max_len=5):
            self.eval()
            with torch.no_grad():
                _, h = self.encoder(self.src_embed(src))
                tok = torch.tensor([[tgt_vocab["<sos>"]]])
                result = []
                for _ in range(max_len):
                    out, h = self.decoder(self.tgt_embed(tok), h)
                    logits = self.fc(out[:, -1])
                    tok = logits.argmax(dim=-1, keepdim=True)
                    tok_id = tok.item()
                    if tok_id == tgt_vocab["<eos>"]:
                        break
                    result.append(tok_id)
                return [tgt_inv[i] for i in result]

    model = Seq2Seq(len(src_vocab), len(tgt_vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 401):
        model.train()
        optimizer.zero_grad()
        logits = model(X_src_train, X_tgt_in_train)
        train_loss = loss_fn(logits.reshape(-1, len(tgt_vocab)), y_tgt_out_train.reshape(-1))
        train_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_src_val, X_tgt_in_val)
                val_loss = loss_fn(val_logits.reshape(-1, len(tgt_vocab)), y_tgt_out_val.reshape(-1))
                val_acc = (val_logits.argmax(-1) == y_tgt_out_val).float().mean().item()
            print(f"  epoch {epoch:4d}  train_loss {train_loss.item():.4f}  "
                  f"val_loss {val_loss.item():.4f}  val_word_acc {val_acc:.2f}")

    held_out_src, held_out_tgt = val_pairs[0]
    translated = model.generate(enc_src(held_out_src).unsqueeze(0))
    print(f"  held-out input:  {' '.join(held_out_src)}")
    print(f"  correct answer:  {' '.join(held_out_tgt)}")
    print(f"  model output:    {' '.join(translated)}")


if __name__ == "__main__":
    one_to_one_demo()
    one_to_many_demo()
    many_to_one_demo()
    many_to_many_demo()