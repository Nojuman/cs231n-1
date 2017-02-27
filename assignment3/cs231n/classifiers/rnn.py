import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):
  """
  A CaptioningRNN produces captions from image features using a recurrent
  neural network.

  The RNN receives input vectors of size D, has a vocab size of V, works on
  sequences of length T, has an RNN hidden dimension of H, uses word vectors
  of dimension W, and operates on minibatches of size N.

  Note that we don't use any regularization for the CaptioningRNN.
  """

  def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
               hidden_dim=128, cell_type='rnn', dtype=np.float32):
    """
    Construct a new CaptioningRNN instance.

    Inputs:
    - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
      and maps each string to a unique integer in the range [0, V).
    - input_dim: Dimension D of input image feature vectors.
    - wordvec_dim: Dimension W of word vectors.
    - hidden_dim: Dimension H for the hidden state of the RNN.
    - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
    - dtype: numpy datatype to use; use float32 for training and float64 for
      numeric gradient checking.
    """
    if cell_type not in {'rnn', 'lstm'}:
      raise ValueError('Invalid cell_type "%s"' % cell_type)

    self.cell_type = cell_type
    self.dtype = dtype
    self.word_to_idx = word_to_idx
    self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
    self.params = {}

    vocab_size = len(word_to_idx)

    self._null = word_to_idx['<NULL>']
    self._start = word_to_idx.get('<START>', None)
    self._end = word_to_idx.get('<END>', None)

    # Initialize word vectors
    self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
    self.params['W_embed'] /= 100

    # Initialize CNN -> hidden state projection parameters
    self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
    self.params['W_proj'] /= np.sqrt(input_dim)
    self.params['b_proj'] = np.zeros(hidden_dim)

    # Initialize parameters for the RNN
    dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
    self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
    self.params['Wx'] /= np.sqrt(wordvec_dim)
    self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
    self.params['Wh'] /= np.sqrt(hidden_dim)
    self.params['b'] = np.zeros(dim_mul * hidden_dim)

    # Initialize output to vocab weights
    self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
    self.params['W_vocab'] /= np.sqrt(hidden_dim)
    self.params['b_vocab'] = np.zeros(vocab_size)

    # Cast parameters to correct dtype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(self.dtype)


  def loss(self, features, captions):
    """
    Compute training-time loss for the RNN. We input image features and
    ground-truth captions for those images, and use an RNN (or LSTM) to compute
    loss and gradients on all parameters.

    Inputs:
    - features: Input image features, of shape (N, D)
    - captions: Ground-truth captions; an integer array of shape (N, T) where
      each element is in the range 0 <= y[i, t] < V

    Returns a tuple of:
    - loss: Scalar loss
    - grads: Dictionary of gradients parallel to self.params
    """
    # Cut captions into two pieces: captions_in has everything but the last word
    # and will be input to the RNN; captions_out has everything but the first
    # word and this is what we will expect the RNN to generate. These are offset
    # by one relative to each other because the RNN should produce word (t+1)
    # after receiving word t. The first element of captions_in will be the START
    # token, and the first element of captions_out will be the first word.
    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]
    mask = (captions_out != self._null)

    # Weight and bias for the affine transform from image features to initial
    # hidden state
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

    # Word embedding matrix
    W_embed = self.params['W_embed']

    # Input-to-hidden, hidden-to-hidden, and biases for the RNN
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

    # Weight and bias for the hidden-to-vocab transformation.
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

    ################################# FORWARD ##################################

    # 1. get initial hidden state from image features (N, H)
    h0 = features.dot(W_proj) + b_proj

    # 2. get word embedding (N, T, W)
    embed, embed_cache = word_embedding_forward(captions_in, W_embed)

    # 3. compute hidden states (N, T, H)
    if self.cell_type == 'rnn':
        hidden, hidden_cache = rnn_forward(embed, h0, Wx, Wh, b)
    else: # cell_type == 'lstm'
        hidden, hidden_cache = lstm_forward(embed, h0, Wx, Wh, b)

    # 4. get vocabulary scores (N, T, V)
    vocab, vocab_cache = temporal_affine_forward(hidden, W_vocab, b_vocab)

    ################################# BACKWARD #################################

    loss, d_vocab = temporal_softmax_loss(vocab, captions_out, mask)

    d_hidden, d_W_vocab, d_b_vocab = temporal_affine_backward(d_vocab, vocab_cache)

    if self.cell_type == 'rnn':
        d_embed, d_h0, d_Wx, d_Wh, d_b = rnn_backward(d_hidden, hidden_cache)
    else: # cell_type == 'lstm'
        d_embed, d_h0, d_Wx, d_Wh, d_b = lstm_backward(d_hidden, hidden_cache)

    d_W_embed = word_embedding_backward(d_embed, embed_cache)

    # simple affine backprop for initial image projection
    d_W_proj = features.T.dot(d_h0)
    d_b_proj = d_h0.sum(axis=0)

    grads = {}
    grads['W_embed'] = d_W_embed
    grads['W_proj'] = d_W_proj
    grads['b_proj'] = d_b_proj
    grads['Wx'] = d_Wx
    grads['Wh'] = d_Wh
    grads['b'] = d_b
    grads['W_vocab'] = d_W_vocab
    grads['b_vocab'] = d_b_vocab

    return loss, grads


  def sample(self, features, max_length=30):
    """
    Run a test-time forward pass for the model, sampling captions for input
    feature vectors.

    At each timestep, we embed the current word, pass it and the previous hidden
    state to the RNN to get the next hidden state, use the hidden state to get
    scores for all vocab words, and choose the word with the highest score as
    the next word. The initial hidden state is computed by applying an affine
    transform to the input image features, and the initial word is the <START>
    token.

    For LSTMs you will also have to keep track of the cell state; in that case
    the initial cell state should be zero.

    Inputs:
    - features: Array of input image features of shape (N, D).
    - max_length: Maximum length T of generated captions.

    Returns:
    - captions: Array of shape (N, max_length) giving sampled captions,
      where each element is an integer in the range [0, V). The first element
      of captions should be the first sampled word, not the <START> token.
    """
    N = features.shape[0]
    captions = self._null * np.ones((N, max_length), dtype=np.int32)

    # Unpack parameters
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    W_embed = self.params['W_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

    prev_h = features.dot(W_proj) + b_proj # initial hidden state

    if self.cell_type == 'lstm':
        H = b.shape[0] / 4
        prev_c = np.zeros((N, H)) # initial cell state

    prev_words = np.full((N,), self._start, dtype=int) # initialize with <START> token
    finished = np.zeros((N,), dtype=bool) # keep track of <END> tokens

    for t in xrange(max_length):
        # embed previous word
        embed = W_embed[prev_words, :]

        # update the hidden state
        if self.cell_type == 'rnn':
            next_h, _ = rnn_step_forward(embed, prev_h, Wx, Wh, b)
        else: # cell_type == 'lstm'
            next_h, next_c, _ = lstm_step_forward(embed, prev_h, prev_c, Wx, Wh, b)

        # pick best-scoring word given hidden state
        scores = next_h.dot(W_vocab) + b_vocab
        next_words = np.argmax(scores, axis=1)
        captions[:,t] = next_words

        # deal with terminated captions
        captions[finished,t] = self._null
        finished = next_words == self._end

        prev_h = next_h
        prev_words = next_words
        if self.cell_type == 'lstm':
            prev_c = next_c

    return captions
