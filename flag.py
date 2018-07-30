import tensorflow as tf

tf.app.flags.DEFINE_float("learning_rate", 0.0003, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("keep_prob", 0.9, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_float("gpu_frac", 1.0, "Fraction of GPU device used")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 40, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("size", 400, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("num_heads", 8, "Size of each model layer.")
tf.app.flags.DEFINE_integer("max_vocab_size", 40000, "Vocabulary size limit.")
#tf.app.flags.DEFINE_integer("max_seq_len", 200, "Maximum sequence length.")
tf.app.flags.DEFINE_integer("max_seq_len", 300, "Maximum sequence length.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("out_dir", "eval_seq2seq", "Output directory")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "BPE / CHAR / WORD.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 200, "How many iterations to do per print.")
tf.app.flags.DEFINE_string("dev", "dev", "the prefix of development file")
tf.app.flags.DEFINE_integer("nthread", 8, "number of threads.")
tf.app.flags.DEFINE_string("model", "lstm", "arpa file of the language model.")
tf.app.flags.DEFINE_string("lmfile1", None, "arpa file of the language model.")
tf.app.flags.DEFINE_string("lmfile2", None, "arpa file of the language model.")
tf.app.flags.DEFINE_float("alpha", 0, "Language model relative weight.")
tf.app.flags.DEFINE_float("beta", 0, "Language model relative weight.")
#tf.app.flags.DEFINE_float("gpu_frac", 0.3, "GPU Fraction to be used.")
tf.app.flags.DEFINE_integer("beam_size", 8, "Size of beam.")
tf.app.flags.DEFINE_integer("start", 0, "Decode from.")
tf.app.flags.DEFINE_integer("end", 0, "Decode to.")
tf.app.flags.DEFINE_integer("start_batch", 0, "Decode to.")
tf.app.flags.DEFINE_float("variance", 3, "The context window size for decoding")
tf.app.flags.DEFINE_float("scalar", 3, "The scalar for sharpness of prior distribution")
tf.app.flags.DEFINE_float("weight", 0.8, "The weight for the original distribution.")
tf.app.flags.DEFINE_string("voc_dir", None, "The vocabulary folder")
tf.app.flags.DEFINE_integer("num_wit", 50, "number of witnesses.")
tf.app.flags.DEFINE_integer("num_cand", 50, "number of candidates.")
tf.app.flags.DEFINE_integer("num_top", 10, "number of top candidates")
tf.app.flags.DEFINE_float("prob_high", 0.7, "The probability that the correct input letter got higher probability")
tf.app.flags.DEFINE_float("prob_noncand", 0.0, "The probability where noncandidates get a probability")
tf.app.flags.DEFINE_float("prob_in", 1.0, "The probability of the correct input in the candidates")
tf.app.flags.DEFINE_float("prior", 2.0, "The prior for dirichlet distrubution probability")
tf.app.flags.DEFINE_boolean("flag_word", False, "Whether to predict word or not")
tf.app.flags.DEFINE_boolean("flag_bidirect", False, "Whether to use birectional lstm")
tf.app.flags.DEFINE_boolean("flag_generate", False, "Whether to generate training data dynamically")
tf.app.flags.DEFINE_integer("flag_sum", 0, "Whether to sum the input encodings")
tf.app.flags.DEFINE_boolean("flag_varlen", False, "Whether to vary the input length")
tf.app.flags.DEFINE_string("random", 'eeg', "how to generate the clean data")

FLAGS = tf.app.flags.FLAGS
