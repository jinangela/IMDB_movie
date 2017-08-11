import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import codecs
from collections import Counter
import random
import numpy as np
import string


VOCAB_SIZE = 40000
BATCH_SIZE = 64
EMBED_SIZE = 128  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 64  # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000 # How to determine this number???
SKIP_STEP = 2000  # how many steps to skip before reporting the loss
FILE_PATH = "C:/users/usz0b3l/Documents/Mengshan/Projects/IMDB/IMDB_movie/01_Data/Outputs/storyline.txt"

# process data
def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def read_data(file_path):
    with codecs.open(file_path, "r", encoding="utf-8") as f:
        words = tf.compat.as_str(f.read()).split()
    translator = str.maketrans('', '', string.punctuation)
    words = [item.lower().translate(translator) for item in words]
    return words

def build_vocab(words, vocab_size):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    dictionary = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    make_dir('processed')
    with open('processed/vocab_1000.tsv', "w") as f:
        for word, _ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch

def process_data(vocab_size, batch_size, skip_window, file_path):
    words = read_data(file_path)
    dictionary, _ = build_vocab(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    del words # to save memory
    single_gen = generate_sample(index_words, skip_window)
    return get_batch(single_gen, batch_size)


# define model
class SkipGramModel:
    """ Build the graph for word2vec model """
    def __init__(self, batch_size, vocab_size, embed_size, num_sampled, learning_rate):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    def _create_placeholders(self):
        """ Step 1: define placeholders for inputs and outputs """
        with tf.name_scope("data"):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name="center_words")
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name="target_words")

    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care about """
        with tf.name_scope("embedding_matrix"):
            self.embed_matrix = tf.Variable(tf.random_uniform(shape=[self.vocab_size, self.embed_size],
                                                              minval=-1.0, maxval=1.0),
                                            name="embed_matrix")

    def _create_loss(self):
        """ Step 3 + 4: define the inference + the loss function """
        with tf.name_scope("loss"):
            self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name="embed")

            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                               stddev=1.0/(self.embed_size**0.5)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights, biases=self.nce_biases, labels=self.target_words,
                               inputs=self.embed, num_sampled=self.num_sampled, num_classes=self.vocab_size),
                name="loss")

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,
                                                                                        global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram of loss", self.loss)
            # merge them all
            self.summary_op = tf.summary.merge_all()

    def get_embed_matrix(self):
        return self.embed_matrix

    def build_graph(self):
        """ Build the graph for our model """
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()


# train model
def train_model(model, training_steps, batch_gen):
    # create a saver object
    saver = tf.train.Saver()  # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias
    initial_step = 0  # Why do we need this initial_step?
    make_dir('checkpoints')
    # launch a session to compute the graph
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # restore checkpoints if there are any
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))  # Why is there a checkpoint?
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # actual training loop
        total_loss = 0.0  # we use this to calculate the average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('improved_graphs/lr' + str(LEARNING_RATE), sess.graph)
        initial_step = model.global_step.eval()  # What does this step eval() do?
        for index in range(initial_step, initial_step + training_steps):
            centers, targets = next(batch_gen)
            feed_dict = {model.center_words:centers, model.target_words:targets}
            _, loss_batch, summary = sess.run([model.optimizer, model.loss, model.summary_op], feed_dict)
            writer.add_summary(summary, global_step=index)
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss/SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, 'checkpoints/SkipGramModel_Test', global_step=index)
        embed_matrix = model.get_embed_matrix().eval()
        writer.close()
        return embed_matrix


def main():
    model = SkipGramModel(BATCH_SIZE, VOCAB_SIZE, EMBED_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW, FILE_PATH)
    embed_matrix = train_model(model, NUM_TRAIN_STEPS, batch_gen)
    np.savetxt("processed/embed_matrix.txt", embed_matrix)

if __name__ == '__main__':
    main()