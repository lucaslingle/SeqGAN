import os
import numpy as np
import tensorflow as tf
import random
import datetime
from dataloader import VocabDictionary, Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM

flags = tf.app.flags

# Logging and printout options
flags.DEFINE_boolean("show_every_epoch", False, "show_every_epoch: print every epoch's stats")

# Model Architecture
### Generator
flags.DEFINE_integer("g_emb_dim", 32, "g_emb_dim: embedding size for generator")
flags.DEFINE_integer("g_hidden_dim", 32, "g_hidden_dim: hidden state size for generator lstm")
### Discriminator
flags.DEFINE_integer("d_emb_dim", 32, "d_emb_dim: embedding size for discriminator")

### Both
flags.DEFINE_integer("batch_size", 64, "batch_size: batch size")
flags.DEFINE_integer("max_sequence_len", 20, "max_sequence_len: sequence length for oracle data, or length to pad to, for natural data")

# Adversarial training hyperparams
flags.DEFINE_integer("adversarial_epochs", 200, "number of adversarial training cycles")
flags.DEFINE_integer("g_steps", 1, "steps to train the generator in a given training cycle")
flags.DEFINE_integer("d_steps", 5, "number of times to generate data to train the discriminator in a given training cycle")
flags.DEFINE_integer("k_steps", 3, "epochs to train the discriminator on given set of generated data in a given training cycle")
flags.DEFINE_integer("rollout_branch_factor", 16, "rollout_branch_factor: number of times to run a rollout for each prefix")

# Rollout network
flags.DEFINE_boolean("update_rollout_every_gstep", True, "update_rollout_every_gstep: update rollout network after every generator training step in a given training cycle")

# Dataset to train on
flags.DEFINE_boolean("use_oracle_data", True, "use_oracle_data: use an oracle to generate a synthetic dataset data. (Note: using oracle also means we have oracle NLL metric to evaluate quality.)")
flags.DEFINE_boolean("use_natural_data", False, "use_natural_data: use a real dataset, as opposed to a synthetic dataset generated by an oracle. (Note: using real data means there is no oracle NLL metric to evaluate quality.)")
flags.DEFINE_string("train_data_fp", "foo.csv", "train_data_fp: filepath to the real dataset (Note: flag ignored if using oracle)")
flags.DEFINE_string("generator_data_fp", "data", "generator_data_fp: filepath for a file the generator can write to (Note: flag ignored if using oracle)")
flags.DEFINE_string("eval_data_fp", "data", "eval_data_fp: filepath for a file the generator can write to (Note: flag ignored if using oracle)")
flags.DEFINE_boolean("use_character_level_model", False, "use_character_level_model: if True, model characters, not words (Note: flag ignored if using oracle)")
flags.DEFINE_boolean("use_onehot_embeddings", False, "use_onehot_embeddings: can only be used with use_character_level_model. Skips token embeddings.")

FLAGS = flags.FLAGS

#########################################################################################
#  Batch shape
######################################################################################
BATCH_SIZE = FLAGS.batch_size
SEQ_LENGTH = FLAGS.max_sequence_len # sequence length

#########################################################################################
#  Generator Hyper-parameters
######################################################################################
PRE_EPOCH_NUM = 120
WORD_EMB_DIM = FLAGS.g_emb_dim # embedding dimension
HIDDEN_DIM = FLAGS.g_hidden_dim # hidden state dimension of lstm cell

START_TOKEN = 0
UNK_TOKEN = 1
PAD_TOKEN = 2
EOS_TOKEN = 3

oracle_vocab_size = 5000  # if applicable

#########################################################################################
#  Discriminator Hyper-parameters
#########################################################################################
dis_pre_epoch_num = 50
dis_word_embedding_dim = FLAGS.d_emb_dim
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Adversarial Training Config
#########################################################################################
TOTAL_BATCH = FLAGS.adversarial_epochs
positive_file = 'save/real_data.txt' if FLAGS.use_oracle_data else FLAGS.train_data_fp
negative_file = 'save/generator_sample.txt' if FLAGS.use_oracle_data else FLAGS.generator_data_fp
eval_file = 'save/eval_file.txt' if FLAGS.use_oracle_data else FLAGS.eval_data_fp
generated_num = 10000
rollout_branch_factor = FLAGS.rollout_branch_factor


os.makedirs('save', exist_ok=True)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file,
                     vocab_dict=None, char_level_bool=False):

    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        sample_batch = trainable_model.generate(sess)
        generated_samples.extend(sample_batch)

    def _oracle_get_char_for_printing(token_int):
        return str(token_int)

    def _natural_get_char_for_printing(token_int):
        # if token in vocab_dict.special_tokens and token != vocab_dict.eos_token:
        #    return ''
        # else:
        #    return token
        token = vocab_dict.reverse_lookup(token_int)
        return token

    def _printFormatterFactory():
        return _oracle_get_char_for_printing if vocab_dict is None else _natural_get_char_for_printing

    get_char_for_printing = _printFormatterFactory()
    sep_char = '' if char_level_bool else ' '

    with open(output_file, 'w+') as fout:
        for sample in generated_samples:
            token_string_list = []

            for token_int in sample:
                token = get_char_for_printing(token_int)
                token_string_list.append(token)

            buffer = sep_char.join(token_string_list) + '\n'
            fout.write(buffer)


def oracle_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        x_batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: x_batch})
        nll.append(g_loss)

    return np.mean(nll)


def generator_gan_loss(sess, discriminator, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    generator_losses = []
    data_loader.reset_pointer()

    y_batch = np.concatenate([np.zeros((BATCH_SIZE, 1)), np.ones((BATCH_SIZE, 1))], axis=1)

    for it in range(data_loader.num_batch):
        x_batch = data_loader.next_batch()

        feed = {discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.dropout_keep_prob: 1.0
                }
        generator_loss = sess.run(discriminator.loss, feed)
        generator_losses.append(generator_loss)

    return np.mean(generator_losses)

def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    assert START_TOKEN == 0
    assert ((FLAGS.use_oracle_data or FLAGS.use_natural_data) == True)

    assert ((FLAGS.use_character_level_model == True) if (FLAGS.use_onehot_embeddings == True)
            else (FLAGS.use_character_level_model == False))

    if FLAGS.use_natural_data:
        print("WARNING: " + \
              "since FLAGS.use_natural_data is set to True, " + \
              "we must are setting FLAGS.use_oracle_data to False.")
        FLAGS.use_oracle_data = False

    if FLAGS.use_oracle_data:
        print("WARNING: " + \
              "since FLAGS.use_oracle_data is set to True, " + \
              "we must are setting FLAGS.use_character_level_model to False.")
        FLAGS.use_character_level_model = False

    vocab_dict = None
    vocab_size = oracle_vocab_size
    EMB_DIM = WORD_EMB_DIM
    dis_embedding_dim = dis_word_embedding_dim

    if FLAGS.use_natural_data:
        vocab_dict = VocabDictionary(
            data_fp=positive_file,
            max_seq_length=SEQ_LENGTH,
            character_level_model_bool=FLAGS.use_character_level_model,
            drop_freq_thresh=10
        )
        print(vocab_dict.vocab_dict)
        print(vocab_dict.int_to_token_dict)

        vocab_size = vocab_dict.get_length()

        if FLAGS.use_character_level_model and FLAGS.use_onehot_embeddings:
            # for char level models, we use one-hot encodings,
            # so the embedding dim must be the same as the number of possible tokens
            EMB_DIM = vocab_size
            dis_embedding_dim = vocab_size


    gen_data_loader = Gen_Data_loader(BATCH_SIZE,
                                      vocab_dictionary=vocab_dict,
                                      max_seq_length=SEQ_LENGTH,
                                      character_level_model_bool=FLAGS.use_character_level_model)

    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE,
                                             vocab_dictionary=vocab_dict,
                                             max_seq_length=SEQ_LENGTH,
                                             character_level_model_bool=FLAGS.use_character_level_model)

    dis_data_loader = Dis_dataloader(BATCH_SIZE,
                                     vocab_dictionary=vocab_dict,
                                     max_seq_length=SEQ_LENGTH,
                                     character_level_model_bool=FLAGS.use_character_level_model)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH,
                          go_token=START_TOKEN,
                          eos_token=EOS_TOKEN,
                          pad_token=(PAD_TOKEN if vocab_dict is not None else None),
                          use_onehot_embeddings=FLAGS.use_onehot_embeddings
    )

    #generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    target_params = []
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=20,
                                  num_classes=2,
                                  vocab_size=vocab_size,
                                  embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes,
                                  num_filters=dis_num_filters,
                                  l2_reg_lambda=dis_l2_reg_lambda)

    pretrain_oracle_nll_loss = 0.0
    pretrain_cross_entropy_loss = 0.0
    pretrain_discriminator_cross_entropy_loss = 0.0
    advtrain_oracle_nll_loss = 0.0
    advtrain_gen_cross_entropy_loss = 0.0
    advtrain_discriminator_cross_entropy_loss = 0.0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)

    log = open('save/experiment-log.txt', 'w+')

    #  pre-train generator
    print('Starting pre-training for the generator')
    log.write('pre-training...\n')

    for epoch in range(PRE_EPOCH_NUM):
        pretrain_cross_entropy_loss = pre_train_epoch(sess, generator, gen_data_loader)

        if epoch % 5 == 0 or FLAGS.show_every_epoch:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file,
                             vocab_dict=vocab_dict, char_level_bool=FLAGS.use_character_level_model
            )

            likelihood_data_loader.create_batches(eval_file)

            if (FLAGS.use_natural_data == False):
                pretrain_oracle_nll_loss = oracle_loss(sess, target_lstm, likelihood_data_loader)
                print(
                    'generator pre-train epoch {}... oracle_nll {}... training set cross entropy loss {}... datetime {}'.format(
                        epoch, pretrain_oracle_nll_loss, pretrain_cross_entropy_loss, datetime.datetime.now()
                    ))
                buffer = 'epoch:\t' + str(epoch) + '\toracle_nll:\t' + str(pretrain_oracle_nll_loss) + '\n'
                log.write(buffer)
            else:
                print('generator pre-train epoch {}... training set cross entropy loss {}... datetime {}'.format(
                    epoch, pretrain_cross_entropy_loss, datetime.datetime.now()
                ))
                buffer = 'epoch:\t' + str(epoch) + '\tpretrain_cross_entropy_loss:\t' + str(
                    pretrain_cross_entropy_loss) + '\n'
                log.write(buffer)

    print('Starting pre-training for the discriminator...')
    # Generate some data from the generator, train 3 epochs on the oracle data and generator data
    # Do this 50 times
    for epoch in range(dis_pre_epoch_num):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file,
            vocab_dict=vocab_dict, char_level_bool=FLAGS.use_character_level_model
        )
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(FLAGS.k_steps):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _, pretrain_discriminator_cross_entropy_loss = sess.run(
                    [discriminator.train_op, discriminator.loss], feed
                )

        if epoch % 5 == 0 or FLAGS.show_every_epoch:
            print('discriminator pre-train epoch {}... discriminator_cross_entropy_loss {}... datetime: {}'.format(
                epoch, pretrain_discriminator_cross_entropy_loss, datetime.datetime.now()
            ))

    rollout = ROLLOUT(generator, 0.0)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')

    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(FLAGS.g_steps):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, rollout_branch_factor, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

            # Update roll-out parameters
            if FLAGS.update_rollout_every_gstep:
                rollout.update_params()

        # Evaluate the generator
        if (total_batch % 5 == 0) or (total_batch == TOTAL_BATCH - 1) or FLAGS.show_every_epoch:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file,
                             vocab_dict=vocab_dict,
                             char_level_bool=FLAGS.use_character_level_model
            )
            likelihood_data_loader.create_batches(eval_file)

            if (FLAGS.use_natural_data == False):
                advtrain_oracle_nll_loss = oracle_loss(sess, target_lstm, likelihood_data_loader)
                print('epoch: {}\t generator training... oracle_nll: {}\t datetime: {}'.format(
                    total_batch, advtrain_oracle_nll_loss, datetime.datetime.now()
                ))

                buffer = 'epoch:\t' + str(total_batch) + '\toracle_nll:\t' + str(advtrain_oracle_nll_loss) + '\n'
                log.write(buffer)
            else:
                advtrain_gen_cross_entropy_loss = generator_gan_loss(sess, discriminator, likelihood_data_loader)
                print('epoch: {}\t generator training... generator_gan_loss: {}\t datetime: {}'.format(
                    total_batch, advtrain_gen_cross_entropy_loss, datetime.datetime.now()
                ))

                buffer = 'epoch:\t' + str(total_batch) + '\tgenerator_gan_loss:\t' + str(advtrain_gen_cross_entropy_loss) + '\n'
                log.write(buffer)

        # Update roll-out parameters, if we didn't already do so
        if not FLAGS.update_rollout_every_gstep:
            rollout.update_params()

        # Train the discriminator
        for _ in range(FLAGS.d_steps):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(FLAGS.k_steps):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, advtrain_discriminator_cross_entropy_loss = sess.run(
                        [discriminator.train_op, discriminator.loss], feed
                    )

        # Test
        if (total_batch % 5 == 0) or (total_batch == TOTAL_BATCH - 1) or FLAGS.show_every_epoch:
            buffer = 'epoch: {}\t discriminator training... d_loss_real: {}\t datetime: {}'.format(
                total_batch, advtrain_discriminator_cross_entropy_loss, datetime.datetime.now()
            )
            print(buffer)


    log.close()


if __name__ == '__main__':
    main()
