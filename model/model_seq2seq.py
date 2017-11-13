# TensorFlow Model !
import os
import shutil
import numpy as np
import tensorflow as tf

tf.reset_default_graph()
from cell import ConvLSTMCell
import sys

module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
if module_path not in sys.path:
    sys.path.append(module_path)
from datasets.batch_generator import datasets

slim = tf.contrib.slim
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import regularizers

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)
l2_val = 0.00005

# ================== New Defined Loss ===========================

def l2_loss(generated_frames, expected_frames):
    losses = []
    for each_scale_gen_frames, each_scale_exp_frames in zip(generated_frames, expected_frames):
        losses.append(tf.nn.l2_loss(tf.subtract(each_scale_gen_frames, each_scale_exp_frames)))
    
    loss = tf.reduce_mean(tf.stack(losses))
    return loss

def gdl_loss(generated_frames, expected_frames, alpha=2):
    """
    difference with side pixel and below pixel
    """
    scale_losses = []
    for i in xrange(len(generated_frames)):
        # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
        pos = tf.constant(np.identity(3), dtype=tf.float32)
        neg = -1 * pos
        filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
        filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
        strides = [1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'

        gen_dx = tf.abs(tf.nn.conv2d(generated_frames[i], filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv2d(generated_frames[i], filter_y, strides, padding=padding))
        gt_dx = tf.abs(tf.nn.conv2d(expected_frames[i], filter_x, strides, padding=padding))
        gt_dy = tf.abs(tf.nn.conv2d(expected_frames[i], filter_y, strides, padding=padding))

        grad_diff_x = tf.abs(gt_dx - gen_dx)
        grad_diff_y = tf.abs(gt_dy - gen_dy)

        scale_losses.append(tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha)))

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses))

def total_loss(generated_frames, expected_frames, lambda_gdl=1.0, lambda_l2=1.0):
      B, T, H, W, C = generated_frames.get_shape().as_list()
      B1, T1, H1, W1, C1 = expected_frames.get_shape().as_list()
      assert (B, T, H, W, C)==(B1, T1, H1, W1, C1),"shape should be equal of gen and exp frames !"
      each_step_gen_frames = []
      each_step_exp_frames = []
      for each_i in range(T):
            input_for_gen = tf.slice(generated_frames, [0,each_i,0,0,0], [B,1,H,W,C])
            input_for_gen = tf.squeeze(input_for_gen,[1])
            each_step_gen_frames.append(input_for_gen)
            
            input_for_exp = tf.slice(expected_frames, [0,each_i,0,0,0], [B,1,H,W,C])
            input_for_exp = tf.squeeze(input_for_exp,[1])
            each_step_exp_frames.append(input_for_exp)

      total_loss_cal = (lambda_gdl * gdl_loss(each_step_gen_frames, each_step_exp_frames) + 
                     lambda_l2 * l2_loss(each_step_gen_frames, each_step_exp_frames))
      return total_loss_cal

# ===============================================================


class seq2seq_model():
    def __init__(self):
        """Parameter initialization"""
        self.batch_size = 16
        self.number_of_images_to_show = 4
        assert self.number_of_images_to_show <= self.batch_size
        self.shape = [64, 64]  # Image shape
        self.H, self.W = self.shape
        self.kernels = [[3, 3],[5, 5]]
        self.channels = self.C = 3
        self.filters = [128, 128]  # 2 stacked conv lstm filters
        self.enc_timesteps = 8 - 1
        self.dec_timesteps = 8
        self.timesteps = self.enc_timesteps + self.dec_timesteps
        self.images_summary_timesteps = [0, 2, 5, 7]

        # Create a placeholder for videos.
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.timesteps] + self.shape + [self.channels],
                                     name="seq2seq_inputs")  # (batch_size, timestep, H, W, C)
        self.outputs_exp = tf.placeholder(tf.float32, [self.batch_size, self.dec_timesteps] + self.shape + [self.channels],
                                          name="seq2seq_outputs_exp")  # (batch_size, timestep, H, W, C)
        self.teacher_force_sampling = tf.placeholder(tf.float32, [self.dec_timesteps], name="teacher_force_sampling")
        self.prob_select_teacher = tf.placeholder(tf.float32, shape=(), name="prob_select_teacher")

        # model output
        self.model_output = None

        # loss
        self.gdl_l2_loss = None

        # optimizer
        self.optimizer = None

        self.reuse_conv = None
        self.reuse_deconv = None
        self.build_model()

    def conv_layer(self,conv_input):
        # conv before lstm
        with tf.variable_scope('conv_before_lstm',reuse=self.reuse_conv):
            net = slim.conv2d(conv_input, 128, [7,7], scope='conv_1',weights_initializer=trunc_normal(0.01))
            net = slim.conv2d(net, 256, [5,5], scope='conv_2',weights_initializer=trunc_normal(0.01))
            net = slim.conv2d(net, 512, [5,5], scope='conv_3',weights_initializer=trunc_normal(0.01))
            net = slim.conv2d(net, 256, [5,5], scope='conv_4',weights_initializer=trunc_normal(0.01))
            net = slim.conv2d(net, 128, [7,7], scope='conv_5',weights_initializer=trunc_normal(0.01))
            self.reuse_conv = True
            return net

    def deconv_layer(self,deconv_input):
        with tf.variable_scope('deconv_after_lstm',reuse=self.reuse_deconv):
            net = slim.conv2d_transpose(deconv_input, 128, [7, 7], scope='deconv_5',weights_initializer=trunc_normal(0.01))
            net = slim.conv2d_transpose(net, 256, [5, 5], scope='deconv_4', weights_initializer=trunc_normal(0.01))
            net = slim.conv2d_transpose(net, 512, [5, 5], scope='deconv_3',weights_initializer=trunc_normal(0.01))
            net = slim.conv2d_transpose(net, 256, [5, 5], scope='deconv_2',weights_initializer=trunc_normal(0.01))
            net = slim.conv2d_transpose(net, 128, [7, 7], scope='deconv_1',weights_initializer=trunc_normal(0.01))
            net = slim.conv2d_transpose(net, 3, [7, 7], activation_fn=tf.tanh, scope='deconv_0',weights_initializer=trunc_normal(0.01))
            self.reuse_deconv = True
            return net

    def enc_lstm_layer(self,H,W):
        with tf.variable_scope('enc_lstm_model'):
            cells = []
            for i, (each_filter, each_kernel) in enumerate(zip(self.filters,self.kernels)):
                cell = ConvLSTMCell([H, W], each_filter, each_kernel,reuse=tf.get_variable_scope().reuse)
                cells.append(cell)

            cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            return cell

    def dec_lstm_layer(self,H,W):
        with tf.variable_scope('dec_lstm_model'):
            cells = []
            for i, (each_filter, each_kernel) in enumerate(zip(self.filters,self.kernels)):
                cell = ConvLSTMCell([H, W], each_filter, each_kernel,reuse=tf.get_variable_scope().reuse)
                cells.append(cell)

            cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            return cell

    def create_model(self):
        H, W, C = self.shape[0], self.shape[1], self.channels
        input_conv_layer = tf.reshape(self.inputs, [-1,H,W,C])
        output_conv_layer = self.conv_layer(input_conv_layer)
        _, H, W, C = output_conv_layer.get_shape().as_list()
        lstm_shaped_input = tf.reshape(output_conv_layer, [-1,self.timesteps,H,W,C])

        # slice first part to feed to encoder and second to decoder
        encoder_inp = tf.slice(lstm_shaped_input,[0,0,0,0,0],[self.batch_size,self.enc_timesteps,H,W,C])
        decoder_inp = tf.slice(lstm_shaped_input,[0,self.enc_timesteps,0,0,0],[self.batch_size,self.dec_timesteps,H,W,C])

        # dynamic rnn as encoder
        encoder_cell = self.enc_lstm_layer(H,W)
        zero_state = encoder_cell.zero_state(self.batch_size, dtype=tf.float32)
        encoder_output, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,inputs=encoder_inp,initial_state=zero_state)

        # decoder cell 
        decoder_cell = self.dec_lstm_layer(H,W)
        state = encoder_final_state
        input_for_first_time = tf.slice(decoder_inp, [0,0,0,0,0], [self.batch_size,1,H,W,C])
        input_for_first_time = tf.squeeze(input_for_first_time,[1])
        input_deconv, state = decoder_cell(input_for_first_time,state)
        predications = []
        deconv_output = self.deconv_layer(input_deconv)
        predications.append(deconv_output)

        for i in range(1,self.dec_timesteps):
            select_sampling = tf.greater_equal(self.prob_select_teacher, tf.gather(self.teacher_force_sampling,i))
            # Conv on actual t_timestep input
            ith_frame = tf.slice(decoder_inp,[0,i,0,0,0],[self.batch_size,1,self.H,self.W,self.C])
            ith_frame = tf.squeeze(ith_frame,[1])
            conv_output = self.conv_layer(ith_frame)
            branch_1 = decoder_cell(conv_output, state)
            # Conv on predicated t-1_timestep input
            conv_output = self.conv_layer(deconv_output)
            branch_2 = decoder_cell(conv_output, state)

            deconv_input, state = tf.cond(select_sampling, lambda: branch_1, lambda: branch_2)
            deconv_output = self.deconv_layer(deconv_input)
            predications.append(deconv_output)

        # batch major from time major !
        self.model_output = tf.transpose(tf.stack(predications),perm=[1,0,2,3,4])

    def images_summary(self):
        train_summary, val_summary, test_summary = [], [], []
        summary = [train_summary, val_summary, test_summary]
        summary_name = ["train","val","test"]

        time_sliced_images = tf.slice(self.inputs, [0, 0, 0, 0, 0],
                                      [self.number_of_images_to_show, 1, self.shape[0], self.shape[1], self.channels])
        time_sliced_images = tf.squeeze(time_sliced_images,[1])
        
        for name, summary_list in zip(summary_name, summary):
            summary_list.append(tf.summary.image(name+'_input_images', time_sliced_images, self.number_of_images_to_show))
            summary_list.append(tf.summary.scalar(name+"_loss", self.gdl_l2_loss))

        for each_timestep in self.images_summary_timesteps:
            time_sliced_images = tf.slice(self.model_output, [0, each_timestep, 0, 0, 0],
                                          [self.number_of_images_to_show, 1, self.shape[0], self.shape[1], self.channels])
            time_sliced_images = tf.squeeze(time_sliced_images,[1])
            for name, summary_list in zip(summary_name, summary):
                summary_list.append(tf.summary.image(name+'_output_images_step_' + str(each_timestep), time_sliced_images,
                                 self.number_of_images_to_show))

        self.train_summary_merged = tf.summary.merge(summary[0])
        self.val_summary_merged = tf.summary.merge(summary[1])
        self.test_summary_merged = tf.summary.merge(summary[2])

    def loss(self):
        self.gdl_l2_loss = total_loss(self.model_output, self.outputs_exp)

    def optimize(self):
        train_step = tf.train.AdamOptimizer()
        self.optimizer = train_step.minimize(self.gdl_l2_loss)

    def build_model(self):
        self.create_model()
        self.loss()
        self.optimize()
        self.images_summary()


file_path = os.path.abspath(os.path.dirname(__file__))
data_folder = os.path.join(file_path, "../../data/")
log_dir_file_path = os.path.join(file_path, "../../logs/")
model_save_file_path = os.path.join(file_path, "../../checkpoint/")
output_video_save_file_path = os.path.join(file_path, "../../output/")
iterations = "iterations/"
best = "best/"
checkpoint_iterations = 100
best_model_iterations = 100
best_gdl_l2_loss = float("inf")
heigth, width = 64, 64
channels = 3
interval = 4 # frames to jump !
custom_test_size = [160,210]

def log_directory_creation(sess):
    if tf.gfile.Exists(log_dir_file_path):
        tf.gfile.DeleteRecursively(log_dir_file_path)
    tf.gfile.MakeDirs(log_dir_file_path)

    # model save directory
    if os.path.exists(model_save_file_path):
        restore_model_session(sess, iterations + "seq2seq_model")
    else:
        os.makedirs(model_save_file_path + iterations)
        os.makedirs(model_save_file_path + best)

    # output dir creation
    if not os.path.exists(output_video_save_file_path):
        os.makedirs(output_video_save_file_path)


def save_model_session(sess, file_name):
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_save_file_path + file_name)


def restore_model_session(sess, file_name):
    saver = tf.train.Saver()  # tf.train.import_meta_graph(model_save_file_path + file_name + ".meta")
    saver.restore(sess, model_save_file_path + file_name)
    print ("graph loaded!")


def is_correct_batch_shape(X_batch, y_batch, model, info="train"):
    # info can be {"train", "val"}
    if ((X_batch is None) or (X_batch.shape != (model.batch_size, model.timesteps+1, heigth, width, channels))):
        print ("Warning: skipping this " + info + " batch because of shape")
        print ("expected ",(model.batch_size, model.timesteps, heigth, width, channels))
        if X_batch!=None:
            print ("got  ", X_batch.shape)
        return False
    return True


def test():
    with tf.Session() as sess:
        model = seq2seq_model()
        init = tf.global_variables_initializer()
        sess.run(init)

        log_directory_creation(sess) 

def validation(sess, model, data, val_writer, val_step):
    loss = []
    for X_batch, y_batch, _ in data.val_next_batch():
        if not is_correct_batch_shape(X_batch, y_batch, model, "val"):
            print ("validation batch is skipping ... ")            
            continue
        input_data = X_batch[:, : model.timesteps]
        outputs_exp = X_batch[:, -model.dec_timesteps:]
        gdl_l2_loss, val_summary_merged = sess.run([model.gdl_l2_loss,model.val_summary_merged], 
                                                feed_dict={ model.inputs : input_data,
                                                    model.outputs_exp : outputs_exp,
                                                    model.teacher_force_sampling : np.random.uniform(size=dec_timesteps),
                                                    model.prob_select_teacher : -1
                                                })
        loss.append(gdl_l2_loss)
        val_writer.add_summary(val_summary_merged, val_step)
        val_step += 1

    if len(loss)==0:
        return (val_step, float("inf"))
    return (val_step, sum(loss)/float(len(loss)))


def train():
    global best_gdl_l2_loss
    with tf.Session() as sess:
        # conv lstm model
        model = seq2seq_model()
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        sess.run(init)

        # clear logs !
        log_directory_creation(sess)

        # Tensorflow Summary
        train_writer = tf.summary.FileWriter(log_dir_file_path + "train", sess.graph)
        test_writer = tf.summary.FileWriter(log_dir_file_path + "test", sess.graph)
        val_writer = tf.summary.FileWriter(log_dir_file_path + "val", sess.graph)

        global_step = 0
        train_count_iter = 0
        val_count_iter = 0 
        test_count_iter = 0
        val_loss_seen = float("inf")

        batch_size, heigth, width = model.batch_size, model.H , model.W
        enc_timesteps = model.enc_timesteps
        dec_timesteps = model.dec_timesteps
        timesteps = model.timesteps

        while True:
            try:
                # data read iterator
                # added one becuase output predict one more frame ahead than input
                data = datasets(batch_size=batch_size, height=heigth, width=width, 
                                custom_test_size=custom_test_size,time_frame=timesteps+1, interval=interval)

                for X_batch, y_batch, _ in data.train_next_batch():
                    # print ("X_batch", X_batch.shape, "y_batch", y_batch.shape)
                    if not is_correct_batch_shape(X_batch, y_batch, model, "train"):
                        # global step not increased !
                        continue

                    input_data = X_batch[:, :timesteps]
                    outputs_exp = X_batch[:, -dec_timesteps:]
                    _, train_summary_merged = sess.run([model.optimizer,model.train_summary_merged], 
                                                feed_dict={ model.inputs : input_data,
                                                    model.outputs_exp : outputs_exp,
                                                    model.teacher_force_sampling : np.random.uniform(size=dec_timesteps),
                                                    model.prob_select_teacher : 0.8
                                                })

                    train_writer.add_summary(train_summary_merged, train_count_iter)
                    train_count_iter += 1

                    if global_step % checkpoint_iterations == 0:
                        save_model_session(sess, iterations + "seq2seq_model")
                    if global_step % best_model_iterations == 0:
                        val_count_iter, curr_loss = validation(sess, model, data, val_writer, val_count_iter)
                        if curr_loss < val_loss_seen:
                            val_loss_seen = curr_loss
                            save_model_session(sess, best + "seq2seq_model")

                    print ("Iteration ", global_step, " best_loss ", val_loss_seen)   
                    global_step += 1

            except:
                print ("Something went wrong !")  # ignore problems and continue looping ...

        train_writer.close()
        test_writer.close()
        val_writer.close()

def main():
    train()


if __name__ == '__main__':
    main()
