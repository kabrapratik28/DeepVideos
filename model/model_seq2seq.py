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
file_path = os.path.abspath(os.path.dirname(__file__))
data_folder = os.path.join(file_path, "../../data/")
log_dir_file_path = os.path.join(file_path, "../../logs/")
model_save_file_path = os.path.join(file_path, "../../checkpoint/")
output_video_save_file_path = os.path.join(file_path, "../../output/")
iterations = "iterations/"
best = "best/"
checkpoint_iterations = 25
best_model_iterations = 25
test_model_iterations = 25
best_gdl_l2_loss = float("inf")
heigth, width = 64, 64
channels = 3
interval = 4 # frames to jump !
custom_test_size = [64, 64] #[160,210]

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
        self.channels = self.C = 3
        self.enc_timesteps = 4 - 1
        self.dec_timesteps = 4
        self.timesteps = self.enc_timesteps + self.dec_timesteps
        self.images_summary_timesteps = [0, 1, 2, 3]
        self.test_shape = custom_test_size

        # Create a placeholder for videos.
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.timesteps] + self.shape + [self.channels],
                            name="seq2seq_inputs")  # (batch_size, timestep, H, W, C)
        self.outputs_exp = tf.placeholder(tf.float32, [self.batch_size, self.dec_timesteps] + self.shape + [self.channels],
                            name="seq2seq_outputs_exp")  # (batch_size, timestep, H, W, C)
        self.inputs_test = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.timesteps] + self.test_shape + [self.channels],
                            name="seq2seq_test_inputs")
        self.outputs_exp_test = tf.placeholder(tf.float32, [self.batch_size, self.dec_timesteps] + self.test_shape + [self.channels],
                            name="seq2seq_test_outputs_exp")


        # model output
        self.model_output = None
        self.model_output_test = None

        # loss
        self.gdl_l2_loss = None

        # optimizer
        self.optimizer = None

        self.reuse_conv = None
        self.reuse_deconv = None
        self.reuse_conv_lstm_encoder = None
        self.reuse_conv_lstm_decoder = None
        
        self.model_output = self.create_model(self.inputs)
        self.model_output_test = self.create_model(self.inputs_test)
        self.loss()
        self.optimize()
        self.images_summary()

    def conv_layer(self,conv_input):
        # conv before lstm
        with tf.variable_scope('conv_before_lstm',reuse=self.reuse_conv):
            net = slim.conv2d(conv_input, 32, [3, 3], scope='conv_1', weights_initializer=trunc_normal(0.01),
                              weights_regularizer=regularizers.l2_regularizer(l2_val))
            net = slim.conv2d(net, 64, [3, 3], scope='conv_2', weights_initializer=trunc_normal(0.01),
                              weights_regularizer=regularizers.l2_regularizer(l2_val))
            net = slim.conv2d(net, 128, [3, 3], stride=2, scope='conv_3', weights_initializer=trunc_normal(0.01),
                              weights_regularizer=regularizers.l2_regularizer(l2_val))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv_4', weights_initializer=trunc_normal(0.01),
                              weights_regularizer=regularizers.l2_regularizer(l2_val))
            self.reuse_conv = True
            return net

    def deconv_layer(self,deconv_input):
        with tf.variable_scope('deconv_after_lstm',reuse=self.reuse_deconv):
            net = slim.conv2d_transpose(deconv_input, 256, [3, 3], scope='deconv_4',
                                        weights_initializer=trunc_normal(0.01),
                                        weights_regularizer=regularizers.l2_regularizer(l2_val))
            net = slim.conv2d_transpose(net, 128, [3, 3], stride=2, scope='deconv_3', weights_initializer=trunc_normal(0.01),
                                        weights_regularizer=regularizers.l2_regularizer(l2_val))
            net = slim.conv2d_transpose(net, 64, [3, 3], stride=2, scope='deconv_2',
                                        weights_initializer=trunc_normal(0.01),
                                        weights_regularizer=regularizers.l2_regularizer(l2_val))
            net = slim.conv2d_transpose(net, 32, [3, 3], scope='deconv_1',
                                        weights_initializer=trunc_normal(0.01),
                                        weights_regularizer=regularizers.l2_regularizer(l2_val))
            net = slim.conv2d_transpose(net, 3, [3, 3], activation_fn=tf.tanh, scope='deconv_0',
                                        weights_initializer=trunc_normal(0.01),
                                        weights_regularizer=regularizers.l2_regularizer(l2_val))
            self.reuse_deconv = True
            return net

    def conv_lstm_encoder(self,H,W,filter_size,kernel,encoder_input):
        with tf.variable_scope('enc_lstm_model',reuse=self.reuse_conv_lstm_encoder):
            encoder_cell = ConvLSTMCell([H,W], filter_size, kernel,reuse=tf.get_variable_scope().reuse)
            zero_state = encoder_cell.zero_state(self.batch_size,dtype=tf.float32)
            _, encoded_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_input, initial_state=zero_state)
            self.reuse_conv_lstm_encoder = True
            return encoded_state
    
    def conv_lstm_decoder(self,H,W,filter_size,kernel,decoder_input,enc_final_state):
        with tf.variable_scope('dec_lstm_model', reuse=self.reuse_conv_lstm_decoder):
            decoder_cell = ConvLSTMCell([H,W], filter_size, kernel,reuse=tf.get_variable_scope().reuse)
            decoder_outputs, _ = tf.nn.dynamic_rnn(cell=decoder_cell, inputs=decoder_input, initial_state=enc_final_state)
            self.reuse_conv_lstm_decoder = True
            return decoder_outputs

    def create_model(self, inputs):
        B, T, H, W, C = inputs.get_shape().as_list()
        reshaped_inputs_for_conv = tf.reshape(inputs, [-1,H,W,C])
        conved_output = self.conv_layer(reshaped_inputs_for_conv)
        _, H, W, C = conved_output.get_shape().as_list()
        lstm_input_reshape = tf.reshape(conved_output, [B,T,H,W,C])

        B, T, H, W, C = lstm_input_reshape.get_shape().as_list()
        # split conv input into two parts 
        encoder_input_from_conv = tf.slice(lstm_input_reshape,[0,0,0,0,0],[B,self.enc_timesteps,H,W,C])
        decoder_input_from_conv = tf.slice(lstm_input_reshape,[0,self.enc_timesteps,0,0,0],[B,self.dec_timesteps,H,W,C])

        filter_size = C
        kernel_size = [3,3]
        encoded_state = self.conv_lstm_encoder(H,W,filter_size,kernel_size,encoder_input_from_conv)

        decoder_output = self.conv_lstm_decoder(H,W,filter_size,kernel_size,decoder_input_from_conv,encoded_state)

        # pass through deconv layer
        B, T, H, W, C = decoder_output.get_shape().as_list()
        deconv_layer_input = tf.reshape(decoder_output,[-1,H, W, C])
        predication = self.deconv_layer(deconv_layer_input)

        _, H, W, C = predication.get_shape().as_list()
        model_output = tf.reshape(predication,[B,T,H,W,C])
        return model_output

    def loss(self):
        self.gdl_l2_loss = total_loss(self.model_output, self.outputs_exp)

    def optimize(self):
        train_step = tf.train.AdamOptimizer()
        self.optimizer = train_step.minimize(self.gdl_l2_loss)

    def images_summary(self):
        train_summary, val_summary, test_summary = [], [], []
        summary = [train_summary, val_summary]
        summary_name = ["train","val"]

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

        # testing phase ...
        time_sliced_images = tf.slice(self.inputs_test, [0, 0, 0, 0, 0],[self.number_of_images_to_show, 1, self.test_shape[0], self.test_shape[1], self.channels])
        time_sliced_images = tf.squeeze(time_sliced_images,[1])
        test_summary.append(tf.summary.image('test_input_images', time_sliced_images, self.number_of_images_to_show))

        for each_timestep in self.images_summary_timesteps:
            time_sliced_images = tf.slice(self.model_output_test, [0, each_timestep, 0, 0, 0],
                                          [self.number_of_images_to_show, 1, self.test_shape[0], self.test_shape[1], self.channels])
            time_sliced_images = tf.squeeze(time_sliced_images,[1])
            test_summary.append(tf.summary.image('test_output_images_step_' + str(each_timestep), time_sliced_images,
                                 self.number_of_images_to_show))


        self.train_summary_merged = tf.summary.merge(summary[0])
        self.val_summary_merged = tf.summary.merge(summary[1])
        self.test_summary_merged = tf.summary.merge(test_summary)

# ===============================================================

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


def is_correct_batch_shape(X_batch, y_batch, model, heigth, width, info="train"):
    # info can be {"train", "val"}
    if ((X_batch is None) or (X_batch.shape != (model.batch_size, model.timesteps+1, heigth, width, channels))):
        print ("Warning: skipping this " + info + " batch because of shape")
        print ("expected ",(model.batch_size, model.timesteps+1, heigth, width, channels))
        if X_batch is not None:
            print ("got  ", X_batch.shape)
        return False
    return True 

def test(sess, model, data, test_writer, test_step, is_store_output=False):
    
    batch_size, heigth, width = model.batch_size, custom_test_size[0], custom_test_size[1]
    enc_timesteps = model.enc_timesteps
    dec_timesteps = model.dec_timesteps
    timesteps = model.timesteps

    for X_batch, y_batch, file_names in data.get_custom_test_data():
        if not is_correct_batch_shape(X_batch, y_batch, model, custom_test_size[0], custom_test_size[1], "val"):
            print ("test batch is skipping ... ")            
            continue

        B, T, H, W, C = model.inputs_test.get_shape().as_list()
        outputs_exp = X_batch[:, -model.dec_timesteps:]
        input_data = np.zeros((B,T,H,W,C))
        # +1 because decoder initial frame we provide ... !
        input_data[:,:enc_timesteps+1] = X_batch[:,:enc_timesteps+1]
        predicated_output = np.zeros((B,dec_timesteps,H,W,C))
        for i in range(dec_timesteps):
            test_summary_merged, model_output_test = sess.run([model.test_summary_merged, 
                                                                          model.model_output_test], 
                                                            feed_dict={ model.inputs_test : input_data})
            test_writer.add_summary(test_summary_merged, test_step)
            test_step += 1
            predicated_output[:,i] = model_output_test[:,i]
            if i!=(dec_timesteps-1):
                    input_data[:,enc_timesteps+1+i] = model_output_test[:,i]

        if is_store_output:
            # image post processing is happening inside of store ... 
            # store 
            store_file_names_gen = data.frame_ext.generate_output_video(predicated_output, file_names, ext_add_to_file_name="_generated_large")
            store_file_names_exp = data.frame_ext.generate_output_video(outputs_exp, file_names, ext_add_to_file_name="_expected_large")
            speed = 1
            data.frame_ext.generate_gif_videos(store_file_names_gen,speed=speed)
            data.frame_ext.generate_gif_videos(store_file_names_exp,speed=speed)

    return test_step

def validation(sess, model, data, val_writer, val_step):
    loss = []
    batch_size, heigth, width = model.batch_size, model.H , model.W
    enc_timesteps = model.enc_timesteps
    dec_timesteps = model.dec_timesteps
    timesteps = model.timesteps

    for X_batch, y_batch, _ in data.val_next_batch():
        if not is_correct_batch_shape(X_batch, y_batch, model, model.H, model.W, "val"):
            print ("validation batch is skipping ... ")            
            continue
        
        B, T, H, W, C = model.inputs.get_shape().as_list()
        outputs_exp = X_batch[:, -model.dec_timesteps:]
        input_data = np.zeros((B,T,H,W,C))
        # +1 because decoder initial frame we provide ... !
        input_data[:,:enc_timesteps+1] = X_batch[:,:enc_timesteps+1]
        for i in range(dec_timesteps):
                # fetch loss also ...
                gdl_l2_loss, val_summary_merged, model_output = sess.run([model.gdl_l2_loss,model.val_summary_merged, 
                                                                          model.model_output], 
                                                                    feed_dict={ model.inputs : input_data,
                                                                        model.outputs_exp : outputs_exp
                                                                    })
                if i!=(dec_timesteps-1):
                    input_data[:,enc_timesteps+1+i] = model_output[:,i]


        # last step gdl loss only appended ... 
        # that's what we are expecting , other time its predicating intermediate frames and all others are zero ...
        loss.append(gdl_l2_loss)
        val_writer.add_summary(val_summary_merged, val_step)
        val_step += 1

    if len(loss)==0:
        return (val_step, float("inf"))
    return (val_step, sum(loss)/float(len(loss)))

def test_wrapper():
	with tf.Session() as sess:
		model = seq2seq_model()
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        sess.run(init)

        # clear logs !
        log_directory_creation(sess)

        batch_size, heigth, width = model.batch_size, model.H , model.W
        enc_timesteps = model.enc_timesteps
        dec_timesteps = model.dec_timesteps
        timesteps = model.timesteps

        data = datasets(batch_size=batch_size, height=heigth, width=width, 
                                custom_test_size=custom_test_size,time_frame=timesteps+1, interval=interval)

        test_count_iter = 0
        test_writer = tf.summary.FileWriter(log_dir_file_path + "test", sess.graph)
        test_count_iter = test(sess, model, data, test_writer, test_count_iter, is_store_output=True)
        test_writer.close()

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
                    if not is_correct_batch_shape(X_batch, y_batch, model, model.H, model.W, "train"):
                        # global step not increased !
                        continue

                    input_data = X_batch[:, :timesteps]
                    outputs_exp = X_batch[:, -dec_timesteps:]
                    _, train_summary_merged = sess.run([model.optimizer,model.train_summary_merged], 
                                                feed_dict={ model.inputs : input_data,
                                                    model.outputs_exp : outputs_exp
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
                    if global_step % test_model_iterations == 0:
                        test_count_iter = test(sess, model, data, test_writer, test_count_iter, is_store_output=False)

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
