# TensorFlow Model !
import os
import shutil
import numpy as np
import tensorflow as tf
from cell import ConvLSTMCell
import sys
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
if module_path not in sys.path:
    sys.path.append(module_path)
from datasets.batch_generator import datasets

class conv_lstm_model():
    def __init__(self):
        # Run when your in trouble ... !
        tf.reset_default_graph()

        """Parameter initialization"""
        self.batch_size = 4
        self.timesteps = 32
        self.shape = [64, 64]  # Image shape
        self.kernel = [3, 3]
        self.channels = 3
        self.filters = [32, 128, 32, 3]  # 4 stacked conv lstm filters
        
        # Create a placeholder for videos.
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.timesteps] + self.shape + [self.channels])  # (batch_size, timestep, H, W, C)
        self.outputs_exp = tf.placeholder(tf.float32, [self.batch_size, self.timesteps] + self.shape + [self.channels])  # (batch_size, timestep, H, W, C)
        
        # model output
        self.model_output = None
        
        # loss
        self.l2_loss = None
        
        # optimizer
        self.optimizer = None
        
    def create_model(self):
        cells = []
        for i, each_filter in enumerate(self.filters):
            cell = ConvLSTMCell(self.shape, each_filter, self.kernel)
            cells.append(cell)
            
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)        
        states_series, current_state = tf.nn.dynamic_rnn(cell, self.inputs, dtype=self.inputs.dtype)
        # current_state => Not used ... 
        self.model_output = states_series
    
    def loss(self):
        frames_difference = tf.subtract(self.outputs_exp, self.model_output)
        batch_l2_loss = tf.nn.l2_loss(frames_difference)
        # divide by batch size ... 
        l2_loss = tf.divide(batch_l2_loss, float(self.batch_size))
        self.l2_loss = l2_loss
    
    def optimize(self):
        train_step = tf.train.AdamOptimizer().minimize(self.l2_loss)
        self.optimizer = train_step
        
    def build_model(self):
        self.create_model()
        self.loss()
        self.optimize()

file_path = os.path.abspath(os.path.dirname(__file__))
data_folder = os.path.join(file_path, "../../data/") 
log_dir_file_path = os.path.join(file_path, "../../logs/")
model_save_file_path = os.path.join(file_path, "../../checkpoint/")
output_video_save_file_path = os.path.join(file_path, "../../output/")
iterations = "iterations/"
best = "best/"
checkpoint_iterations = 25
best_model_iterations = 25
best_l2_loss = float("inf")
heigth, width = 64, 64
channels = 3

def log_directory_creation():
    if tf.gfile.Exists(log_dir_file_path):
        tf.gfile.DeleteRecursively(log_dir_file_path)
    tf.gfile.MakeDirs(log_dir_file_path)
    
    # model save directory
    if os.path.exists(model_save_file_path):
        shutil.rmtree(model_save_file_path)
    os.makedirs(model_save_file_path + iterations)
    os.makedirs(model_save_file_path + best)
    
def save_model_session(sess, file_name):
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_save_file_path + file_name + ".ckpt")
    
def restore_model_session(sess, file_name):
    saver = tf.train.Saver()
    saver.restore(sess, model_save_file_path + file_name + ".ckpt")

def is_correct_batch_shape(X_batch, y_batch, model, info="train"):
    # info can be {"train", "val"}
    if (X_batch is None or y_batch is None or 
        X_batch.shape!=(model.batch_size, model.timesteps,heigth,width,channels) or
        y_batch.shape!=(model.batch_size, model.timesteps,heigth,width,channels)):
            print ("Warning: skipping this " + info + " batch because of shape")
            return False
    return True

def test():
    log_directory_creation()
    
    # conv lstm model
    model = conv_lstm_model()
    model.build_model()
    
    # data read iterator
    data = datasets(batch_size=model.batch_size, heigth=heigth, width=width)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Start training
    sess = tf.InteractiveSession()
    sess.run(init)
    
    global_step = 0
    for X_batch, y_batch in data.train_next_batch():
        # print ("X_batch", X_batch.shape, "y_batch", y_batch.shape)
        if not is_correct_batch_shape(X_batch, y_batch, model, "test"):
            # global step not increased !
            continue

        input_data = np.zeros_like(X_batch)
        input_data[:,0] = X_batch[:,0]

        for i in range(model.timesteps):
            output_predicated = sess.run(model.model_output , feed_dict={ model.inputs: input_data })
            if (i < (model.timesteps-1)):
                input_data[:,i+1] = output_predicated[:,i]
                
        data.frame_ext.generate_output_video(output_predicated)
        
        global_step += 1
        print ("test step ",global_step)

    
def train():
    global best_l2_loss
    # clear logs !
    log_directory_creation()
    
    # conv lstm model
    model = conv_lstm_model()
    model.build_model()
    
    # data read iterator
    data = datasets(batch_size=model.batch_size, heigth=heigth, width=width)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Start training
    sess = tf.InteractiveSession()
    sess.run(init)
    
    # Tensorflow Summary
    tf.summary.scalar("train_l2_loss", model.l2_loss)
    summary_merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir_file_path + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(log_dir_file_path + "/test", sess.graph)

    global_step = 0
    for X_batch, y_batch in data.train_next_batch():
        # print ("X_batch", X_batch.shape, "y_batch", y_batch.shape)
        if not is_correct_batch_shape(X_batch, y_batch, model, "train"):
            # global step not increased !
            continue
        _, summary = sess.run([model.optimizer, summary_merged], feed_dict={
            model.inputs: X_batch, model.outputs_exp: y_batch})
        train_writer.add_summary(summary, global_step)
        
        if global_step % checkpoint_iterations == 0:
            save_model_session(sess, iterations + "conv_lstm_model")
        
        if global_step % best_model_iterations == 0:            
            val_l2_loss_history = list()
            batch_counter = 0
            # iterate on validation batch ...
            for X_val, y_val in data.val_next_batch():
                batch_counter += 1
                # print ("X_val", X_val.shape, "y_val", y_val.shape)
                if not is_correct_batch_shape(X_val, y_val, model, "val_"+str(batch_counter)):
                    continue
                test_summary, val_l2_loss = sess.run([summary_merged, model.l2_loss], feed_dict={model.inputs: X_val, model.outputs_exp: y_val})
                test_writer.add_summary(test_summary, global_step)
                val_l2_loss_history.append(val_l2_loss)
            temp_loss = sum(val_l2_loss_history) * 1.0 / len(val_l2_loss_history)
            
            # save if better !
            if best_l2_loss > temp_loss:
                best_l2_loss = temp_loss 
                save_model_session(sess, best + "conv_lstm_model")
            
        print ("Iteration ", global_step, " best_l2_loss ", best_l2_loss)
        global_step += 1

    train_writer.close()
    test_writer.close()

def main():
    test()

if __name__ == '__main__':
    main()
