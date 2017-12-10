# DeepVideos [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

DeepVideos is a tool to do future video frame prediction by looking at current frames!

> More detailed description can be found at  [https://team-pragmatic-chaos.github.io](https://team-pragmatic-chaos.github.io)

- - - -

## Applications
* Autonomous Driving
* Slowing down video motion, by inserting more frames between existing frames
* Predicting blurred frames in video
* View Sythesis

- - - -

## Datasets
* [UCF 101](http://crcv.ucf.edu/data/UCF101.php)
* [KITTI](http://www.cvlibs.net/datasets/kitti/)

- - - -

## Code orgnization

>datasets    

The dataset contains frame extraction logic from raw videos and generating sequences which can be consumed by model training, testing and validation phases. It also contains the logic related to storing generated video and making gifs. Take a look at `frame_extraction.py` and `batch_generator.py`

>model    

We tried several models and each file has a single model in itself. 

* `cell.py` Code related to ConvLSTM cell.
* `model_GAN.py` Multi-Scale architecture model which predict 4 future frame, and trained using GAN (**Works best**).
* `model_GAN_8.py` Same as above model but for 8 frame prediction (setiings are different, code is same) (**Works best**).
* `model_batch_norm_teacher_conv_lstm_deconv.py` Teacher Forcing removed [more explantion here](https://www.quora.com/What-is-the-teacher-forcing-in-RNN) (at training time, with certain probability give previous input to next input) and Batch normalization tried with increased conv and deconv layers.
* `model_eval.py` Evaluation script to evaluate each single model.
* `model_multiscale_architecture.py` Multi scale model without GAN, (just generator).
* `model_seq2seq.py` Seq2Seq model takes 4 frames as input and tries to predict 4 frames.
* `model_skip_autoencoder.py` Autoencoder with skip connection, while encoding take previous frames and while decoding it tries to predict net frame (**Works good**).
* `model_teacher_conv_lstm_deconv.py` teacher forcing removed while training conv lstm deconv model.

* `model_conv_lstm.py` Vanilla LSTM ([Conv-LSTM](https://arxiv.org/abs/1506.04214)) to predict future frames based on single frame (this is different problem than above model where we predict 4 frame output by taking input of 4 frame)
* `model_conv_lstm_deconv.py` Conv and Deconv layer with LSTM cell ([Conv-LSTM](https://arxiv.org/abs/1506.04214)).

>notebooks

These notebooks were created while building the Tensorflow model to try out different things and check dimension at each step.

>tests

Dataset batch generation logic checking. **Nothing to do with mode train and test**

- - - -

For rest of details visit : [https://team-pragmatic-chaos.github.io](https://team-pragmatic-chaos.github.io)
