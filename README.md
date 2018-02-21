![alt text](https://www.sewp.nasa.gov/images/ch_logos/ch_54/ch_54.png "Technica")

# Introduction

The purpose of this project is to serve as a workspace for developers interested in fine-tuning or applying transfer learning models defined in the Tensorflow Slim API using custom images converted to a TFRecord Format. This project was originally motivated by problems in converting models and our unexplained results when a Tensorflow model is converted to run on a Movidius™ Neural Compute Stick using the [Tensorflow For Poets](https://github.com/googlecodelabs/tensorflow-for-poets-2) code lab and other resources officially provided on the Tensorflow repository.

Following the creation of project, a brief [Tensorflow Compliance Guide](https://movidius.github.io/ncsdk/tf_compile_guidance.html) has been released by the Movidius team. 

The following are the biggest issues we've encountered we converting from Tensorflow to Movidius Graph:

## 1. Placeholder Input
Most models in Tensorflow are trained using NWHC formatted images; however networks are often defined by using input placeholders where N is None (e.g. [None, 224, 224,3]). The NCSDK Toolkit currently doesn't support a shape include a None dimension.

## 2. is_training
The is_training parameter when using the TF Slim API causes different behavior when training versus inferencing. Setting is_training to False removes Dropout nodes and sets batch normalization to use the static average as opposed to dynamically calculating over each new batch. 

## Solution
Using the checkpoints saved intermediately during training, one can create a inference version of the graph and use the Saver API to restore the weights. The inference version of the graph includes a fixed batch size for the Input Placeholder (e.g. [1,224,224,3]). The [export_inference_graph.py](export_inference_graph.py) provided by the Slim API, with some small modificiations fufills this functionality. Using this script also allows the user to define a meta graph with the is_training parameter set to false such that the other main issue is resolved. 

We believe that this should work in vanilla Tensorflow, but have found the slim API optimal for transfer/fine-tuning models. 

## Updates
+ (2/21/18) Change train script to use full Slim API (previously used monitoredtrainingsession+Slim API)
+ (1/5/18) Added transfer learning to scripts as opposed to fine-tuning. User can define to select the last Logits/AuxLogits layer to train instead of the entire model using the --trainable_variables argument in train.py. (Note: this hasn't yielded any different results when porting to MVNC)

# Requirements
+ Dependencies listed in requirements.txt
+ Dataset directory of images to use stored in jpg format where each subfolder of the master folder represents an image class
+ Generated labels file from tfrecord conversion process
+ [NCSDK v1.12](https://github.com/movidius/ncsdk/releases/tag/v1.12.00.01)
+ Tested Tensorflow Versions: 1.4, 1.5 with CUDA8/9 and CuDNN v6/v7, respectively

## To Do:
- [ ] Define a new custom CNN model using TF-Slim API and converting to Movidius model
- [ ] Test Transfer Learning in Vanilla Tensorflow
- [X] Investigate drop-off in accuracy when running the converted model on the Movidius Neural Compute Stick (MobileNet Arch.)
- [X] Look into the Movidius' TensorflowParser.py and look into compatibility with models defined in native Tensorflow
- [X] Investigate is_training parameter (related to batch normalization) - odd behavior displayed when running eval scripts with different batch size, even though normalization should be the same with is_training=False
- [X] Try different model architecture outside of MobileNet

## Example Workflow

Current workflow looks like this:
1. Convert data to tfrecord format using preprocess_img_dir/create_tfrecord.py
2. Use train_multi_gpu.py to train on train split of data
3. Use eval.py to evaluate on validation split of data
4. Iterate on steps 2-3 until desired loss/accuracy is achieved
5. Export inference graph of desired model architecture defined in nets folder
	a. We've used two different constructs available here, one is getting the inference graph then exporting it as a meta file along with the checkpoint using slim.restore_from_ckpt. The resulting meta file is sufficient to use with the graph tool
	b. The other way we've tested is exporting just the empty inference graph only using import_graph_def then freezing it using the tool
6. (If 5b) Freeze graph using inference graph metadef and desired training checkpoint
7. (If 5b) Retest frozen graph on subsample of images, make sure model still persists
8. Use either meta file or frozen graph and feed it in to [mvNCCompile](https://github.com/movidius/ncsdk/blob/master/docs/tools/compile.md) tool provided with Movidius to convert Tensorflow graph

### Sample Usage
Not going to go into too much detail of the usage, since this is more of a workspace so a good base knowledge of Tensorflow (& Slim API) should be sufficient for you to navigate around the scripts.

Convert -> Train -> Eval -> Export -> mvNCProfile/mvNCCheck/mvNCCompile


```
python train_multi_gpu.py --dataset_dir=data/processed/tfrecord/ --file_pattern name_%s_*.tfrecord --file_pattern_for_counting name --checkpoint_path=./models/checkpoints/mobilenet_v1_1.0_224.ckpt ---trainable_scopes="MobilenetV1/Logits" --checkpoint_exclude_scopes="MobilenetV1/Logits" --log_dir=/tmp/test/ --batch_size 16 --preprocessing_name inception --model mobilenet_v1 --max_number_of_steps 10000
```
```
python eval.py --checkpoint_path ../tf-train-slim/trainlogs/1.2/run_4/ --num_classes 2 --dataset_dir ~/data/ --file_pattern data_%s_*.tfrecord --file_pattern_for_counting data --batch_size 100
```
```
python export_inference_graph.py --model_name inception_v3 --image_size 224 --batch_size 1 --ckpt_path ./trainlogs//model.ckpt-1560 --output_ckpt_path ./output/tf-mvnc
```
```
python export_inference_graph.py --model_name inception_v3 --is_training True --batch_size 1 --num_classes 2 --output_file ./trainlogs/inference_graph.def --image_size 299
```
```
python freeze_graph.py --input_node_names input --output_node_names final_result --input_binary True --input_graph inference_graph.def --input_checkpoint test.ckpt --output_graph frozen-inf.pb
```


### Solutions Tried w/o Success
Here's a few solutions we've tried out without much success
+ Using the retrain script in the Tensorflow for poets tutorial to produce a frozen model and convert to Movidius

## Credit
Credits to Kwotsin's projects for serving as basis to navigate around TF-Slim API. Available [here](https://github.com/kwotsin/create_tfrecords) and [here](https://github.com/kwotsin/transfer_learning_tutorial). For the most we're using code pulled from his projects and the slim folder in the Tensorflow [models repo](https://github.com/tensorflow/models/), made some personal taste modifications and necessary changes to make it exportable to the Movidius NCS. Also the various documentation the Movidius team has [provided](https://github.com/movidius/ncsdk).
