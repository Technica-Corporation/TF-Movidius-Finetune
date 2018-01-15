![alt text](https://www.sewp.nasa.gov/images/ch_logos/ch_54/ch_54.png "Technica")

# Introduction

The purpose of this project is to serve as a workspace for developers interested in fine-tuning or applying transfer learning models defined in the Tensorflow Slim API using custom images converted to a TFRecord Format. This project is motivated by problems in converting models and our unexplained results when a Tensorflow model is converted to run on a Movidius™ Neural Compute Stick using the Tensorflow For Poets](https://github.com/googlecodelabs/tensorflow-for-poets-2) code lab and other resources officially provided on the Tensorflow repository.

We've encountered several issues with converting Tensorflow models to be imported using the Movidius API. Most of these issues can be alleviated by exporting an inference metagraph file with a batch_size of 1 in the Placeholder input. A remaining issue is using the metagraph file exported from the training script results in an undefined shape of the Placeholder (e.g. shape of [?, 224, 224, 3]) or a non-one batch size (e.g. [8, 224, 224, 3]) which is not compatible with the MVNCSDK Compile Tool. We found that exporting a separate metagraph using the freeze checkpoint tool to freeze the graph to constants allows us to export a compatible version of the model that successfully compiled via the Movidius Compile Tool. In general, we've found that the key is to define a new inference GraphDef structure without any training/evaluation operations, a new default Placeholder with batch-size 1, and an output_layer that interfaces with the Predictions end_point used often in the Slim API. We've tried doing a similar thing with the vanilla Tensorflow API, but have had issues so far when using the MVNC tool to compile the graphs defined in vanilla Tensorflow, the process errors out. We have reason to believe that the TF Slim API is the optimal way to define models to be compatible with Movidius.

~~When we have successfully converted Tensorflow to the Movidius, we’ve observed a significant accuracy dropoff running the converted model on the Movidius Neural Compute Stick. Our internal test sets have shown up to 19% drop on a binary classification problem on 3 color channel images. This is described in the Issues section below.~~


## Updates
+ (1/5/18) Added transfer learning to scripts as opposed to fine-tuning. User can define to select the last Logits/AuxLogits layer to train instead of the entire model using the --trainable_variables argument in train.py. (Note: this hasn't yielded any different results when porting to MVNC)
+ (1/15/18) The latest [NCSDK release (1.12.00.01)](https://github.com/movidius/ncsdk/releases/tag/v1.12.00.01) has bug fixes. We've tested the new NCSDK + tools and the issues we've seen with the MobileNet architecture are resolved. See their release bug fixes for more details. 

# Requirements
+ Dependencies listed in requirements.txt
+ Dataset directory of images to use stored in jpg format where each subfolder of the master folder represents an image class
+ Generated labels file from tfrecord conversion process

## To Do:
- [ ] Define a new custom CNN model using TF-Slim API and converting to Movidius model
- [ ] Investigate drop-off in accuracy when running the converted model on the Movidius Neural Compute Stick (MobileNet Arch.)
- [ ] Look into the Movidius' TensorflowParser.py and look into compatibility with models defined in native Tensorflow
- [ ] Investigate is_training parameter (related to batch normalization) - odd behavior displayed when running eval scripts with different batch size, even though normalization should be the same with is_training=False
- [ ] Try different model architecture outside of MobileNet

## Example Workflow

Current workflow looks like this:
1. Convert data to tfrecord format using preprocess_img_dir/create_tfrecord.py
2. Use train.py to train on train split of data
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
python train.py --dataset_dir=data/processed/tfrecord/ --labels_file=labels.txt --num_epochs 15 --image_size 224 --checkpoint_path=./models/checkpoints/mobilenet_v1_1.0_224.ckpt --checkpoint_exclude_scopes="MobilenetV1/Logits, MobilenetV1/AuxLogits" --log_dir=/tmp/test/ --batch_size 16 --preprocessing inception --model mobilenet_v1
```
```
python eval.py --checkpoint_path ../tf-train-slim/trainlogs/1.2/run_4/ --num_classes 2 --labels_file labels.txt --dataset_dir ~/data/ --file_pattern data_%s_*.tfrecord --file_pattern_for_counting data --batch_size 2
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


## Issues
~~1. We've observed a significant accuracy dropoff on our internal test sets when converting from Tensorflow to the Movidius API (up to 19% drop on a binary classification problem on 3 color channel images). As of now, we're unsure if it's an issue with running in single precision, or something incorrect with the conversion process, or the model architecture we're using (MobileNet 1.0 224).
	+ We've used the built in inspect_checkpoint function in Tensorflow to verify that the checkpoint files are the same as the ones originally downloaded / used in the ncappzoo.
	+ In all our tests whether the graph has been loaded from a checkpoint (meta, data, index files) or frozen for inference (protobuf format), accuracy is stable on our validation set.
	+ We've validated that the graph structure is the same as some of the Mobilenet examples provided in the NCAppZoo using the [MVNCProfile](https://github.com/movidius/ncsdk/blob/master/docs/tools/profile.md) tool
	+ Our suspicion is it's something with the weights that the parser isn't translating 1-to-1.
	+ We've also looked into ensuring the preprocessing is similar from train/test time, so we've only done simple scaling [0, 1] and mean subtraction.~~
~~2. Export inference graph not working using model checkpoint and InceptionV3 architecture (works for MobileNet)~~

### Solutions Tried w/o Success
Here's a few solutions we've tried out without much success
+ Using the retrain script in the Tensorflow for poets tutorial to produce a frozen model and convert to Movidius
+ Created a simple 1-layer CNN defined in Native Tensorflow trained on MNIST and converted to Movidius, saw a drop from 95% accuracy to 80% accuracy on the test set


## Credit
Credits to Kwotsin's projects for serving as basis to navigate around TF-Slim API. Available [here](https://github.com/kwotsin/create_tfrecords) and [here](https://github.com/kwotsin/transfer_learning_tutorial). For the most we're using code pulled from his projects and the slim folder in the Tensorflow [models repo](https://github.com/tensorflow/models/), made some personal taste modifications and necessary changes to make it exportable to the Movidius NCS. Also the various documentation the Movidius team has [provided](https://github.com/movidius/ncsdk).
