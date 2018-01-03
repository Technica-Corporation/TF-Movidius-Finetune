![alt text](https://www.sewp.nasa.gov/images/ch_logos/ch_54/ch_54.png "Technica")

# Introduction

Workspace to leverage own data and use it to fine-tune a model defined in TF-Slim from a pre-existing model checkpoint. Provides necessary functions to then export the model to a Movidius Compatible format.

We've encounted that there are several issues with porting Tensorflow models to the Movidius API, most of which can be alleviated by exporting a inference metagraph file with a batch_size of 1 in the Placeholder input. Previous issues using the metagraph file exporting from the training script due to undefined shape of the Placeholder (e.g. shape of [?, 224, 224, 3]) or a non-one batch size (e.g. [8, 224, 224, 3]). Exporting a seperate metagraph then using the freeze checkpoint tool to freeze the graph to consts allows to export a compatible version of the model that succesfully exportings via the Movidius Compile Tool. We've succesfully exported these graphs, which we've had previous problems doing for other models we've trained using Tensorflow for [poets 2 code lab](https://github.com/googlecodelabs/tensorflow-for-poets-2). In general, we've found that the key is to define a new model structure without any training/evaluation operations with a new default Placeholder with batch-size 1 and a output_layer that interfaces with the Predictions end_point used often in the Slim API.

We've tried doing a similar thing in vanilla Tensorflow, but have had issues so far.

# Requirements
+ Dependencies listed in requirements.txt
+ Dataset directory of images to use stored in jpg format where each subfolder of the master folder represents an image class
+ Generatd labels file from tfrecord conversion process

## To Do:
- [ ] Define custom CNN model using TF-Slim API and converting to Movidius
- [ ] Investigate issues with accuracy
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

## Issues
+ We've observed a significant accuracy dropoff on our internal test sets when converting from Tensorflow to the Movidius API (up to 19% drop on a binary classification problem on 3 color channel images). As of now, we're unsure if it's an issue with running in single precision, something incorrect with the conversion process, the model architecture we're using (MobileNet 1.0 224).
+ We've used the built in inspect_checkpoint function in Tensorflow to verify that the checkpoint files are the same as the ones originally downloaded / used in the ncappzoo.
+ In all our tests whether the graph has been loaded from a checkpoint (meta, data, index files) or frozen for inference (protobuf format), accuracy is stable on our validation set.
+ We've validated that the graph structure is the same as some of the Mobilenet examples provided in the NCAppZoo using the [MVNCProfile](https://github.com/movidius/ncsdk/blob/master/docs/tools/profile.md) tool
+ Our suspicion is that it's something with the weights that the parser isn't translating 1-to-1.
+ We've also looked into ensuring the preprocessing is similar from train/test time, so we've only done simple scaling [0, 1] and mean subtraction.


### Solutions Tried w/o Success
Here's a few solutions we've tried out without much success
+ Using the retrain script in the Tensorflow for poets tutorial to produce a frozen model and convert to movidius
+ Created a simple 1-layer CNN defined in Native Tensorflow and converted to Movidius, saw a drop from 95% accuracy to 80% accuracy


## Credit
Credits to Kwotsin's projects for serving as basis to navigate around TF-Slim API. Available [here](https://github.com/kwotsin/create_tfrecords) and [here](https://github.com/kwotsin/transfer_learning_tutorial). For the most we're using code pulled from his projects and the slim folder in the Tensorflow [models repo](https://github.com/tensorflow/models/), made some personal taste modifications and necessary changes to make it exportable to the Movidius NCS. Also the various documentation the Movidius team has [provided](https://github.com/movidius/ncsdk).
