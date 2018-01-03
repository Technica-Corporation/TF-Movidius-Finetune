Workspace to leverage own data and use it to fine-tune a model defined in TF-Slim from a pre-existing model checkpoint. Provides necessary functions to then export the model to a Movidius Compatible format.

We've encounted that there are several issues with porting Tensorflow models to the Movidius API, most of which can be alleviated by exporting a inference metagraph file with a batch_size of 1 in the Placeholder input. Previous issues using the metagraph file exporting from the training script due to undefined shape of the Placeholder (e.g. shape of [?, 224, 224, 3]) or a non-one batch size (e.g. [8, 224, 224, 3]). Exporting a seperate metagraph then using the freeze checkpoint tool to freeze the graph to consts allows to export a compatible version of the model that succesfully exportings via the Movidius Compile Tool. We've succesfully exported these graphs, which we've had previous problems doing for other models we've trained using Tensorflow's (for poets 2 code lab)[https://github.com/googlecodelabs/tensorflow-for-poets-2].

We've tried doing a similar thing in vanilla Tensorflow, but have had issues so far.

To note, we've observed a significant accuracy dropoff on our internal test sets when converting from Tensorflow to the Movidius API (up to 19% drop on a binary classification problem). As of now, we're unsure if it's an issue with running in single precision, something incorrect with the conversion process, the model architecture we're using (MobileNet 1.0 224).

Things to Do:
- [ ] Define custom CNN model using TF-Slim API and converting to Movidius
- [ ] Investigate issues with accuracy
- [ ] Look into the Movidius' TensorflowParser.py and look into compatibility with models defined in native Tensorflow

Current workflow looks like this:
1. Convert data to tfrecord format using preprocess_img_dir/create_tfrecord.py
2. Use train.py to train on train split of data
3. Use eval.py to evaluate on validation split of data
4. Iterate on steps 2-3 until desired loss/accuracy is achieved
5. Export inference graph of desired model architecture defined in nets folder
6. Freeze graph using inference graph metadef and desired training checkpoint
7. Retest frozen graph on subsample of images, make sure model still persists
8. Use frozen pb file in conjunction with mvNCCompile tool provided with Movidius to convert Tensorflow graph

Credits to Kwotsin's projects for serving as basis to navigate around TF-Slim API. Available [here](https://github.com/kwotsin/create_tfrecords) and [here](https://github.com/kwotsin/transfer_learning_tutorial). For the most we're using code pulled from his projects and the slim folder in the Tensorflow (models repo)[https://github.com/tensorflow/models/], made some personal taste modifications and necessary changes to make it exportable to the Movidius NCS. This is not a complete project as it serves more of a workspace containing the necessary tools to fine-tune a imagenet trained model on a custom dataset. Internally we've only tested the Mobilenet model but we're expecting the other model architectures to work as well.

