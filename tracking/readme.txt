tensorflow 1.14
tensorflow model: https://github.com/tensorflow/models
labelImg: https://github.com/tzutalin/labelImg
xml_to_csv: https://github.com/datitran/raccoon_dataset

save_images.py: get images to be labelled

labelImg.py: label images

xml_to_csv.py: convert xml labels to csv

crop_img.py: returns coordinates to the bounding box locations of inputted png

python generate_tfrecord.py --csv_input={INSERT TEST CSV DIRECTORY}/ryu.csv --output_path={INSERT TEST DATA DIRECTORY}/test.record --image_dir={INSERT IMAGE DIRECTORY}

python generate_tfrecord.py --csv_input={INSERT TRAIN CSV DIRECTORY}/ryu.csv --output_path={INSERT TRAINING DATA DIRECTORY}/train.record --image_dir={INSERT IMAGE DIRECTORY}

python train.py --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config --logtostderr

python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt-{INSERT CHECKPOINT NUMBER} --output_directory new_graph

With Street Fighter 3 open
custom_model.py: detect characters and place bounding box over them and classify moves being performed
