# Dataset preparation

In order to reproduce the results in the paper smoothly, the dataset needs to be ingested correctly. Here we identify two modality of datasets

## Video dataset
We use class [GluonCVMotionDataset](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/torch/data/gluoncv_motion_dataset) to
represent each video-based dataset, for example MOT17, TAO-person, AOT, etc.

In order to ingest the original video dataset to *GluonCVMotionDataset* format, we provide ingestion scripts in [data/ingestion](../siammot/data/ingestion) folder, 
please follow the examples to ingest the video datasets. 

To ingest your own dataset, organize it in the following structure:
~~~
${dataset_root}
| -- raw_data
~~~
All artifacts related to the raw dataset are put to the *raw_data* folder. 
After ingestion, the dataset structure is expected to be like the following:
~~~
${dataset_root}
|-- annotation
  |-- anno.json
  |-- splits.json
|-- cache
|-- raw_data
~~~

We also provide the following ingested datasets (`anno.json` and `splits.json` files are provided),
Please make sure that `cfg.DATASETS.ROOT_DIR` in the [configuration](../configs/) has been pointed to `dataset_root`.
* MOT17: ` MOT17 videos with all 3 set of detections (DPM, FRCNN, SDP)` [Ingested annotation](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/ingested_dataset/MOT17/annotation_mot17.zip)
* MOT17_DPM: `MOT17 videos with DPM detection` [Ingested annotation](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/ingested_dataset/MOT17/annotation_mot17.zip)
* TAO: `TAO-person dataset`
* CRP: `Caltech Roadside Pedestrains dataset`
* AOT: `AOT dataset for airbone object detection and tracking`

In order to train with the above ingested datatset, the raw videos need to be downloaded the original data page, and extract them into `raw_data` folder. 


## Image dataset
We use class [COCO](https://github.com/cocodataset/cocoapi) to represent each image-based dataset, for example, COCO17, CrowdHuman.
Please follow the [example](https://github.com/cocodataset/cocoapi) to ingest the image-based datasets.

We provide the following ingested person detection datasets:
* COCO17_person_train [Ingested annotation](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/ingested_dataset/MSCOCO2017_train_person.json)
* CrowdHuman_fbox_train [Ingested annotation](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/ingested_dataset/CrowdHuman_train_fbox.json)
* CrowdHuman_vbox_train [Ingested annotation](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/ingested_dataset/CrowdHuman_train_vbox.json)

