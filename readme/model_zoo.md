# Model Zoos

We provide the following pre-trained SiamMOT models.

The name follows the convention `SiamMOT`-`BACKBONE`-`MOTION_MODEL`

## Airbone Object Tracking Challenge (AOT)
You can find the details of the challenge in the [offical homepage](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge/) hosted by AICrowd. 
Top methods will share the cash prize pool of $50,000.

The baseline results in the [AOT leaderboard](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge/leaderboards) are provided with this model: 
[SiamMOT-DLA34-EMM](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/model_zoos/DLA-34-FPN_EMM_AOT.pth)

## MOTChallenge-2017 Test (Public detection)
In order to run the model with the public detection, you need to 
1) set `INFERENCE.USE_GIVEN_DETECTION = True` in the configuration file

2) set `INPUT.AMODAL = True` in the configuration file (MOT17 uses amodal bounding box annotations)

3) ingest the public detection to [DataSample](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/torch/data/gluoncv_motion_dataset) object, 
we provide the ingested [public detection](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/ingested_dataset/MOT17/annotation_mot17.zip). 
After extraction, they should be placed under `dataset_root/annotation` folder.


|     Model    |        Training data       | MOTA  |  IDF1  |
|--------------|----------------------------|-------|--------|
|[SiamMOT-DLA34-EMM](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/model_zoos/DLA-34-FPN_EMM_crowdhuman.pth) | CrowdHuman         | 65.01 | 61.86
|[SiamMOT-DLA34-EMM](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/model_zoos/DLA-34-FPN_EMM_crowdhuman_mot17.pth) | CrowdHuman + MOT17 | 66.09 | 63.49
|SiamMOT-DLA34-IMM | CrowdHuman         |   -   |   - 
|SiamMOT-DLA34-IMM | CrowdHuman + MOT17 |   -   |   - 


## TAO-Person
Use the default [configuration file](../configs/dla/DLA_34_FPN_EMM.yaml) to generate the following results.

|         Model        |    Training data   | TAP@0.5  |  TAP@0.75 |
|----------------------|--------------------|----------|-----------|
|[SiamMOT-DLA34-EMM](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/model_zoos/DLA-34-FPN_EMM_coco_crowdhuman.pth) | CrowdHuman + COCO17|   37.98  |     19.99
|SiamMOT-DLA169-EMM| CrowdHuman + COCO17|     -    |     -


## Pre-trained Faster-RCNN on COCO-2017
The following models (Faster-RCNN with FPN) can be used to initialize SiamMOT, and they are pre-trained on COCO-2017 80-class object detection dataset (training split).
The following table summarize their results in COCO-2017 80-class validation set.

In order to initiate SiamMOT during training, download the corresponding model weight, and point its path to `MODEL.WEIGHT`
in the configuration file.

| Backbone     |  box-MAP  |
|--------------|-----------|
|[DLA-34](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/model_zoos/faster-rcnn/frcnn_dla34.pth)          |    35.9   |   
|[DLA-102](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/model_zoos/faster-rcnn/frcnn_dla102.pth)        |    38.3   |    
|[DLA-169](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/model_zoos/faster-rcnn/frcnn_dla169.pth)        |    39.8   | 
|[DLA-102-DCN](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/model_zoos/faster-rcnn/frcnn_dla102_dcn.pth)|    42.0   |
|[DLA-169-DCN](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/model_zoos/faster-rcnn/frcnn_dla169_dcn.pth)|    42.9   |
|ResNet-50     |    37.3   |      
|ResNet-101    |    39.5   |       
|ResNet-50-DCN |    40.6   |       
|ResNet-101-DCN|    43.0   |       

