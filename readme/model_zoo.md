# Model Zoos

We provide the following pre-trained SiamMOT models.

The name follows the convention `SiamMOT`-`BACKBONE`-`MOTION_MODEL`

### Airbone Tracking Dataset (AOT)

[SiamMOT-DLA34-EMM](https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/model_zoos/DLA-34-FPN_box_track_aot_d4.pth)

### MOTChallenge-2017 Test (Public detection)
More models and results are to be released soon

| Model    |  Training data | MOTA  | IDF1 |
|--------------|-----------|-------|--------|
|[SiamMOT-DLA34-EMM]() | CrowdHuman |   -   |  -
|[SiamMOT-DLA34-EMM]() | CrowdHuman + MOT17 |   -   |  -
|[SiamMOT-DLA34-IMM]() | CrowdHuman  |  -  |   - 
|[SiamMOT-DLA34-IMM]() | CrowdHuman + MOT17 |  -  |   - 


### TAO-Person
More models and results are to be released soon


## Pre-trained Faster-RCNN on COCO-2017
The following models (Faster-RCNN with FPN) can be used to initialize SiamMOT, and they are pre-trained on COCO-2017 80-class object detection dataset (training split).
The following table summarize their results in COCO-2017 80-class validation set.

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

