dataset_maps = dict()
"""
each item in the dataset maps are a list of the following info
(
dataset_folder, 
annotation file name (video dataset) / path of annotation file (image dataset), 
split file name (video dataset) / path of image folder (image dataset) , 
modality
)
"""

dataset_maps['TAO'] = ['TAO',
                       'anno_person.json',
                       'splits_person.json',
                       'video']

dataset_maps['CRP'] = ['caltech_roadside_pedestrians',
                       'anno.json',
                       'splits.json',
                       'video']

dataset_maps['MOT17_DPM'] = ['MOT17',
                             'anno.json',
                             'splits_DPM.json',
                             'video']

dataset_maps['MOT17'] = ['MOT17',
                         'anno.json',
                         'splits.json',
                         'video']

dataset_maps['AOT'] = ['airbone_object_tracking',
                       'anno.json',
                       'splits.json',
                       'video']

dataset_maps['COCO17_train'] = ['mscoco',
                                'annotations/MSCOCO2017_train_person.json',
                                'images/train2017',   # all raw images would be in dataset_root/mscoco/images/train2017
                                'image']

dataset_maps['crowdhuman_train_fbox'] = ['CrowdHuman',
                                         'annotations/annotation_train_fbox.json',
                                         'Images',  # all raw images would be in dataset_root/CrowdHuman/Images
                                         'image']

dataset_maps['crowdhuman_train_vbox'] = ['CrowdHuman',
                                         'annotations/annotation_train_vbox.json',
                                         'Images',
                                         'image']

dataset_maps['crowdhuman_val_fbox'] = ['CrowdHuman',
                                       'annotations/annotation_val_fbox.json',
                                       'Images',
                                       'image']

dataset_maps['crowdhuman_val_vbox'] = ['CrowdHuman',
                                       'annotations/annotation_val_vbox.json',
                                       'Images',
                                       'image']