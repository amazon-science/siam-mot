def get_model_name(cfg,
                   model_suffix=None,
                   is_train=True,
                  ):
    """
    Automatically generate a model name that carries key information about configuration;
    Those information includes the backbone, functionality (detection / tracking),
    trained dataset and any manually attached experiment identifier

    :param cfg:  experiment configuration file
    :param model_suffix: manually attached experiment identifier
    :param is_train: whether it is for training
    :return:
    """
    backbone = cfg.MODEL.BACKBONE.CONV_BODY
    branch_suffix = _get_branch_suffix(cfg)

    assert is_train is True, "This function is called only during training."
    dataset_list = cfg.DATASETS.TRAIN
    dataset_suffix = _get_dataset_suffix(dataset_list)

    output_dir = ""
    output_dir += backbone
    output_dir += branch_suffix
    output_dir += dataset_suffix
    if model_suffix is not None:
        if len(model_suffix) > 0:
            output_dir += ('_' + model_suffix)

    return output_dir


def _get_branch_suffix(cfg):
    suffix = ""
    if cfg.MODEL.BOX_ON:
        suffix += '_box'
    if cfg.MODEL.TRACK_ON:
        suffix += ('_'+cfg.MODEL.TRACK_HEAD.MODEL)
    return suffix


def _get_dataset_suffix(dataset_list):
    suffix = ""

    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError("dataset_list should be a list of strings, got {}".format(dataset_list))
    for dataset_key in dataset_list:
        suffix += ("_"+dataset_key)
    return suffix
