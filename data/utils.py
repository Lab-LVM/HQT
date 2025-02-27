import importlib



def get_dataset(dataset_type, class_name="load_data", **kwargs):
    module = importlib.import_module(f"data.{dataset_type}")
    model = getattr(module, class_name)
    return model(**kwargs)