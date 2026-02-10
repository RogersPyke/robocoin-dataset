class MetadataUtils:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def get_metadata(self):
        return self.metadata
    
    def extract_to_yaml(self):
        return self.metadata