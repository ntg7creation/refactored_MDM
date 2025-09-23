# dataset_api/dataset_registry.py

from dataset_api.gigahands.gigahands_interface import GigaHandsInterface
from dataset_api.dataset_interface import DatasetInterface
from dataset_api.ASL.asl_interface import ASLInterface
class DatasetInterfaceRegistry:
    registry = {
        "gigahands": GigaHandsInterface(),
        #  "gigahands": ASLInterface(),
        # "asl": ASLInterface(),
        # Add other datasets here
    }

    @classmethod
    def get(cls, dataset_name: str)-> DatasetInterface: 
        if dataset_name not in cls.registry:
            raise ValueError(f"No registered interface for dataset: {dataset_name}")
        return cls.registry[dataset_name]
