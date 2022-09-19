from .vot import VOTDataset, VOTLTDataset
from .otb import OTBDataset
from .uav import UAVDataset
from .lasot import LaSOTDataset
from .nfs import NFSDataset
from .trackingnet import TrackingNetDataset
from .got10k import GOT10kDataset
from .dtb import DTBDataset
from .uavdark70 import UAVDark70Dataset
#from .uamt100 import UAMT100Dataset
from .uamt20l import UAMT20LDataset
from .uavtrack112 import UAVTrack112Dataset
from .uav10 import UAV10Dataset
from .visdrone import VISDRONED2018Dataset
from .uav20l import UAV20Dataset
from .uavdt import UAVDTDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'OTB' in name:
            dataset = OTBDataset(**kwargs)
        elif 'LaSOT' == name:
            dataset = LaSOTDataset(**kwargs)
        elif 'NFS' in name:
            dataset = NFSDataset(**kwargs)
        elif 'VOT2018' == name or 'VOT2016' == name or 'VOT2019' == name:
            dataset = VOTDataset(**kwargs)
        elif 'DTB70' == name:
            dataset = DTBDataset(**kwargs)
        elif 'UAVDark70' == name:
            dataset = UAVDark70Dataset(**kwargs)
        elif 'VOT2018-LT' == name:
            dataset = VOTLTDataset(**kwargs)
        elif 'TrackingNet' == name:
            dataset = TrackingNetDataset(**kwargs)
        elif 'GOT-10k' == name:
            dataset = GOT10kDataset(**kwargs)
        elif 'UAMT100' == name:
            dataset = UAMT100Dataset(**kwargs)
        elif 'UAMT20L' == name:
            dataset = UAMT20LDataset(**kwargs)
        elif 'UAV10' == name:
            dataset = UAV10Dataset(**kwargs)
        elif 'UAV123' == name:
            dataset = UAVDataset(**kwargs)

        elif 'UAVTrack112' == name:
            dataset = UAVTrack112Dataset(**kwargs)
        elif 'Visdrone2018' == name:
            dataset = VISDRONED2018Dataset(**kwargs)
        elif 'UAVDT' == name:
            dataset = UAVDTDataset(**kwargs)
        elif 'UAV20l' == name:
            dataset = UAV20Dataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

