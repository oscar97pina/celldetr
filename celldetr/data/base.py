import abc
class BaseCellCOCO:
    
    @property
    @abc.abstractmethod
    def num_classes(self):
        pass
        
    @property
    @abc.abstractmethod
    def class_names(self):
        pass
    
    @abc.abstractmethod
    def image_size(self, image_id=None, idx=None):
        pass

    @abc.abstractmethod
    def get_raw_image(self, image_id=None, idx=None):
        pass

def DetectionWrapper(base_class):
    class Detection(base_class):
        @property
        def num_classes(self):
            return 1

        @property
        def class_names(self):
            return ['nuclei']
        
        def __getitem__(self, idx):
            img, tgt = super(Detection, self).__getitem__(idx)
            for i in range(len(tgt)):
                tgt[i]['category_id'] = 1 if tgt[i]['category_id'] > 0 else tgt[i]['category_id']
            return img, tgt
    
    Detection.__name__ = 'Detection' + base_class.__name__
    return Detection