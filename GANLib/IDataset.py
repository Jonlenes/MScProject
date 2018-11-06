import abc
from abc import ABCMeta, abstractmethod


class IDataset(abc.ABC):
    @classmethod
    def __init__(self, height=28, width=28, channel=1):
        self.height = height
        self.width = width
        self.channel = channel
        self.index = 0
        self.load_data(self)
        
    @abstractmethod
    def load_data(self): raise NotImplementedError
        
        
    @classmethod
    def __len__(self): 
        return len(self.x_train)
        
        
    @classmethod
    def iteration_to_begin(self):
        self.index = 0
    
    
    @classmethod
    def nexts(self, lot_size):
        last_index = self.index + lot_size
        if last_index >= self.__len__():
            if self.index < self.__len__():
                last_index = self.__len__() - 1
            else:
                raise Exception('EOF!')
        
        x = self.x_train[self.index:last_index]
        self.index += lot_size
        
        return x
    
    @classmethod
    def has_next(self):
        return self.index < self.__len__()