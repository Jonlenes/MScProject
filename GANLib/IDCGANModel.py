import abc
from abc import ABCMeta, abstractmethod

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.models import load_model


class IDCGANModel(abc.ABC):
    @classmethod
    def __init__(self):
        self._discriminator = self._build_new_discriminator(self)
        self._generator = self._build_new_generator(self)
        
        self._adversarial = None
        
        
    @abstractmethod
    def _build_new_discriminator(self): raise NotImplementedError
        
        
    @abstractmethod
    def _build_new_generator(self): raise NotImplementedError
        
    
    @classmethod
    def discriminator(self): return self._discriminator
    
    
    @classmethod
    def generator(self): return self._generator
    
    
    @classmethod
    def adversarial(self): return self._adversarial
    
    
    @classmethod
    def save_discriminator(self, path_save): self._discriminator.save(path_save)
    
    
    @classmethod
    def load_discriminator(self, path_save): self._discriminator = load_model(path_save)
    
    
    @classmethod
    def save_generator(self, path_save): self._generator.save(path_save)
    
    
    @classmethod
    def load_generator(self, path_save): self._generator = load_model(path_save)
    
    
    @classmethod
    def compile_discriminator(self, optimizer=RMSprop(lr=0.0002, decay=6e-8), loss='binary_crossentropy', metrics=['accuracy']):
        self._discriminator.trainable = True
        self._discriminator.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return self._discriminator

    
    @classmethod
    def compile_adversarial(self, optimizer=RMSprop(lr=0.0001, decay=3e-8), loss='binary_crossentropy', metrics=None):
        self._discriminator.trainable = False
        self._adversarial = Sequencial()
        self._adversarial.add(self._generator)
        self._adversarial.add(self._discriminator)
        self._adversarial.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return self._adversarial