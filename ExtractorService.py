from abc import ABCMeta, abstractmethod
import numpy as np

# This is the interface for the service (Implementor)
class ExtractorService(metaclass=ABCMeta):

    # Define uniform interface for the bridge pattern
    def __init__(self):
        pass

    @staticmethod
    def to_db(aValue):
        return 20*np.log10(aValue)

    @staticmethod
    def inverse_db(aValue):
        return 10**(aValue/10)

    @staticmethod
    def _twos_comp(val, bitwidth = 8):
        mask = 2**(bitwidth-1)
        return -(val & mask) + (val & ~mask)

    @abstractmethod
    def open(self, aFilePath):
        pass

    @abstractmethod
    def open_stream(self, aStreamLocation, mode = 0, bufferSize = 1):
        pass

    @abstractmethod
    def get_csi_symbols(self):
        pass

    @abstractmethod
    def get_receiver_count(self):
        pass

    @abstractmethod
    def get_transmitter_count(self):
        pass

    @abstractmethod
    def get_symbol_count(self):
        pass

    @abstractmethod
    def device_driver_type(self):
        pass

    @abstractmethod
    def device_subcarriers(self):
        pass

    @abstractmethod
    def get_full_RSSI(self):
        pass

    @abstractmethod
    def get_scaled_RSSI(self):
        pass

