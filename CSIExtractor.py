from abc import ABCMeta, abstractmethod
from Tools.ExtractorDeviceFactory import *
#from plot_test import *

# ------ This is the code for the bridge pattern below ------
class DeviceExtractorBase( metaclass = ABCMeta ): # Abstraction
    def __init__(self, aExtractorService):
        self._mExtractor = aExtractorService
        self._mSymbolData = None

    def receiver_antenna_count(self):
        self._mExtractor.get_receiver_count()

    def transmitter_antenna_count(self):
        self._mExtractor.get_transmitter_count()

    def symbol_count(self):
        self._mExtractor.get_symbol_count()

class CSIExtractor(DeviceExtractorBase):
    def __init__(self, aExtractorService):
        super().__init__(aExtractorService)

    def sub_carrier_count(self):
        return self._mExtractor.device_subcarriers()

    def get_csi(self):
        return self._mExtractor.get_csi_symbols()

    def open_csi_file(self, aFilePath):
        self._mExtractor.open(aFilePath)

    def open_stream(self, aStreamLocation, mode = 0, bufferSize = 1):
        return self._mExtractor.open_stream(aStreamLocation, mode, bufferSize)

    def convert_to_csi_matrix(self):
        return self._mExtractor.convert_to_csi_matrix()

class RSSIExtractor(DeviceExtractorBase):
    def __init__(self, aExtractorService):
        super().__init__(aExtractorService)

    def get_rssi(self):
        self._mExtractor.get_full_RSSI()

    def get_scaled_rssi(self):
        self._mExtractor.get_scaled_RSSI()

    def open_rssi_file(self, aFilePath):
        self._mExtractor.ParseSymbolFile(aFilePath)
"""
deviceObjects = WiFiDeviceFactory()
print(deviceObjects)

intel = deviceObjects.CreateDevice('intel')
test = CSIExtractor(intel)

test.OpenCSIFile('../../data/log.all_csi.6.7.6')
csi = test.GetCSI()
elem = csi[28]['csi']
test.ConvertToCSIMatrix()

print('receiver:', test.ReceiverAntennaCount(), ' Transmitter:', test.TransmitterAntennaCount())

# convert to numpy matrix data structure to be used in all processing
temp = np.zeros((3,3,30,2500), complex)
for y in range(3): # transmitter
    for x in range(3): # receiver
        temp[y,x,:,0] = elem[:,y,x]

temp = temp[..., np.newaxis]
#print(temp.shape)
new = np.zeros((3,3,30), complex)

#print(temp[:,:,:,0])
#print(elem[:,0,0])

#print(np.reshape(elem, (1,3,30)))

# Atheros test is here
# test = CSIExtractor(deviceObjects.CreateDevice('atheros'))
# plot = CSI_Plot()

#Continous wavelet transform
"""
"""
from scipy import signal
import matplotlib.pyplot as plt
t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
print(sig.shape)
widths = np.arange(6, 9)
cwtmatr = signal.cwt(sig, signal.ricker, widths)
print(cwtmatr.shape)
plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()
"""
