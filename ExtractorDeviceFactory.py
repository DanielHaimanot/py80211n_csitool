from Tools.IntelWiFiDevice import *
from Tools.AtherosWiFiDevice import *

class WiFiDeviceFactory():

    @staticmethod
    def create_device(aDeviceName):
        if aDeviceName.lower() == 'iwl5300':
            return IntelDeviceService(Intel_IWL5300_API())
        elif aDeviceName.lower() == 'ath9k':
            return AtherosDeviceService(Atheros_ATH9K_API())
        else:
            raise NotImplementedError('The requested device class has not been implemented!')

#test = WiFiDeviceFactory.CreateDevice('ath9k')
#test.Open('../../data/ath_data.dat')