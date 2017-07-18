import numpy as np
import functools as func
import matplotlib.pyplot as plt
from Tools.ExtractorService import *
from BlockData.BlockMatrix import *

import sys
import struct
import os

class Atheros_ATH9K_API():
    def __init__(self):
        self.deviceDriverName = 'ATH9K'
        self.deviceSubCarriers = 30
        self.deviceSymbolData = []

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

    def open(self, filePath):
        self.__parse_symbol_file(filePath)

    def open_stream(self, aStreamSource):
        raise NotImplementedError

    def __parse_symbol_file(self, aFilePath):
        self.deviceSymbolData = []
        try:
            f = open(aFilePath, "rb")
        except IOError as e:
            print(e.errno)
            print(e)
            return False

        # seek end of file
        f.seek(0, os.SEEK_END)
        fileLength = f.tell()
        f.seek(0, os.SEEK_SET)

        #print('file length is:', fileLength)
        endianCode = struct.unpack('B', f.read(1))[0]

        if endianCode == 255:
            eF = '>'
        elif endianCode == 0:
            eF = '<'
        else:
            print('Incorrect endian code:', endianCode)

        cur = 0

        while cur < (fileLength - 4):
            fieldLen = struct.unpack(eF + 'H', f.read(2))[0]
            cur = cur + 2

            if cur + fieldLen > fileLength:
                print('Exceeded file length')
                break

            timestamp = struct.unpack(eF + 'Q', f.read(8))[0]
            cur += 8
            csi_len = struct.unpack(eF + 'H', f.read(2))[0]
            cur += 2
            tx_channel = struct.unpack(eF + 'H', f.read(2))[0]
            cur += 2
            err_info = struct.unpack(eF + 'B', f.read(1))[0]
            cur += 1
            noise_floor = struct.unpack(eF + 'B', f.read(1))[0]
            cur += 1
            rate = struct.unpack(eF + 'B', f.read(1))[0]
            cur += 1
            bandWidth = struct.unpack(eF + 'B', f.read(1))[0]
            cur += 1
            num_tones = struct.unpack(eF + 'B', f.read(1))[0]
            cur += 1
            nr = struct.unpack(eF + 'B', f.read(1))[0]
            cur += 1
            nc = struct.unpack(eF + 'B', f.read(1))[0]
            cur += 1
            rssi = struct.unpack(eF + 'B', f.read(1))[0]
            cur += 1
            rssi1 = struct.unpack(eF + 'B', f.read(1))[0]
            cur += 1
            rssi2 = struct.unpack(eF + 'B', f.read(1))[0]
            cur += 1
            rssi3 = struct.unpack(eF + 'B', f.read(1))[0]
            cur += 1

            payload_len = struct.unpack(eF + 'H', f.read(2))[0]
            cur += 2

            self._deviceSubCarriers = num_tones
            csiSymbol = {'channel': tx_channel, 'err_info': err_info, 'noise_floor': noise_floor, 'rate': rate,
                         'bandwidth': bandWidth, 'num_tones': num_tones, 'nr': nr, 'nt': nc, 'rssi': rssi,
                         'rssi_1': rssi1, 'rssi_2': rssi2, 'rssi_3': rssi3}

            if csi_len > 0:
                csi_buf = f.read(csi_len)
                csi = self.__parse_to_symbol(csi_buf, nr, nc, num_tones)

                # @@@@ Place holder code
                csiSymbol['csi'] = csi
                # @@@@ Place holder code

                cur += csi_len
                # print('csi_values are:', csi)
            else:
                csiSymbol['csi'] = None

            if payload_len > 0:
                data_buf = f.read(payload_len)  # struct.unpack(eF+'B', f.read(payload_len))[0]
                cur += payload_len
                csiSymbol['payload'] = data_buf
            else:
                csiSymbol['payload'] = None

            self.deviceSymbolData.append(csiSymbol)
            # return

    def __parse_to_symbol(self, bytes, nr, nc, num_tones):
        # We process 16 bits at a time
        # 10 bit resolution for H real and imag
        bits_per_symbol = 10
        tone_40m = 114
        bits_per_byte = 8
        bits_per_complex_symbol = 2 * bits_per_symbol

        bitmask = ( 1 << bits_per_symbol ) - 1
        idx = 0
        h_idx = 0

        idx += 1
        h_data = bytes[idx]

        # not sure if this is right
        idx += 1
        h_data += (bytes[idx] << bits_per_byte) & 255
        current_data = h_data & ((1 << 16) - 1)

        # @@@ Finish parsing code here @@@
        # create empty place holder for now with random data in the correct format
        #mtrx = np.zeros((56, 2, 3), complex) # carriers, transmit, receiver
        mtrx = np.random.rand(56, 2, 3)
        # print(mtrx)
        return mtrx

    def __convert_to_total_RSS(self, symbol):
        rssi_mag = 0

        if symbol['rssi'] != 0:
            rssi_mag += self.inverse_db(symbol['rssi'])
        if symbol['rssi_1'] != 0:
            rssi_mag += self.inverse_db(symbol['rssi_1'])
        if symbol['rssi_2'] != 0:
            rssi_mag += self.inverse_db(symbol['rssi_2'])

        return 10 * np.log10(rssi_mag) - 44 - symbol['noise_floor']

# The API implementation for Atheros
class AtherosDeviceService(ExtractorService):
    def __init__(self, aDeviceAPI):
        super().__init__()
        self._mDriver = aDeviceAPI

    def device_driver_type(self):
        return self._mDriver.deviceDriverName

    def device_subcarriers(self):
        return self._mDriver.deviceSubCarriers

    def get_receiver_count(self):
        return self._mDriver.deviceSymbolData[0]['nr']

    def get_transmitter_count(self):
        return self._mDriver.deviceSymbolData[0]['nt']

    def get_full_RSSI(self):
        rssi = []
        rssi_1 = []
        rssi_2 = []
        for i in self._mDriver.deviceSymbolData:
            rssi.append(i['rssi'])
            rssi_1.append(i['rssi_1'])
            rssi_2.append(i['rssi_2'])
        return [['rssi', rssi], ['rssi_1', rssi_1], ['rssi_2', rssi_2]]

    def get_scaled_RSSI(self):
        temp = []
        for i in self._mDriver.deviceSymbolData:
            temp.append(self.___convert_to_total_rss(i))
        return temp

    def get_symbol_count(self):
        return len(self._mDriver.deviceSymbolData)

    def get_symbol_noise(self, symbolIndex):
        return self._mDriver.deviceSymbolData[symbolIndex]['noise_floor']

    def get_symbol_rate(self, symbolIndex):
        return self._mDriver.deviceSymbolData[symbolIndex]['rate']

    def get_csi_symbols(self):
        temp = []
        for i in self._mDriver.deviceSymbolData:
            temp.append(i['csi'])
        return temp

    def open(self, filepath):
        self._mDriver.open(filepath)

    def open_stream(self, aStreamSource):
        self._mDriver.open_stream(aStreamSource)

    def convert_to_csi_matrix(self):
        mtrx = csiMatrix(self.get_transmitter_count(), self.get_receiver_count(), self.device_subcarriers(), self.get_symbol_count())

        for syms in range(self.get_symbol_count()):
            tmp = self._mDriver.deviceSymbolData[syms]
            for t in range(tmp['Ntx']):
                for r in range(tmp['Nrx']):
                    mtrx[t, r, :, syms] = tmp['csi'][:, t, r]

    def ___convert_to_total_rss(self, symbol):
        rssi_mag = 0

        if symbol['rssi'] != 0:
            rssi_mag += self.InvDb(symbol['rssi'])
        if symbol['rssi_1'] != 0:
            rssi_mag += self.InvDb(symbol['rssi_1'])
        if symbol['rssi_2'] != 0:
            rssi_mag += self.InvDb(symbol['rssi_2'])

        return 10 * np.log10(rssi_mag) - 44 - symbol['noise_floor']

#test = AtherosDeviceExtractor()
#test.ExtractFromFile('../../data/ath_data.dat')

#print(len(test._deviceSymbolData))
#print(test.GetScaledRSSI())