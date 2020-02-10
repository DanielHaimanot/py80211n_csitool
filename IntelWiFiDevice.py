import numpy as np
import functools as func
from Tools.ExtractorService import *
from BlockData.BlockMatrix import *
import struct
import os, time

class Intel_IWL5300_API():
    def __init__(self):
        self.deviceDriverName = 'INTEL_IWL5300'
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

    def open_stream(self, source):
        # Need to parse file here in stages using the tail -f method developed prior
        self.open(source)


    def __parse_symbol_file(self, aFilePath):
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

        cur = 0
        count = 0
        broken_perm = 0
        triangle = [1, 3, 6]
        self.deviceSymbolData = []

        while cur < (fileLength - 1):
            cur = cur + 3
            size = struct.unpack('>H', f.read(2))[0]  # Big Endian, uint16, 2 bytes for the payload size field
            code = struct.unpack('B', f.read(1))[0]  # uint8, only one byte for the code field

            if code == 187:
                bytes = f.read(size - 1)
                cur = cur + size - 1

                if len(bytes) != (size - 1):
                    f.close()
                    return False
            else:  # skip the rest
                f.seek(size - 1)
                cur = cur + size - 1
                pass

            if code == 187:
                count = count + 1

                ret = self.__unpack_symbol(bytes)
                perm = ret['perm']
                nrx = ret['Nrx']
                ntx = ret['Ntx']
                self.deviceSymbolData.append(ret)

                if sum(perm) != triangle[nrx - 1]:
                    if broken_perm == 0:
                        broken_perm = 1
                        print("WARN ONCE: Found CSI with NRX=", nrx, " and invalid perm=", perm)
                else:
                    csi = self.deviceSymbolData[count - 1]['csi']

                    colByTransmitter = lambda t, csi: [tuple(csi[0:t])] + colByTransmitter(t, csi[t:]) if len(
                        csi) != 0 else []
                    csi = colByTransmitter(ntx, csi)

                    srcList = lambda r, t, csi, cnt: [csi[cnt::r]] + srcList(r, t, csi, cnt + 1) if r != cnt else []
                    csi = srcList(nrx, ntx, csi, 0)

                    deTuple = lambda t: list(map(deTuple, t)) if isinstance(t, (list, tuple)) else t
                    csi = deTuple(csi)

                    # create numpy matrix
                    mtrx = np.zeros((30 * ntx, nrx), complex)

                    for n in range(len(csi)):
                        csi[n] = list(func.reduce(lambda x, y: x + y, csi[n]))

                    for n in range(nrx):
                        mtrx[:, perm[n] - 1] = csi[n]

                    mtrx = mtrx.reshape((30, ntx, nrx))
                    self.deviceSymbolData[count - 1]['csi'] = mtrx
                    self.deviceSymbolData[count - 1]['csi'] = self.__scale_csi_to_ref(self.deviceSymbolData[count - 1])

    def __unpack_symbol(self, bytes):
        timestamp_low = bytes[0] + (bytes[1] << 8) + (bytes[2] << 16) + (bytes[3] << 24)
        nrx = bytes[8]
        ntx = bytes[9]
        rssi_a = bytes[10]
        rssi_b = bytes[11]
        rssi_c = bytes[12]
        noise = -(bytes[13] & 2 ** (8 - 1)) + (bytes[13] & ~2 ** (8 - 1))  # Unsigned, two's complement
        agc = bytes[14]
        antenna_sel = bytes[15]
        length = bytes[16] + (bytes[17] << 8)
        fake_rate_n_flags = bytes[18] + (bytes[19] << 8)
        calc_len = int((30 * (nrx * ntx * 8 * 2 + 3) + 7) / 8)

        csi = []
        index = 0
        ptr = 20  # Starting offset for payload
        perm = []  # Premutation array

        if calc_len != length:
            print("Lengths don't match!")
            return False

        # Compute CSI values
        for i in range(30):
            index += 3
            remainder = index % 8
            for j in range(nrx * ntx):
                tmp1 = (bytes[ptr + int(index / 8)] >> remainder) & 255  # Only care about 1 byte
                tmp2 = (bytes[ptr + int(index / 8) + 1] << (
                8 - remainder)) & 255  # Python bit width isn't fixed!
                re = self._twos_comp(tmp1 | tmp2)

                tmp1 = (bytes[ptr + int(index / 8) + 1] >> remainder) & 255
                tmp2 = (bytes[ptr + int(index / 8) + 2] << (8 - remainder)) & 255
                img = self._twos_comp(tmp1 | tmp2)

                csi.append(complex(re, img))
                index += 16

        # Compute antenna premutation  array
        perm.append(((antenna_sel) & 0x3) + 1)
        perm.append(((antenna_sel >> 2) & 0x3) + 1)
        perm.append(((antenna_sel >> 4) & 0x3) + 1)

        csi_struct = {"timestamp_low": timestamp_low, "bfee_count": 0, "Nrx": nrx, "Ntx": ntx, "rssi_a": rssi_a,
                      "rssi_b": rssi_b, "rssi_c": rssi_c, "noise": noise, "agc": agc, "perm": perm,
                      "rate": fake_rate_n_flags, "csi": csi}

        return csi_struct

    def __convert_to_total_rss(self, symbol):
        rssi_mag = 0

        if symbol['rssi_a'] != 0:
            rssi_mag += self.inverse_db(symbol['rssi_a'])
        if symbol['rssi_b'] != 0:
            rssi_mag += self.inverse_db(symbol['rssi_b'])
        if symbol['rssi_c'] != 0:
            rssi_mag += self.inverse_db(symbol['rssi_c'])

        return 10*np.log10(rssi_mag) - 44 - symbol['agc']

    def __scale_csi_to_ref(self, parsedData): # Already a Numpy Array
        csi = parsedData['csi']
        csi_sqr = csi * np.conjugate(csi)
        csi_pwr = np.sum(csi_sqr)
        rssi_pwr = self.inverse_db(self.__convert_to_total_rss(parsedData))
        scale = rssi_pwr / (csi_pwr / 30)

        if parsedData['noise'] == -127:
            noise_db = -92
        else:
            noise_db = parsedData['noise']

        thermal_noise_pwr = self.inverse_db(noise_db)
        quant_error_pwr = scale * (parsedData['Nrx'] * parsedData['Ntx'])
        total_noise_pwr = thermal_noise_pwr + quant_error_pwr

        scaled = csi * np.sqrt(scale / total_noise_pwr)

        if parsedData['Ntx'] == 2:
            parsedData['csi'] = scaled * np.sqrt(2)
        elif parsedData['Ntx'] == 3:
            parsedData['csi'] = scaled * np.sqrt(self.inverse_db(4.5))
        else:
            parsedData['csi'] = scaled

        return parsedData['csi']

# The API implementation for Intel
class IntelDeviceService(ExtractorService):
    def __init__(self, aIntelDevice):
        super().__init__()
        self._mDriver = aIntelDevice
        self.streamPtr = 0
        self.streamMtrx = None

    def open(self, filePath):
        self._mDriver.open(filePath)

    def reset_stream(self):
        self.streamPtr = 0
        self.streamMtrx = None

    def open_stream(self, aStreamLocation, mode = 0, sampleRate = 1, bufferSize = 1):

        # mode = 0 --> REPLAY, mode = 1 --> LIVE streaming
        if mode == 1:
            return self._mDriver.open_stream(aStreamLocation)
        elif mode == 0:
            if self.streamMtrx == None:
                self._mDriver.open(aStreamLocation)
                self.streamMtrx = self.convert_to_csi_matrix()
            if self.streamPtr >= self.streamMtrx.shape[3]:
                return None

            start = self.streamPtr
            self.streamPtr = self.streamPtr + bufferSize
            time.sleep(1/sampleRate)
            return self.streamMtrx[:, :, :,  start:self.streamPtr:]
        else:
            raise NotImplementedError

    def device_driver_type(self):
        return self._mDriver.deviceDriverName

    def device_subcarriers(self):
        return self._mDriver.deviceSubCarriers

    def get_receiver_count(self):
        return self._mDriver.deviceSymbolData[0]['Nrx'] # Count must be the same for all symbols

    def get_transmitter_count(self):
        return self._mDriver.deviceSymbolData[0]['Ntx']

    def get_full_RSSI(self):
        rssi_a = []
        rssi_b = []
        rssi_c = []

        for i in self._mDriver.deviceSymbolData:
            rssi_a.append(i['rssi_a'])
            rssi_b.append(i['rssi_b'])
            rssi_c.append(i['rssi_c'])

        return [['rssi_a',rssi_a], ['rssi_b', rssi_b], ['rssi_c', rssi_c]]

    def get_scaled_RSSI(self):
        temp = []
        for i in self._mDriver.deviceSymbolData:
            scaled = self.__convert_to_total_rss(i)
            temp.append(scaled)
        return temp

    def get_symbol_count(self):
        return len(self._mDriver.deviceSymbolData)

    def get_symbol_agc(self, symbolIndex):
        return self._mDriver.deviceSymbolData[symbolIndex]['agc']

    def get_symbol_noise(self, symbolIndex):
        return self._mDriver.deviceSymbolData[symbolIndex]['noise']

    def get_symbol_rate(self, symbolIndex):
        return self._mDriver.deviceSymbolData[symbolIndex]['rate']

    def convert_to_csi_matrix(self):
        mtrx = csiMatrix(self.get_transmitter_count(), self.get_receiver_count(), self.device_subcarriers(), self.get_symbol_count())

        for syms in range(self.get_symbol_count()):
            tmp = self._mDriver.deviceSymbolData[syms]
            for t in range(tmp['Ntx']):
                for r in range(tmp['Nrx']):
                    mtrx[t, r, :, syms] = tmp['csi'][:, t, r]
        return mtrx

    def get_csi_symbols(self):
        tempList = []
        for sym in self._mDriver.deviceSymbolData:
            tempList.append(sym['csi'])
        return tempList

    def __convert_to_total_rss(self, symbol):
        rssi_mag = 0

        if symbol['rssi_a'] != 0:
            rssi_mag += self.InvDb(symbol['rssi_a'])
        if symbol['rssi_b'] != 0:
            rssi_mag += self.InvDb(symbol['rssi_b'])
        if symbol['rssi_c'] != 0:
            rssi_mag += self.InvDb(symbol['rssi_c'])

        return 10*np.log10(rssi_mag) - 44 - symbol['agc']

# test = IntelDeviceExtractor(Intel_IWL5300_API())
# test.Open('../../data/log.all_csi.6.7.  6')
# test.ConvertToCSIMatrix()

#print(test.)

# test = IntelDeviceExtractor()
# test.ParseSymbolFile('../../data/log.all_csi.6.7.6')

# print(test._deviceSymbolData)
# print(test.GetCSISymbols()[0].shape)
# print(test.__ScaleCSIToRef(test._deviceSymbolData[0]))
